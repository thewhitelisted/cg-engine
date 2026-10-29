"""Live matchup: where the current week stands and what to do about it (DECISIONS.md L1/L1b).

    (browser) web/espn_live.js -> data/live.json      # live category totals, all matchups this period
    python3 live.py                                   # your matchup now, projected final, streaming picks

Final weekly total = known so far (ESPN live, starters only) + the rest of the week: each remaining scheduled
game is played with probability a (games-played share x injury return path) at the player's blended per-game
rates. Model L4 (DECISIONS.md C1-C3) = L3 (near-term availability, daily 10-starter cap, over-dispersion psi on
categories won) plus game context: each game's expectation is scaled by the opponent's allowed factor (C1, from
this season's box scores, shrunk by games played) and by the minutes of teammates on a 2+ game absence (C3).
kappa_c 1.041, psi 0.350 (fit 2024-25; validated on 2025-26 and 2023-24). MODEL = "L3" restores the previous engine.
Remaining games: scheduled games after today, plus today's games that have no box score yet. Run after a
night's games finish: an in-progress game is counted both in the live totals and as remaining.
K1 lineup locks (DECISIONS.md, LINEUP_LOCKS): once a player's game has started today, his ESPN slot decides: benched
counts 0 for the rest of today; active counts his unplayed share and holds that slot, so today's not-yet-started
players are fitted into the remaining slots. Future days still assume the best lineup.
"""
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

import inseason as I
import model as M
from draft import LIVE_CONFIG

KAPPA_LIVE = 1.087                 # L1b (fallback)
MODEL = "L4"                       # DECISIONS.md C1-C3: L3 + opponent factor (C1) + teammate absences (C3)
RHO = 0.361                        # near-term availability (L2b), fit 2024-25
KAPPA_C, PSI = (1.041, 0.350) if MODEL == "L4" else (1.044, 0.350)   # category calibration, category dependence
OPP_K = {"pts": 40, "reb": 40, "ast": 20, "stl": 20, "blk": 40, "tpm": 40, "tov": 20, "fga": 40, "fta": 40, "fgx": 40}
ABS_BETA = {"pts": 0.082, "reb": 0.044, "ast": 0.075, "stl": 0.025, "blk": -0.067, "tpm": 0.035, "tov": 0.050,
            "fga": 0.066, "fta": 0.084}      # per 240 absent teammate minutes
CTX_STATS = ["pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fga", "fta"]
MC_CAP = 200
LINEUP_LOCKS = True                # DECISIONS.md K1 (registered 2026-10-25, approved by the user the same day)
SLOT_COLS = {0: (0,), 1: (1,), 2: (2,), 3: (3,), 4: (4,), 5: (5,), 6: (6,), 11: (7, 8, 9)}   # ESPN slot -> RL.SLOTS
LOCKED_OUT = (12, 13)              # bench, IR
MY_TEAM = 13
LIVE_FILE = M.DATA / "live.json"


def period_dates(season, period):
    wks = I.PERIOD_WEEKS[period]
    start = season.season_start + pd.Timedelta(days=7 * min(wks))
    return start, start + pd.Timedelta(days=7 * (max(wks) + 1 - min(wks)) - 1)


def remaining_games(season, g, period, today=None, from_tomorrow=False, today_frac=None):
    """Per player: scheduled games left in this matchup period (today's only if not played yet).
    today_frac: {pro_team_id: share of today's game still to play} from the live NBA scoreboard; when given it
    replaces the box-score check, so in-progress games count only their unplayed part."""
    today = I.game_day() if today is None else today
    _, end = period_dates(season, period)
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    first = today + pd.Timedelta(days=1) if from_tomorrow else today
    rem = sch[(sch.date >= first) & (sch.date <= end)]
    done_today = set()
    if not from_tomorrow and len(g):
        sp_today = sch.loc[sch.date == today, "scoring_period"].unique()
        done_today = set(g.loc[g.scoring_period.isin(sp_today), "id"])
    n = np.zeros(len(season.df))
    today_team = set(rem.loc[rem.date == today, "pro_team_id"])
    cnt = rem.groupby("pro_team_id").size()
    for i, (pid, tm) in enumerate(zip(season.df["id"], season.df["pro_team_id"])):
        n[i] = cnt.get(tm, 0)
        if from_tomorrow or tm not in today_team:
            continue
        if today_frac is not None:
            n[i] -= 1 - today_frac.get(tm, 1.0)
        elif pid in done_today:
            n[i] -= 1
    return n


def known_totals(tot):
    k = np.zeros(len(M.STATK))
    for s in ("fga", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        k[M.SK[s]] = tot.get(s, 0.0)
    k[M.SK["fgx"]] = tot.get("fgm", 0.0) - M.LG_FG * tot.get("fga", 0.0)
    k[M.SK["ftx"]] = tot.get("ftm", 0.0) - M.LG_FT * tot.get("fta", 0.0)
    return k


def final_dist(season, roster, known, n_rem, a):
    """Mean and variance of the final weekly totals for one roster (validated L1 structure)."""
    r = [p for p in roster if p >= 0]
    mu = season.df[M.STATK].to_numpy(float)[r]
    ea = (n_rem[r] * a[r])[:, None]
    m = known + (ea * mu).sum(0)
    v = (ea * M.per_game_var(mu) + (n_rem[r] * a[r] * (1 - a[r]))[:, None] * mu ** 2).sum(0)
    return m, np.maximum(v, 1e-9)


def win_week(pc):
    """P(win at least 5 of 9). L3: over-dispersed normal on categories won; L1b: independent Poisson-binomial."""
    if MODEL not in ("L3", "L4"):
        return float(M.Evaluator.matchup_win(pc))
    from scipy.stats import norm
    mu, var = pc.sum(), (pc * (1 - pc)).sum() * (1 + PSI)
    return float(1 - norm.cdf((4.5 - mu) / np.sqrt(max(var, 1e-9))))


def day_schedule(season, period, today=None, today_frac=None):
    """Remaining days of the matchup period: [(date, weight per player)]; weight = share of that day's game
    still to play (today: from the live scoreboard, or 1 if not started and no box score; later days: 1)."""
    today = I.game_day() if today is None else today
    _, end = period_dates(season, period)
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    rem = sch[(sch.date >= today) & (sch.date <= end)]
    team = season.df["pro_team_id"].to_numpy()
    out = []
    done = set()
    if today_frac is None:                         # no clocks: a box score today means the game is over
        g = I.gamelogs()
        sp = sch.loc[sch.date == today, "scoring_period"].unique()
        done = set(g.loc[g.scoring_period.isin(sp), "id"]) if len(g) else set()
    ids = season.df["id"].to_numpy()
    for d, day in rem.groupby("date"):
        w = np.isin(team, day["pro_team_id"].to_numpy()).astype(float)
        if d == today:
            if today_frac is not None:
                w *= np.array([today_frac.get(t, 1.0) for t in team])
            else:
                w[np.isin(ids, list(done))] = 0.0
        out.append((d, w))
    return out


def near_term_p(share, m, j, rho, S):
    """P(play the j-th next team game): Markov chain for healthy players (P(miss next | played) = rho (1 - pi),
    DECISIONS.md L2b), survival return path if absent."""
    pi = np.clip(share, 1e-3, 0.999)
    healthy = pi + (1 - pi) * (1 - rho) ** j
    mm = np.minimum(m, len(S) - 2)
    back = 1 - S[np.minimum(mm + j, len(S) - 1)] / np.maximum(S[mm], 1e-12)
    return np.where(m == 0, healthy, back * pi)


def opponent_factor(actual, expected, games):
    """C1: (multiplicative factors [stat], per-attempt fgx shift) for one opponent from its allowed totals so far."""
    f = np.ones(len(M.STATK))
    for s in CTX_STATS:
        w = games / (games + OPP_K[s]) if games > 0 else 0.0
        e = expected[M.SK[s]]
        f[M.SK[s]] = 1 + w * (actual[M.SK[s]] / e - 1) if e > 0 else 1.0
    w = games / (games + OPP_K["fgx"]) if games > 0 else 0.0
    shift = w * (actual[M.SK["fgx"]] - expected[M.SK["fgx"]]) / max(actual[M.SK["fga"]], 1e-9)
    return f, shift


def apply_context(mu, var, f, shift, x):
    """Per-game lines mu, var [n, stat] adjusted for the opponent (f [n, stat], shift [n]) and absent teammates'
    minutes x [n] (/ 240). With f = 1, shift = 0, x = 0 they come back unchanged."""
    base = mu.copy()
    m, v = mu * f, var * f
    m[:, M.SK["fgx"]] = base[:, M.SK["fgx"]] * f[:, M.SK["fga"]] + m[:, M.SK["fga"]] * shift
    m[:, M.SK["ftx"]] = base[:, M.SK["ftx"]] * f[:, M.SK["fta"]]
    pre = m.copy()
    for s in CTX_STATS:
        up = 1 + ABS_BETA[s] * x
        m[:, M.SK[s]] *= up
        v[:, M.SK[s]] *= up
    m[:, M.SK["fgx"]] = pre[:, M.SK["fgx"]] * (1 + ABS_BETA["fga"] * x)
    m[:, M.SK["ftx"]] = pre[:, M.SK["ftx"]] * (1 + ABS_BETA["fta"] * x)
    return m, v


def cat_probs(mA, vA, mB, vB):
    prev = M.MATCHUP_SD_KAPPA
    M.set_kappa(KAPPA_C if MODEL in ("L3", "L4") else KAPPA_LIVE)
    try:
        return M.Evaluator.cat_probs(mA, vA, mB, vB)
    finally:
        M.set_kappa(prev)


def started_teams(sb):
    """NBA team ids whose game today has started (live_server.scoreboard's flag), or None without a scoreboard."""
    if sb is None:
        return None
    return {t for t, v in sb.items() if len(v) > 4 and v[4]}


class Live:
    def __init__(self, live=None, today_frac=None, lg=None, started=None, slots=None, locks=None):
        """started: NBA team ids whose game today has started (started_teams); slots: {player id: ESPN slot} newer
        than lg's rosters (the live feed's); locks: K1 on/off (default LINEUP_LOCKS)."""
        M.set_kappa(LIVE_CONFIG["kappa"])
        self.lg = I.load_league() if lg is None else lg
        self.live = json.loads(LIVE_FILE.read_text()) if live is None else live
        self.g = I.gamelogs()
        self.locks = LINEUP_LOCKS if locks is None else locks
        if today_frac is None:                     # read tonight's game clocks so finished games aren't re-counted
            try:
                import live_server
                sb = live_server.scoreboard(I.game_day())
                today_frac, started = {k: v[1] for k, v in sb.items()}, started_teams(sb)
            except Exception:
                today_frac = None
        self.started = started
        self.season = I.inseason_season(M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"]), self.g)
        self.B = I.build(self.season, self.lg, self.g)
        self.period = self.live["period"]
        k = self.B["weeks_left"].index(self.period) if self.period in self.B["weeks_left"] else None
        A_now = self.B["A"][:, k] if k is not None else np.ones(len(self.season.df))
        self.a = np.clip(self.season.df["avail"].to_numpy() * A_now, 0, 1)
        self.n_rem = remaining_games(self.season, self.g, self.period, today_frac=today_frac)
        self.n_tmr = remaining_games(self.season, self.g, self.period, from_tomorrow=True)
        self.slot = {t: s for s, t in enumerate(self.B["tids"])}
        self.today_frac = today_frac
        self._set_slots({e[0]: e[2] for t in self.lg["teams"] for e in t["roster"]}, slots)
        if MODEL in ("L2", "L3", "L4"):
            self._l2_setup()

    def update(self, live, today_frac=None, started=None, slots=None):
        """Cheap refresh: new live totals, game clocks, started games and lineup slots; same players and model inputs."""
        self.live = live
        self.n_rem = remaining_games(self.season, self.g, self.period, today_frac=today_frac)
        self.today_frac, self.started = today_frac, started
        if slots:
            self._set_slots(slots)
        if MODEL in ("L2", "L3", "L4"):
            self._l2_setup()

    def _set_slots(self, *maps):
        """ESPN lineup slot per model player (-1 = not rostered); later maps override earlier ones."""
        idx = {int(i): k for k, i in enumerate(self.season.df["id"])}
        self.slot_now = np.full(len(self.season.df), -1, int)
        for m in maps:
            for pid, sl in (m or {}).items():
                if int(pid) in idx:
                    self.slot_now[idx[int(pid)]] = int(sl)

    def set_locks(self, on):
        """Switch K1 on or off (replays, shadow odds); clears the cached expected starts."""
        self.locks = bool(on)
        self._cache, self._daily = {}, {}

    # ---------------- L2: near-term availability + daily lineup cap
    def _l2_setup(self):
        import research_lineup as RL
        av = I.AVAIL
        self.p11, self.S = RHO, np.asarray(av["survival"])
        self.days = day_schedule(self.season, self.period, today_frac=self.today_frac)
        df = self.season.df
        self.elig = np.array([[RL.eligible(e, sl) for sl in RL.SLOTS] for e in df["elig"].to_numpy()], float)
        self.val, _ = M.static_value(df)
        self.m_now = np.asarray(self.B["info"]["m"])
        self.rest = np.asarray(self.B["info"].get("rest", np.zeros(len(df), bool)))    # S3: sits today's game
        self.share = df["avail"].to_numpy()
        self._cache, self._daily = {}, {}
        today = I.game_day()                        # K1: players whose game today has started
        self.lock_day = today if self.started is not None and self.days and self.days[0][0] == today else None
        self.lock_started = np.isin(df["pro_team_id"].to_numpy(), list(self.started or ()))
        if MODEL == "L4":
            self._context_setup()

    def _context_setup(self):
        """C1 opponent factors from this season's box scores before today; C3 absent-teammate minutes by NBA team."""
        df, g = self.season.df, self.g
        today = I.game_day()
        sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
        pair = sch.merge(sch, on="game_id", suffixes=("", "_o"))
        pair = pair[pair.pro_team_id != pair.pro_team_id_o]
        opp_sp = {(int(t), int(sp)): int(o) for t, sp, o in pair[["pro_team_id", "scoring_period", "pro_team_id_o"]].values}
        self.opp_day = {(int(t), d): int(o) for t, d, o in pair[["pro_team_id", "date", "pro_team_id_o"]].values}
        games = sch[sch.date < today].groupby("pro_team_id").size()
        idx = {int(i): k for k, i in enumerate(df["id"])}
        team = df["pro_team_id"].to_numpy()
        X = df[M.STATK].to_numpy(float)
        act, exp = {}, {}
        for r in g.itertuples(index=False) if len(g) else ():
            k = idx.get(int(r.id))
            o = None if k is None else opp_sp.get((int(team[k]), int(r.scoring_period)))
            if o is None:
                continue
            row = known_totals({s: getattr(r, s) for s in ("fga", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts",
                                                           "fgm", "ftm")})
            act[o], exp[o] = act.get(o, 0) + row, exp.get(o, 0) + X[k]
        self.opp_fac = {o: opponent_factor(act[o], exp[o], float(games.get(o, 0))) for o in act}
        absent = np.asarray(self.B["info"]["absent"])
        mins = g.groupby("id")["min"].mean() if len(g) else pd.Series(dtype=float)
        x = {}
        for k in np.where(absent)[0]:              # A1 absent, has played this season, projected >= 10 minutes
            pid = int(df["id"].iloc[k])
            if pid in mins.index and df["espn_min"].iloc[k] >= 10:
                x[int(team[k])] = x.get(int(team[k]), 0.0) + mins[pid] / 240
        self.abs_x = np.array([x.get(int(t), 0.0) for t in team])

    def expected_starts(self, roster, late=()):
        """Per player: expected counted games (starts) left this week, and their Bernoulli variance."""
        from scipy.optimize import linear_sum_assignment
        r = np.array([p for p in roster if p >= 0])
        key = (tuple(r), tuple(late))
        if key in self._cache:
            return self._cache[key]
        E, V = np.zeros(len(self.season.df)), np.zeros(len(self.season.df))
        daily = []
        j = np.zeros(len(r), int)
        rng = np.random.default_rng(11)
        for k, (d, w) in enumerate(self.days):
            wr = w[r].copy()
            if k == 0:
                wr[np.isin(r, list(late)) | self.rest[r]] = 0.0
            on = np.where(wr > 0)[0]
            if not len(on):
                continue
            j[on] += 1
            p = near_term_p(self.share[r[on]], self.m_now[r[on]], j[on], self.p11, self.S)
            fast = len(on) <= 10 and self.elig[r[on]].any(1).all()
            st = self.lock_started[r] if self.locks and k == 0 and d == self.lock_day else None
            if st is not None and st.any():
                if not fast:                            # draw what L4 would, so later days see the same numbers
                    rng.random((MC_CAP, len(on)))
                es = self._locked_starts(r, on, p, st)
            elif fast:
                es = p
            else:
                es = np.zeros(len(on))
                for dr in rng.random((MC_CAP, len(on))) < p:
                    pl = on[dr]
                    if not len(pl):
                        continue
                    Wt = self.elig[r[pl]] * (10 + self.val[r[pl]][:, None])
                    a_, b_ = linear_sum_assignment(-Wt)
                    started = set(r[pl][a_][self.elig[r[pl]][a_, b_] > 0])
                    es += np.array([r[x] in started for x in on])
                es /= MC_CAP
            E[r[on]] += es * wr[on]
            V[r[on]] += es * (1 - es) * wr[on] ** 2
            daily.append((d, r[on], es * wr[on], es * (1 - es) * wr[on] ** 2))
        self._cache[key], self._daily[key] = (E, V), daily
        return E, V

    def _locked_starts(self, r, on, p, st):
        """K1, today: expected starts for `on` (indices into r with a game share left) given which of r's games have
        started (st). Benched in a started game: 0. Active in a started game: plays if available, holding his slot
        (finished games too). Not started: best fit into the slots nobody holds."""
        from scipy.optimize import linear_sum_assignment
        from research_lineup import SLOTS
        slot = self.slot_now[r]
        held = st & np.isin(slot, list(SLOT_COLS))
        out = st & np.isin(slot, LOCKED_OUT)
        cols = list(range(len(SLOTS)))
        for s in slot[held]:
            free = [c for c in SLOT_COLS[int(s)] if c in cols]
            if free:
                cols.remove(free[0])
        es = np.where(held[on], p, 0.0)
        pool = np.where(~held[on] & ~out[on])[0]     # not started (or no known slot)
        if not len(pool) or not cols:
            return es
        el = self.elig[r[on[pool]]][:, cols]
        if len(pool) <= len(cols) and el.any(1).all():
            es[pool] = p[pool]
            return es
        acc = np.zeros(len(pool))
        for dr in np.random.default_rng(12).random((MC_CAP, len(pool))) < p[pool]:
            pl = np.where(dr)[0]
            if not len(pl):
                continue
            a_, b_ = linear_sum_assignment(-(el[pl] * (10 + self.val[r[on[pool[pl]]]][:, None])))
            acc[pl[a_][el[pl][a_, b_] > 0]] += 1
        es[pool] = acc / MC_CAP
        return es

    def games_left(self, tid, rosters=None):
        rost = self.B["rost"] if rosters is None else rosters
        r = [p for p in rost[self.slot[tid]] if p >= 0]
        if MODEL in ("L2", "L3", "L4"):
            return float(self.expected_starts(rost[self.slot[tid]])[0][r].sum())
        return float((self.n_rem * self.a)[r].sum())

    def matchup(self, home, away, rosters=None):
        m = next(x for x in self.live["matchups"] if x["home"] == home and x["away"] == away)
        rost = self.B["rost"] if rosters is None else rosters
        if MODEL in ("L2", "L3", "L4"):
            late = getattr(self, "late", ())
            mA, vA = self._final_l2(rost[self.slot[home]], known_totals(m["home_tot"]), late)
            mB, vB = self._final_l2(rost[self.slot[away]], known_totals(m["away_tot"]), late)
        else:
            mA, vA = final_dist(self.season, rost[self.slot[home]], known_totals(m["home_tot"]), self.n_rem, self.a)
            mB, vB = final_dist(self.season, rost[self.slot[away]], known_totals(m["away_tot"]), self.n_rem, self.a)
        pc = cat_probs(mA, vA, mB, vB)
        return pc, win_week(pc), (mA, mB), m

    def _final_l2(self, roster, known, late=()):
        if MODEL == "L4":
            return self._final_l4(roster, known, late)
        r = [p for p in roster if p >= 0]
        E, V = self.expected_starts(roster, late)
        mu = self.season.df[M.STATK].to_numpy(float)[r]
        m = known + (E[r][:, None] * mu).sum(0)
        v = (E[r][:, None] * M.per_game_var(mu) + V[r][:, None] * mu ** 2).sum(0)
        return m, np.maximum(v, 1e-9)

    def _final_l4(self, roster, known, late=()):
        """L3's expected starts, day by day, with each game's line adjusted for its opponent and absent teammates."""
        r = np.array([p for p in roster if p >= 0])
        self.expected_starts(roster, late)
        X = self.season.df[M.STATK].to_numpy(float)
        team = self.season.df["pro_team_id"].to_numpy()
        m, v = known.copy(), np.zeros(len(M.STATK))
        for d, pl, e, ve in self._daily[(tuple(r), tuple(late))]:
            mu = X[pl]
            f, shift = np.ones_like(mu), np.zeros(len(pl))
            for i, p in enumerate(pl):
                o = self.opp_day.get((int(team[p]), d))
                if o in self.opp_fac:
                    f[i], shift[i] = self.opp_fac[o]
            mu, var = apply_context(mu, M.per_game_var(mu), f, shift, self.abs_x[pl])
            m += (e[:, None] * mu).sum(0)
            v += (e[:, None] * var + ve[:, None] * mu ** 2).sum(0)
        return m, np.maximum(v, 1e-9)

    def week0_pc(self, pairs0, rosters=None):
        """Live category odds for each of this week's matchups, ordered like the season simulation's pairs."""
        out = []
        for h, a in pairs0:
            th, ta = self.B["tids"][h], self.B["tids"][a]
            try:
                pc, _, _, _ = self.matchup(th, ta, rosters)
            except StopIteration:
                pc, _, _, _ = self.matchup(ta, th, rosters)
                pc = 1 - pc
            out.append(pc)
        return np.array(out)


def week0_for(pairs, weeks_left):
    """Live category odds for the week in progress (season simulations), or None when live data is stale."""
    if not LIVE_FILE.exists() or not weeks_left:
        return None
    lv = Live()
    if lv.period != weeks_left[0]:
        return None
    return lv.week0_pc(pairs[0])


def show(lv):
    mine = next(x for x in lv.live["matchups"] if MY_TEAM in (x["home"], x["away"]))
    home = mine["home"] == MY_TEAM
    opp = mine["away"] if home else mine["home"]
    pc, pw, (mA, mB), m = lv.matchup(mine["home"], mine["away"])
    if not home:
        pc, pw, mA, mB = 1 - pc, 1 - pw, mB, mA
    kn_me, kn_op = known_totals(m["home_tot"] if home else m["away_tot"]), known_totals(m["away_tot"] if home else m["home_tot"])
    names = {t["id"]: t["name"] for t in lv.lg["teams"]}

    def fmt(k, cat):
        if cat in ("FG%", "FT%"):
            x, a = ("fgx", "fga") if cat == "FG%" else ("ftx", "fta")
            lg = M.LG_FG if cat == "FG%" else M.LG_FT
            return f"{(k[M.SK[x]] / k[M.SK[a]] + lg):.3f}" if k[M.SK[a]] > 0 else "—"
        s = {"3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "PTS": "pts"}[cat]
        return f"{k[M.SK[s]]:.0f}"

    me_slot, op_slot = lv.slot[MY_TEAM], lv.slot[opp]
    g_me, g_op = int(round(lv.games_left(MY_TEAM))), int(round(lv.games_left(opp)))
    lines = [f"# Live — matchup {lv.period} vs {names[opp]} ({date.today()})", "",
             f"**Chance to win the week: {pw:.0%}**. Expected games left: you {g_me}, them {g_op}.", "",
             "| Category | You now | Them now | Projected final (you / them) | P(you win) | |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(M.CATS):
        tag = "locked" if pc[i] > 0.97 else "lost" if pc[i] < 0.03 else "swing" if 0.25 <= pc[i] <= 0.75 else ""
        lines.append(f"| {c} | {fmt(kn_me, c)} | {fmt(kn_op, c)} | {fmt(mA, c)} / {fmt(mB, c)} | {pc[i]:.0%} | {tag} |")
    # streaming: adds from tomorrow, drop one of our three lowest-value players (position limits respected)
    df = lv.season.df
    val, _ = M.static_value(df)
    rost = lv.B["rost"]
    mine_p = [p for p in rost[me_slot] if p >= 0]
    drops = sorted(mine_p, key=lambda p: val[p])[:3]
    rostered = {e[0] for t in lv.lg["teams"] for e in t["roster"]}
    fa = [p for p in np.argsort(-val)[:250] if int(df["id"][p]) not in rostered and lv.n_tmr[p] > 0][:60]
    limits = lv.lg.get("position_limits") or {"C": 4}
    out = []
    for a_ in fa:
        for d_ in drops:
            r2 = rost.copy()
            r2[me_slot] = np.where(r2[me_slot] == d_, a_, r2[me_slot])
            pos = list(df["pos"].to_numpy()[[p for p in r2[me_slot] if p >= 0]])
            if any(pos.count(k) > v for k, v in limits.items()):
                continue
            n_save = lv.n_rem.copy()
            lv.n_rem[a_] = lv.n_tmr[a_]                         # an add plays from tomorrow
            lv.late = (a_,)
            _, pw2, _, _ = lv.matchup(mine["home"], mine["away"], r2)
            lv.n_rem, lv.late = n_save, ()
            pw2 = pw2 if home else 1 - pw2
            out.append((pw2 - pw, df.name[a_], df.name[d_], int(lv.n_tmr[a_]), int(lv.n_rem[d_])))
    out.sort(reverse=True)
    lines += ["", "## Streaming this week (adds play from tomorrow)", ""]
    good = [o for o in out if o[0] > 0.01][:6]
    if good:
        # season cost: the dropped player is gone for good (paired season simulation, weekly.py)
        import weekly as W
        cx = W.Ctx()
        lines += ["| Add | Drop | Δ P(win week) | Games left (add / drop) | Δ season $ | Verdict |", "|---|---|---|---|---|---|"]
        for dp, a, d, ga, gd in good:
            ai, di = df.index[df.name == a][0], df.index[df.name == d][0]
            r2 = cx.rost.copy()
            r2[cx.me][r2[cx.me] == di] = ai
            base_w0 = cx.week0
            n_save = lv.n_rem.copy()
            lv.n_rem[ai] = lv.n_tmr[ai]
            lv.late = (ai,)
            cx.week0 = lv.week0_pc(cx.pairs[0], r2)          # this week's live odds with the stream
            lv.n_rem, lv.late = n_save, ()
            res = cx.delta(r2, [cx.me])[cx.me]
            cx.week0 = base_w0
            verdict = "do it" if res["dE$"] > -2 * res["se"] else "not worth it"
            lines.append(f"| {a} | {d} | {dp:+.1%} | {ga} / {gd} | {res['dE$']:+.2f} ± {res['se']:.2f} | {verdict} |")
        lines.append("")
        lines.append("_Δ season $ already includes this week's gain: a stream is worth it only if the season total doesn't drop._")
    else:
        lines.append("No add raises this week's win chance by more than 1 point.")
    lines += ["", "_Swing = 25-75%: where streaming and lineup choices matter. Locked / lost = above 97% / below 3%._"]
    rep = "\n".join(lines)
    (M.DATA / "weekly").mkdir(exist_ok=True)
    (M.DATA / "weekly" / f"live_{date.today()}.md").write_text(rep)
    print(rep)


if __name__ == "__main__":
    show(Live())
