"""Live matchup: where the current week stands and what to do about it (DECISIONS.md L1/L1b).

    (browser) web/espn_live.js -> data/live.json      # live category totals, all matchups this period
    python3 live.py                                   # your matchup now, projected final, streaming picks

Final weekly total = known so far (ESPN live, starters only) + the rest of the week: each remaining scheduled
game is played with probability a (games-played share x injury return path) at the player's blended per-game
rates. Model L3 (DECISIONS.md): near-term availability, daily 10-starter cap, kappa_c 1.044 on categories and
over-dispersion psi 0.35 on categories won (fit 2024-25; validated on 2025-26 and 2023-24).
Remaining games: scheduled games after today, plus today's games that have no box score yet. Run after a
night's games finish: an in-progress game is counted both in the live totals and as remaining.
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
MODEL = "L3"                       # DECISIONS.md L3: near-term availability + daily lineup cap + separate
KAPPA_C, PSI, RHO = 1.044, 0.350, 0.361   # category calibration (kappa_c) and category dependence (psi); fit 2024-25
MC_CAP = 200
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
    today = pd.Timestamp(date.today()) if today is None else today
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
    if MODEL != "L3":
        return float(M.Evaluator.matchup_win(pc))
    from scipy.stats import norm
    mu, var = pc.sum(), (pc * (1 - pc)).sum() * (1 + PSI)
    return float(1 - norm.cdf((4.5 - mu) / np.sqrt(max(var, 1e-9))))


def day_schedule(season, period, today=None, today_frac=None):
    """Remaining days of the matchup period: [(date, weight per player)]; weight = share of that day's game
    still to play (today: from the live scoreboard, or 1 if not started and no box score; later days: 1)."""
    today = pd.Timestamp(date.today()) if today is None else today
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


def cat_probs(mA, vA, mB, vB):
    prev = M.MATCHUP_SD_KAPPA
    M.set_kappa(KAPPA_C if MODEL == "L3" else KAPPA_LIVE)
    try:
        return M.Evaluator.cat_probs(mA, vA, mB, vB)
    finally:
        M.set_kappa(prev)


class Live:
    def __init__(self, live=None, today_frac=None, lg=None):
        M.set_kappa(LIVE_CONFIG["kappa"])
        self.lg = I.load_league() if lg is None else lg
        self.live = json.loads(LIVE_FILE.read_text()) if live is None else live
        self.g = I.gamelogs()
        if today_frac is None:                     # read tonight's game clocks so finished games aren't re-counted
            try:
                from datetime import datetime
                import live_server
                today_frac = {k: v[1] for k, v in live_server.scoreboard(datetime.now()).items()}
            except Exception:
                today_frac = None
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
        if MODEL in ("L2", "L3"):
            self._l2_setup()

    def update(self, live, today_frac=None):
        """Cheap refresh: new live totals and game clocks, same players and model inputs."""
        self.live = live
        self.n_rem = remaining_games(self.season, self.g, self.period, today_frac=today_frac)
        self.today_frac = today_frac
        if MODEL in ("L2", "L3"):
            self._l2_setup()

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
        self.share = df["avail"].to_numpy()
        self._cache = {}

    def expected_starts(self, roster, late=()):
        """Per player: expected counted games (starts) left this week, and their Bernoulli variance."""
        from scipy.optimize import linear_sum_assignment
        r = np.array([p for p in roster if p >= 0])
        key = (tuple(r), tuple(late))
        if key in self._cache:
            return self._cache[key]
        E, V = np.zeros(len(self.season.df)), np.zeros(len(self.season.df))
        j = np.zeros(len(r), int)
        rng = np.random.default_rng(11)
        for k, (d, w) in enumerate(self.days):
            wr = w[r].copy()
            if k == 0:
                wr[np.isin(r, list(late))] = 0.0
            on = np.where(wr > 0)[0]
            if not len(on):
                continue
            j[on] += 1
            p = near_term_p(self.share[r[on]], self.m_now[r[on]], j[on], self.p11, self.S)
            if len(on) <= 10 and self.elig[r[on]].any(1).all():
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
        self._cache[key] = (E, V)
        return E, V

    def games_left(self, tid, rosters=None):
        rost = self.B["rost"] if rosters is None else rosters
        r = [p for p in rost[self.slot[tid]] if p >= 0]
        if MODEL in ("L2", "L3"):
            return float(self.expected_starts(rost[self.slot[tid]])[0][r].sum())
        return float((self.n_rem * self.a)[r].sum())

    def matchup(self, home, away, rosters=None):
        m = next(x for x in self.live["matchups"] if x["home"] == home and x["away"] == away)
        rost = self.B["rost"] if rosters is None else rosters
        if MODEL in ("L2", "L3"):
            late = getattr(self, "late", ())
            mA, vA = self._final_l2(rost[self.slot[home]], known_totals(m["home_tot"]), late)
            mB, vB = self._final_l2(rost[self.slot[away]], known_totals(m["away_tot"]), late)
        else:
            mA, vA = final_dist(self.season, rost[self.slot[home]], known_totals(m["home_tot"]), self.n_rem, self.a)
            mB, vB = final_dist(self.season, rost[self.slot[away]], known_totals(m["away_tot"]), self.n_rem, self.a)
        pc = cat_probs(mA, vA, mB, vB)
        return pc, win_week(pc), (mA, mB), m

    def _final_l2(self, roster, known, late=()):
        r = [p for p in roster if p >= 0]
        E, V = self.expected_starts(roster, late)
        mu = self.season.df[M.STATK].to_numpy(float)[r]
        m = known + (E[r][:, None] * mu).sum(0)
        v = (E[r][:, None] * M.per_game_var(mu) + V[r][:, None] * mu ** 2).sum(0)
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
