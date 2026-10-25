"""In-season state shared by weekly.py and dashboard_data.py.

League snapshot (data/league_2027.json, written by sync.py or the browser snippet web/espn_sync.js):
  teams [{id, name, roster [[espn_id, name, lineup_slot, injury]], record [w, l, t], waiver_rank}],
  order (team ids by draft slot), schedule [[period, home, away]], results [[period, home, away, winner,
  hw, hl, ht, {stat: result}]], completed_weeks, current_period, generated.
Player rates are the preseason projections blended with season-to-date stats (waiver.updated_rates, fitted K
per stat); remaining uncertainty about each player's rate shrinks as sqrt(K / (K + n)).
"""
import copy
import json

import numpy as np
import pandas as pd

import fetch
import model as M
import waiver

LEAGUE_FILE = M.DATA / "league_2027.json"
IR_SLOT = 13
K_MED = float(np.median(list(waiver.K_GAMES.values())))


def load_league(path=LEAGUE_FILE):
    lg = json.loads(path.read_text())
    return json.loads(lg) if isinstance(lg, str) else lg


_GAMELOGS = None


def disk_cached(name, ttl_s, fetch_fn):
    """fetch_fn() result cached on disk for ttl_s seconds (shared by the separate publish processes of one cloud run,
    so frequent publishing doesn't re-download ESPN's large player feeds every time)."""
    import pickle
    import time
    f = M.DATA / f".cache_{name}.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < ttl_s:
        try:
            return pickle.loads(f.read_bytes())
        except Exception:
            pass
    out = fetch_fn()
    try:
        f.write_bytes(pickle.dumps(out))
    except Exception:
        pass
    return out


GAMELOG_TTL, STATUS_TTL = 1800, 600


def gamelogs():
    """Season-to-date game logs (completed days only), fetched at most every 30 minutes (ESPN can be slow)."""
    global _GAMELOGS
    if _GAMELOGS is None:
        _GAMELOGS = disk_cached("gamelogs", GAMELOG_TTL, _fetch_gamelogs)
    return _GAMELOGS


def _fetch_gamelogs():
    """Box scores from completed days only (a game in progress would enter as a short, partial game)."""
    g = _fetch_all_gamelogs()
    if len(g):
        sched = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
        sp_today = sched.loc[sched.date >= pd.Timestamp.today().normalize(), "scoring_period"]
        if len(sp_today):
            g = g[g["scoring_period"] < sp_today.min()]
    return g


def _fetch_all_gamelogs():
    try:
        g = fetch.fetch_gamelogs(M.LIVE_SEASON, limit=700)
        return g[g["min"] > 0]
    except Exception:
        return pd.DataFrame(columns=["id", "min"])


def inseason_season(season, g):
    """Copy of `season` whose df carries blended rates; season.err_mult shrinks projection error."""
    s = copy.copy(season)
    df = season.df.copy()
    mu, n = waiver.updated_rates(season, g)
    df[M.STATK] = mu
    df["fgm"] = df["fga"] * (df["fgx"] / df["fga"].clip(lower=0.1) + M.LG_FG)
    df["ftm"] = df["fta"] * (df["ftx"] / df["fta"].clip(lower=0.1) + M.LG_FT)
    df["games_to_date"] = n
    s.df = df
    s.err_mult = np.sqrt(K_MED / (K_MED + n))
    return s


def rosters(season, lg, n=M.ROSTER):
    """Active (non-IR) rosters [12, n] of model indices by draft slot (-1 = empty or unmodeled)."""
    idx = {int(i): k for k, i in enumerate(season.df["id"])}
    teams = {t["id"]: t for t in lg["teams"]}
    rost = np.full((12, n), -1, int)
    unmodeled, ir = [], {}
    for s, tid in enumerate(lg["order"]):
        active = [e for e in teams[tid]["roster"] if e[2] != IR_SLOT]
        ir[tid] = [e for e in teams[tid]["roster"] if e[2] == IR_SLOT]
        m = [idx[e[0]] for e in active if e[0] in idx][:n]
        unmodeled += [e[1] for e in active if e[0] not in idx]
        rost[s, : len(m)] = m
    return rost, list(lg["order"]), [teams[t]["name"] for t in lg["order"]], unmodeled, ir


# ESPN matchup periods -> model calendar weeks (Mon-Sun from the season start), read off the league scoreboard
# on 2026-10-19: matchups 1-17 are single weeks from Oct 20-25; matchup 18 is the All-Star double week Feb 15-28.
PERIOD_WEEKS = {**{p: [p - 1] for p in range(1, 18)}, 18: [17, 18], 19: [19]}


# The NBA adds each team's last 2 games (NBA Cup knockout or replacement games, Dec 9-15) after the Cup group
# stage; until ESPN lists them, a team's unscheduled games are placed in the week of Dec 7-13 (model week 7).
N_GAMES, TBD_WEEK = 82, 7


def week_games(season):
    """Team x week scheduled games, with not-yet-scheduled games placed in TBD_WEEK."""
    wg = season.wg.copy()
    if TBD_WEEK in wg:
        wg[TBD_WEEK] = wg[TBD_WEEK] + (N_GAMES - wg.sum(axis=1)).clip(lower=0)
    return wg


def games_by_period(season, periods):
    """Scheduled games per player for each listed matchup period, then the three playoff weeks: [P, n + 3]."""
    team = season.df["pro_team_id"]
    wg = week_games(season)
    cols = [sum(team.map(wg[w] if w in wg else 0).fillna(0).to_numpy() for w in PERIOD_WEEKS[p])
            for p in periods]
    return np.column_stack(cols + [season.games_matrix[:, 1:]]).astype(float)


def state(lg, tids, n_reg=19):
    """Remaining regular-season schedule as slot pairs [weeks, 6, 2] and categories won so far [12]."""
    slot = {t: s for s, t in enumerate(tids)}
    done = set(lg.get("completed_weeks", []))
    sched = sorted(m for m in lg["schedule"] if m[0] <= n_reg)
    weeks = sorted({m[0] for m in sched if m[0] not in done})
    pairs = [[[slot[h], slot[a]] for w, h, a in sched if w == wk] for wk in weeks]
    teams = {t["id"]: t for t in lg["teams"]}
    rec = {t: teams[t].get("record") or lg.get("records", {}).get(str(t)) or [0, 0, 0] for t in tids}
    tot0 = np.array([rec[t][0] + 0.5 * rec[t][2] for t in tids], float)
    return pairs, tot0, weeks, sched


# ---------------------------------------------------------------- availability (DECISIONS.md A1)
AVAIL = json.loads((M.DATA / "availability.json").read_text()) if (M.DATA / "availability.json").exists() else None
OUT_STATUSES = ("OUT", "INJURY_RESERVE", "SUSPENSION")
START_M = 4                     # an OUT/IR player with no in-season absence yet: typical start of a long absence
N_IR = 2


_STATUSES = None


def all_statuses():
    """ESPN injury status for every player (public player feed), fetched once per run."""
    global _STATUSES
    if _STATUSES is None:
        def fetch_all():
            try:
                pl = fetch.flatten(fetch.fetch_players(M.LIVE_SEASON, limit=700), M.LIVE_SEASON)
                return {int(i): str(s) for i, s in zip(pl["id"], pl["injury"].fillna(""))}
            except Exception:
                return {}
        _STATUSES = disk_cached("statuses", STATUS_TTL, fetch_all)
    return _STATUSES


def participation(season, g, today=None):
    """Per player: team games so far c, games played n, current run of missed team games m."""
    sched = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    today = pd.Timestamp.today().normalize() if today is None else today
    past = sched[sched.date < today]
    team_past = past.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
    played = g.groupby("id")["scoring_period"].apply(set) if len(g) else pd.Series(dtype=object)
    df = season.df
    c, n, m = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df), int)
    for i, (pid, tm) in enumerate(zip(df["id"], df["pro_team_id"])):
        tp = team_past.get(tm, np.array([]))
        pl = played.get(pid, set())
        seq = np.array([p in pl for p in tp], bool)
        c[i], n[i] = len(seq), seq.sum()
        run = 0
        for x in seq[::-1]:
            if x:
                break
            run += 1
        m[i] = run
    return c, n, m


def availability(season, g, status, periods, G):
    """Blended games-played share (df avail) and per-period availability multipliers A [P, n + 3].

    status: ESPN injury status by player index. A player out (current run >= 2, or listed OUT / on IR) gets
    P(back) at each remaining team game from the absence-length survival curve; others get A = 1."""
    df = season.df
    K = AVAIL["K_gp"]
    S = np.asarray(AVAIL["survival"])
    c, n, m = participation(season, g)
    out_now = np.array([status.get(i, "") in OUT_STATUSES for i in range(len(df))])
    m_eff = np.where(out_now, np.maximum(m, START_M), m)
    absent = m_eff >= AVAIL["absent_m"]
    c0 = np.where(absent, c - np.minimum(m, c), c)          # data before the current absence
    n0 = n                                                   # games played all fall before the absence
    share = (K * df["avail"].to_numpy() + n0) / (K + c0)
    A = np.ones_like(G)
    cum = np.cumsum(G, axis=1)                               # team games through each remaining period
    for i in np.where(absent)[0]:
        if G[i].sum() == 0:
            continue
        start = np.r_[0, cum[i, :-1]]
        for k in range(G.shape[1]):
            if G[i, k] <= 0:
                continue
            j = np.arange(int(round(start[k])) + 1, int(round(start[k] + G[i, k])) + 1)
            idx = np.minimum(m_eff[i] + j, len(S) - 1)
            p_out = S[idx] / S[m_eff[i]] if S[m_eff[i]] > 0 else np.ones(len(j))
            A[i, k] = float(np.mean(1 - p_out)) if len(j) else 1.0
    return share, A, {"c": c, "n": n, "m": m_eff, "absent": absent}


def build(season, lg, g, today=None):
    """Everything the in-season simulations need from the league snapshot and game logs."""
    rost, tids, names, unmodeled, ir = rosters(season, lg)
    pairs, tot0, weeks_left, sched = state(lg, tids)
    G = games_by_period(season, weeks_left)
    idx = {int(i): k for k, i in enumerate(season.df["id"])}
    status = {idx[pid]: st for pid, st in all_statuses().items() if pid in idx}       # free agents too
    status.update({idx[e[0]]: e[3] for t in lg["teams"] for e in t["roster"] if e[0] in idx})
    share, A, info = availability(season, g, status, weeks_left, G)
    season.df["avail"] = np.clip(share, 0, 1)
    irs = np.full((12, N_IR), -2, int)
    Geff = G * A
    val, _ = M.static_value(season.df)
    for s, tid in enumerate(tids):
        ir_idx = [idx[e[0]] for e in ir.get(tid, []) if e[0] in idx][:N_IR]
        irs[s, : len(ir_idx)] = ir_idx
        active = [p for p in rost[s] if p >= 0]
        weakest = sorted(active, key=lambda p: val[p])
        for q, p_ir in enumerate(ir_idx):                    # a returning IR player displaces the weakest active
            if q < len(weakest):
                Geff[weakest[q]] *= 1 - A[p_ir]
    return {"rost": rost, "ir": irs, "ir_dict": ir, "tids": tids, "names": names, "unmodeled": unmodeled, "pairs": pairs,
            "tot0": tot0, "weeks_left": weeks_left, "sched": sched, "G": G, "Geff": Geff, "A": A, "info": info}


def full(rost, irs):
    """Active roster plus IR slots: [12, 13 + N_IR] (-1 empty active slot, -2 empty IR slot)."""
    return np.concatenate([rost, irs], axis=1)
