"""Forward test of the 2026-27 season (fresh holdout for v3).

Uses the rosters recorded by draft.py (data/draft_state.json): no extra input about opponents.
Optionally data/my_roster.json (list of player names) overrides your own roster after waiver moves.

    python3 track.py predict [week]   # before the week: write data/track/pred_wkNN.json (timestamped, data hash)
    python3 track.py score [week]     # after the week: actual category results -> data/track/scores.csv
    python3 track.py report           # cumulative calibration of matchup predictions so far

Week numbers are Mon-Sun weeks from the season start (model.Season.wg columns).
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fetch
import model as M
from draft import LIVE_CONFIG

M.set_kappa(LIVE_CONFIG["kappa"])

TRACK = M.DATA / "track"
STATE = M.DATA / "draft_state.json"


def rosters(season):
    st = json.loads(STATE.read_text())
    df = season.df
    id2idx = {pid: i for i, pid in enumerate(df["id"])}
    R = [[] for _ in range(M.N_TEAMS)]
    for k, pid in enumerate(st["pick_ids"]):
        t = (k % 12) if (k // 12) % 2 == 0 else 11 - (k % 12)
        R[t].append(id2idx.get(pid, -1))
    me = st["slot"] - 1
    mine = M.DATA / "my_roster.json"
    if mine.exists():
        names = json.loads(mine.read_text())
        name2idx = {n.lower(): i for i, n in enumerate(df["name"])}
        R[me] = [name2idx.get(n.lower(), -1) for n in names]
    out = np.full((M.N_TEAMS, M.ROSTER + 2), -1)
    for t, r in enumerate(R):
        out[t, : len(r)] = r[: M.ROSTER + 2]
    return out, me


def predict(week: int):
    season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
    df = season.df.copy()
    # current injury status from a fresh pull: OUT players contribute nothing this week
    fresh = fetch.flatten(fetch.fetch_players(M.LIVE_SEASON, limit=600), M.LIVE_SEASON).set_index("id")
    out = df["id"].map(fresh["injury"]).isin(["OUT", "INJURY_RESERVE", "SUSPENSION"]).to_numpy()
    ev = M.Evaluator(season, proj_unc=True)
    g = df["pro_team_id"].map(season.wg[week]).fillna(0).to_numpy()[:, None] * np.ones((1, 4))
    avail = np.where(out, 0.0, ev.avail)
    wmu, wvar, own = ev.weekly(ev.mu, avail, g)
    ev.wmu, ev.wvar, ev.own = wmu, wvar, own
    R, me = rosters(season)
    m, v = ev.team_stats(R)
    p = M.Evaluator.matchup_win(M.Evaluator.cat_probs(m[:, None, 0], v[:, None, 0], m[None, :, 0], v[None, :, 0]))
    TRACK.mkdir(parents=True, exist_ok=True)
    rec = {"week": week, "made_at": datetime.now().isoformat(timespec="seconds"), "kappa": M.MATCHUP_SD_KAPPA,
           "data_hash": hashlib.sha256(json.dumps(fresh["injury"].to_dict(), sort_keys=True).encode()).hexdigest()[:16],
           "me": me, "p_win": p.round(4).tolist()}
    path = TRACK / f"pred_wk{week:02d}.json"
    if path.exists():
        sys.exit(f"{path.name} already exists: predictions are write-once")
    path.write_text(json.dumps(rec))
    print(f"wrote {path.name}; your P(win) vs each team:", np.round(p[me], 2).tolist())


def score(week: int):
    season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
    pred = json.loads((TRACK / f"pred_wk{week:02d}.json").read_text())
    g = fetch.fetch_gamelogs(M.LIVE_SEASON, limit=700)
    sch = pd.read_csv(M.DATA / f"schedule_{M.LIVE_SEASON}.csv", parse_dates=["date"])
    g["date"] = g.scoring_period.map(sch.groupby("scoring_period").date.first())
    g["wk"] = (g["date"] - season.season_start).dt.days // 7
    g = g[(g.wk == week) & (g["min"] > 0)]
    g["fgx"], g["ftx"] = g.fgm - M.LG_FG * g.fga, g.ftm - M.LG_FT * g.fta
    tot = g.groupby("id")[M.STATK].sum()
    df = season.df
    R, me = rosters(season)
    T = np.zeros((M.N_TEAMS, len(M.STATK)))
    for t in range(M.N_TEAMS):
        ids = [df["id"].iloc[i] for i in R[t] if i >= 0]
        T[t] = tot.reindex(ids).fillna(0).sum().to_numpy()
    k = M.SK
    stats = np.stack([T[:, k["fgx"]] / np.maximum(T[:, k["fga"]], 1), T[:, k["ftx"]] / np.maximum(T[:, k["fta"]], 1)]
                     + [T[:, k[s]] for s in ("tpm", "reb", "ast", "stl", "blk")] + [-T[:, k["tov"]], T[:, k["pts"]]], -1)
    rows = []
    P = np.array(pred["p_win"])
    for a in range(M.N_TEAMS):
        for b in range(a + 1, M.N_TEAMS):
            w, l = (stats[a] > stats[b]).sum(), (stats[a] < stats[b]).sum()
            rows.append({"week": week, "a": a, "b": b, "p": P[a, b], "y": 1.0 if w > l else 0.5 if w == l else 0.0})
    path = TRACK / "scores.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    old = old[old.get("week", pd.Series(dtype=int)) != week] if len(old) else old
    pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)
    report()


def report():
    r = pd.read_csv(TRACK / "scores.csv")
    print(f"{r.week.nunique()} weeks, {len(r)} matchup predictions. Brier {((r.p - r.y) ** 2).mean():.4f} "
          f"(coin flip 0.25)")
    r["bin"] = pd.cut(r.p, [0, .2, .4, .6, .8, 1])
    print(r.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).round(3).T.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    else:
        season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
        today_wk = (pd.Timestamp.today().normalize() - season.season_start).days // 7
        wk = int(sys.argv[2]) if len(sys.argv) > 2 else (today_wk if cmd == "predict" else today_wk - 1)
        {"predict": predict, "score": score}[cmd](wk)
