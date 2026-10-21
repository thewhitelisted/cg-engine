"""Fantasy-playoff-window (Mar 8-28 equivalent) participation vs rest of season, by prior-season team record.

Output: data/rest_window.csv and the fitted line  drop = a + b * prior_win_pct  used in model.py.
"""
import numpy as np
import pandas as pd

WINDOWS = {2024: "2024-03-11", 2025: "2025-03-10", 2026: "2026-03-09"}  # Monday nearest Mar 8; 3 weeks


def build():
    R = pd.read_csv("data/team_records.csv")
    out = []
    for s, start in WINDOWS.items():
        g = pd.read_csv(f"data/gamelogs_{s}.csv")
        sch = pd.read_csv(f"data/schedule_{s}.csv", parse_dates=["date"])
        p = pd.read_csv(f"data/players_{s + 1 if s == 2026 else s}.csv").set_index("id")
        g["date"] = g.scoring_period.map(sch.groupby("scoring_period").date.first())
        lo, hi = pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=20)
        tg = sch.assign(win=(sch.date >= lo) & (sch.date <= hi)).groupby("pro_team_id").win.agg(["sum", "size"])
        d = g.assign(win=(g.date >= lo) & (g.date <= hi)).groupby("id").win.agg(["sum", "size"])
        d = d.join(p["pro_team_id"].rename("t")).dropna().join(tg, on="t", rsuffix="_team")
        d = d[g.groupby("id")["min"].mean().reindex(d.index) >= 25]
        d["part_win"] = d["sum"] / d["sum_team"]
        d["part_out"] = (d["size"] - d["sum"]) / (d["size_team"] - d["sum_team"])
        d["wp_prev"] = d.t.map(R[R.season == s - 1].set_index("pro_team_id").win_pct)
        d["season"] = s
        out.append(d)
    d = pd.concat(out).dropna(subset=["wp_prev"])
    d["drop"] = d.part_win - d.part_out
    d.to_csv("data/rest_window.csv")
    return d


if __name__ == "__main__":
    d = build()
    X = np.c_[np.ones(len(d)), d.wp_prev]
    b = np.linalg.lstsq(X, d["drop"], rcond=None)[0]
    r = d["drop"] - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())
    print(f"drop = {b[0]:.3f} + {b[1]:.3f} * wp_prev   (se {se.round(3)}, n={len(d)})")
    print("baseline participation outside window:", round(d.part_out.mean(), 3))
    for s, g in d.groupby("season"):
        Xs = np.c_[np.ones(len(g)), g.wp_prev]
        print(s, np.linalg.lstsq(Xs, g["drop"], rcond=None)[0].round(3))
