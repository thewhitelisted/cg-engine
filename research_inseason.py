"""In-season updating: how much weight should season-to-date stats get vs the preseason projection?

Model per stat: updated = (K * projection + n * to_date_mean) / (K + n), n = games played so far.
For seasons 2024-2026 (projections built from prior seasons only), at checkpoints of n = 5..40 games,
predict each player's per-game average over the REST of the season; fit K per stat by least squares,
fitting on 2024-2025 and checking the 2026 improvement over projection-only and to-date-only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import model as M

STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fga", "fta"]
CHECKPOINTS = (5, 10, 20, 30, 40)


def panel(t):
    s = M.get_season(t)
    proj = s.df.set_index("id")
    g = pd.read_csv(M.DATA / f"gamelogs_{t}.csv")
    g = g[(g["min"] > 0) & g.id.isin(proj.index)].sort_values(["id", "scoring_period"])
    g["n"] = g.groupby("id").cumcount() + 1
    rows = []
    for pid, gp in g.groupby("id"):
        tot = len(gp)
        for n in CHECKPOINTS:
            if tot < n + 10:
                continue
            first, rest = gp[gp.n <= n], gp[gp.n > n]
            rows.append({"season": t, "id": pid, "n": n,
                         **{f"proj_{k}": proj.at[pid, k] for k in STATS},
                         **{f"td_{k}": first[k].mean() for k in STATS},
                         **{f"y_{k}": rest[k].mean() for k in STATS}, "w": len(rest)})
    return pd.DataFrame(rows)


def sse(d, k, K):
    pred = (K * d[f"proj_{k}"] + d.n * d[f"td_{k}"]) / (K + d.n)
    return (d.w * (d[f"y_{k}"] - pred) ** 2).sum()


if __name__ == "__main__":
    P = pd.concat([panel(t) for t in (2024, 2025, 2026)])
    fit, test = P[P.season < 2026], P[P.season == 2026]
    out = {}
    print(f"{'stat':5} {'K(games)':>9} {'test SSE: proj only':>20} {'to-date only':>13} {'blend':>8}")
    for k in STATS:
        K = minimize_scalar(lambda K: sse(fit, k, K), bounds=(0.5, 300), method="bounded").x
        out[k] = round(float(K), 1)
        print(f"{k:5} {K:9.1f} {sse(test, k, 1e9):20.1f} {sse(test, k, 1e-9):13.1f} {sse(test, k, K):8.1f}")
    pd.Series(out).to_json(M.DATA / "inseason_K.json")
    print("saved data/inseason_K.json", out)
