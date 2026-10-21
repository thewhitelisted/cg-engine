"""Do teammates' outcomes co-move? (a) season-level projection residuals, (b) weekly production residuals."""
import itertools

import numpy as np
import pandas as pd

import projections as Pj

STATS = ["pts", "reb", "ast", "tpm", "stl", "blk", "tov"]


def pair_corr(df, val_cols, group):
    """Mean within-group pairwise correlation of residuals vs across-group baseline."""
    same, diff = [], []
    for k in val_cols:
        x = df[k].to_numpy(); gr = df[group].to_numpy()
        z = (x - x.mean()) / x.std()
        # within-group: sum of pairwise products = ((sum z)^2 - sum z^2)/2
        tmp = pd.DataFrame({"g": gr, "z": z})
        s = tmp.groupby("g").z.agg(["sum", "size"])
        s2 = tmp.assign(z2=z**2).groupby("g").z2.sum()
        num = ((s["sum"] ** 2 - s2) / 2).sum()
        npairs = (s["size"] * (s["size"] - 1) / 2).sum()
        same.append(num / npairs)
    return dict(zip(val_cols, np.round(same, 3)))


# (a) season-level: leave-one-season-out residuals of the stacked projection
P = Pj.panel(range(2022, 2027))
rows = []
for t in range(2022, 2027):
    te = P[(P.season == t) & Pj.rate_mask(P)]
    pr = Pj.predict(te, Pj.fit(P[P.season != t]))
    F = Pj.files()[t + 1 if t == 2026 else t]
    r = pd.DataFrame({k: (te[f"y_{k}"] - pr[k]) / np.sqrt(pr[k].clip(0.3)) for k in STATS})
    r["min"] = te["y_min"] - te["e_min"]
    r["team"] = F["pro_team_id"].reindex(te.index).astype(str) + f"_{t}"
    rows.append(r)
R = pd.concat(rows).dropna()
print("(a) season residual corr between teammates (null = ~0):", pair_corr(R, STATS + ["min"], "team"))

# (b) weekly residuals: weekly total - games * season per-game mean, teammates in the same week
for s in (2025, 2026):
    g = pd.read_csv(f"data/gamelogs_{s}.csv"); g = g[g["min"] > 0]
    F = Pj.files()[s + 1 if s == 2026 else s]
    g["team"] = g.id.map(F["pro_team_id"])
    g["wk"] = g.scoring_period // 7
    rot = g.groupby("id")["min"].mean() >= 24
    g = g[g.id.map(rot)]
    mu = g.groupby("id")[STATS].mean()
    W = g.groupby(["id", "wk"]).agg(**{k: (k, "sum") for k in STATS}, n=("pts", "size"), team=("team", "first")).reset_index()
    # residual given games played (usage competition), and raw (includes shared schedule)
    for k in STATS:
        W[k + "_cond"] = (W[k] - W["n"] * W.id.map(mu[k])) / np.sqrt(W["n"] * W.id.map(mu[k]).clip(0.1))
    W["grp"] = W.team.astype(str) + "_" + W.wk.astype(str)
    print(f"(b) {s} weekly corr between teammates, given games played:",
          pair_corr(W, [k + "_cond" for k in STATS], "grp"))
    W["n_c"] = W["n"] - W.groupby("wk")["n"].transform("mean")
    print(f"    {s} weekly corr of games played between teammates:", pair_corr(W, ["n_c"], "grp"))
