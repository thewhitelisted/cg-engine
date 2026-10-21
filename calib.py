"""Calibration of the weekly matchup model against real 2025-26 weeks.

Random 13-man rosters from rotation players; each real week, team totals from actual box scores.
Compare observed category/matchup wins with model predictions:
  A) independent normal per category, ACTUAL games that week (isolates correlation/normal error)
  B) same but AVERAGE games/week (what model.py does; ignores games-count variance)
  C) correlated: Monte Carlo from multivariate normal using each player's per-game covariance,
     actual games (tests whether modelling cross-category correlation fixes A)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

g = pd.read_csv("data/gamelogs_2026.csv")
g = g[g["min"] > 0].copy()
rfg, rft = g.fgm.sum() / g.fga.sum(), g.ftm.sum() / g.fta.sum()
g["fgx"] = g.fgm - rfg * g.fga
g["ftx"] = g.ftm - rft * g.fta
g["wk"] = g.scoring_period // 7
K = ["fgx", "fga", "ftx", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"]

stats = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
pool = stats[(stats.n >= 45) & (stats.mins >= 24)].index.to_numpy()
mu = g[g.id.isin(pool)].groupby("id")[K].mean().loc[pool].to_numpy()
cov = np.stack([g[g.id == p][K].cov().to_numpy() for p in pool])
weeks = sorted(w for w in g.wk.unique() if g[g.wk == w].scoring_period.nunique() == 7)
# actual weekly totals & games per player-week
wt = g[g.id.isin(pool)].groupby(["id", "wk"])[K].sum()
wg = g[g.id.isin(pool)].groupby(["id", "wk"]).size()
T = np.zeros((len(pool), len(weeks), len(K)))
G = np.zeros((len(pool), len(weeks)))
pidx = {p: i for i, p in enumerate(pool)}
widx = {w: i for i, w in enumerate(weeks)}
for (p, w), row in wt.iterrows():
    if w in widx:
        T[pidx[p], widx[w]] = row.to_numpy(); G[pidx[p], widx[w]] = wg[(p, w)]
avg_g = G.mean()

def cat_p(mA, vA, mB, vB):
    out = []
    for x, a in ((0, 1), (2, 3)):
        sd = np.sqrt(vA[..., x] / mA[..., a] ** 2 + vB[..., x] / mB[..., a] ** 2)
        out.append(norm.cdf((mA[..., x] / mA[..., a] - mB[..., x] / mB[..., a]) / sd))
    for s in range(4, 11):
        p = norm.cdf((mA[..., s] - mB[..., s]) / np.sqrt(vA[..., s] + vB[..., s]))
        out.append(1 - p if s == 9 else p)
    return np.stack(out, -1)

def outcome(tA, tB):
    o = [(tA[..., 0] / tA[..., 1] > tB[..., 0] / tB[..., 1]), (tA[..., 2] / tA[..., 3] > tB[..., 2] / tB[..., 3])]
    for s in range(4, 11):
        o.append(tA[..., s] < tB[..., s] if s == 9 else tA[..., s] > tB[..., s])
    return np.stack(o, -1).astype(float)

def pbin(p):
    d = np.zeros(p.shape[:-1] + (10,)); d[..., 0] = 1
    for c in range(9):
        pc = p[..., c:c + 1]
        d = np.concatenate([d[..., :1] * (1 - pc), d[..., 1:] * (1 - pc) + d[..., :-1] * pc], -1)
    return d[..., 5:].sum(-1)

rng = np.random.default_rng(0)
N = 600
rows = []
var = np.stack([np.diagonal(c) for c in cov])
for i in range(N):
    A, B = np.split(rng.choice(len(pool), 26, replace=False), 2)
    for w in range(len(weeks)):
        tA, tB = T[A, w].sum(0), T[B, w].sum(0)
        if tA[1] == 0 or tB[1] == 0: continue
        obs = outcome(tA, tB)
        gA, gB = G[A, w][:, None], G[B, w][:, None]
        pa = cat_p((gA * mu[A]).sum(0), (gA * var[A]).sum(0), (gB * mu[B]).sum(0), (gB * var[B]).sum(0))
        pb = cat_p(avg_g * mu[A].sum(0), avg_g * var[A].sum(0), avg_g * mu[B].sum(0), avg_g * var[B].sum(0))
        # correlated MC
        mA_, mB_ = (gA * mu[A]).sum(0), (gB * mu[B]).sum(0)
        cA, cB = (G[A, w][:, None, None] * cov[A]).sum(0), (G[B, w][:, None, None] * cov[B]).sum(0)
        sA = rng.multivariate_normal(mA_, cA, 300); sB = rng.multivariate_normal(mB_, cB, 300)
        oc = outcome(sA, sB)
        rows.append(dict(obs_cats=obs.sum(), obs_win=float(obs.sum() >= 5),
                         pA=pbin(pa), pB=pbin(pb), pC=(oc.sum(-1) >= 5).mean(),
                         cA=pa.sum(), cB=pb.sum(), **{f"o{c}": obs[c] for c in range(9)}, **{f"a{c}": pa[c] for c in range(9)}))
    if i % 100 == 0: print(i, flush=True)
r = pd.DataFrame(rows)
r.to_csv("data/calib.csv", index=False)
print(f"\n{len(r)} matchup-weeks")
for m in "ABC":
    p = r[f"p{m}"]
    print(f"{m}: Brier {((p - r.obs_win) ** 2).mean():.4f}  logloss {-(r.obs_win*np.log(p.clip(1e-4))+(1-r.obs_win)*np.log((1-p).clip(1e-4))).mean():.4f}")
    r["bin"] = pd.cut(p, [0, .1, .25, .4, .5, .6, .75, .9, 1])
    print(r.groupby("bin", observed=True).agg(pred=(f"p{m}", "mean"), obs=("obs_win", "mean"), n=("obs_win", "size")).round(3).T.to_string())
# per-category calibration under A
cats = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]
print("\nper-cat Brier (A):", {c: round(((r[f"a{i}"] - r[f"o{i}"]) ** 2).mean(), 3) for i, c in enumerate(cats)})
# correlation of category outcomes (residual)
res = np.stack([r[f"o{i}"] - r[f"a{i}"] for i in range(9)], 1)
print("residual corr of category outcomes:")
print(pd.DataFrame(np.corrcoef(res.T), index=cats, columns=cats).round(2).to_string())
