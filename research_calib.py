"""Candidate 2: one multiplicative factor kappa on the matchup standard deviation.

Random 13-man rosters of rotation players play each real week. Predicted P(win matchup) uses the
production model's weekly structure (mean games, per-game variance fits, games variance) with
per-player season means; observed = real category outcome. Fit kappa on 2024-25, test on 2025-26
(pre-registered in DECISIONS.md: adopt only if 2025-26 Brier improves).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import model as M

STATK = M.STATK


def build(season: int, n_match=500, seed=0):
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0].copy()
    g["fgx"] = g.fgm - M.LG_FG * g.fga
    g["ftx"] = g.ftm - M.LG_FT * g.fta
    g["wk"] = g.scoring_period // 7
    st = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
    pool = st[(st.n >= 40) & (st.mins >= 24)].index.to_numpy()
    g = g[g.id.isin(pool)]
    mu = g.groupby("id")[STATK].mean().loc[pool].to_numpy()
    weeks = [w for w in sorted(g.wk.unique()) if g[g.wk == w].scoring_period.nunique() == 7]
    tot = g.groupby(["id", "wk"])[STATK].sum()
    ng = g.groupby(["id", "wk"]).size()
    pidx = {p: i for i, p in enumerate(pool)}
    widx = {w: i for i, w in enumerate(weeks)}
    T = np.zeros((len(pool), len(weeks), len(STATK)))
    G = np.zeros((len(pool), len(weeks)))
    for (p, w), row in tot.iterrows():
        if w in widx:
            T[pidx[p], widx[w]] = row.to_numpy()
            G[pidx[p], widx[w]] = ng[(p, w)]
    gm = G.mean(1)  # expected games/week incl. missed weeks (availability)
    var_pg = M.per_game_var(mu)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_match):
        A, B = np.split(rng.choice(len(pool), 26, replace=False), 2)
        mA = (gm[A, None] * mu[A]).sum(0)
        mB = (gm[B, None] * mu[B]).sum(0)
        vA = (gm[A, None] * var_pg[A] + M.GAMES_VAR_RATIO * gm[A, None] * mu[A] ** 2).sum(0)
        vB = (gm[B, None] * var_pg[B] + M.GAMES_VAR_RATIO * gm[B, None] * mu[B] ** 2).sum(0)
        for w in range(len(weeks)):
            a, b = T[A, w].sum(0), T[B, w].sum(0)
            if a[1] == 0 or b[1] == 0:
                continue
            sa = np.r_[a[0] / a[1], a[2] / a[3], a[4:9], -a[9], a[10]]
            sb = np.r_[b[0] / b[1], b[2] / b[3], b[4:9], -b[9], b[10]]
            won, lost = (sa > sb).sum(), (sa < sb).sum()
            rows.append((mA, vA, mB, vB, 1.0 if won > lost else 0.5 if won == lost else 0.0))
    return rows


def predict(rows, kappa):
    mA = np.array([r[0] for r in rows]); vA = np.array([r[1] for r in rows])
    mB = np.array([r[2] for r in rows]); vB = np.array([r[3] for r in rows])
    p = M.Evaluator.matchup_win(M.Evaluator.cat_probs(mA, vA * kappa**2, mB, vB * kappa**2))
    return p, np.array([r[4] for r in rows])


def brier(rows, kappa):
    p, y = predict(rows, kappa)
    return ((p - y) ** 2).mean()


if __name__ == "__main__":
    fit_rows, test_rows = build(2025), build(2026, seed=1)
    k = minimize_scalar(lambda k: brier(fit_rows, k), bounds=(0.7, 2.0), method="bounded").x
    print(f"kappa fit on 2024-25: {k:.3f}   (Brier fit: k=1 {brier(fit_rows, 1):.4f} -> {brier(fit_rows, k):.4f})")
    b1, bk = brier(test_rows, 1.0), brier(test_rows, k)
    print(f"2025-26 test Brier: kappa=1 {b1:.4f}   kappa={k:.3f} {bk:.4f}   -> {'ADOPT' if bk < b1 else 'REJECT'}")
    for kk in (1.0, k):
        p, y = predict(test_rows, kk)
        bins = pd.cut(p, [0, .1, .25, .4, .6, .75, .9, 1])
        print(f"calibration on 2025-26, kappa={kk:.2f}:")
        print(pd.DataFrame({"pred": p, "obs": y, "bin": bins}).groupby("bin", observed=True).agg(
            pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size")).round(3).T.to_string())
