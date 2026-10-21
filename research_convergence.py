"""Item 11: convergence/robustness of the decision engine.

At picks 12, 36, 60 (earlier picks by ADP): (1) is the best of a 40-candidate pool inside the
16-candidate shortlist? (2) does replacing our in-rollout policy (gradient category weights) with plain
static value change the ranking of candidates? Shipping config: P(championship), kappa 1.182.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import model as M
import sim as S

M.set_kappa(1.182)
season = M.get_season(2027)
rows = []
for k in (11, 35, 59):
    picks = [int(p) for p in np.argsort(season.df["mkt"].to_numpy())[:k]]
    d = S.Draft(season, picks=picks, objective="champ")
    short = d.candidate_pool(16)
    wide = list(dict.fromkeys(short + d.candidate_pool(40)))
    rw = d.evaluate(wide, n_roll=400, seed=5)
    best = int(rw.idx.iloc[0])
    gap_short = rw.J.iloc[0] - rw[rw.idx.isin(short)].J.max()
    ds = S.Draft(season, picks=picks, objective="champ", policy="static")
    rs = ds.evaluate(short, n_roll=400, seed=5)
    rg = rw[rw.idx.isin(short)].set_index("idx").J
    rho = spearmanr(rg.loc[rs.idx], rs.J).correlation
    rows.append({"pick": k + 1, "pool": len(wide), "best_in_wide": season.df.name[best],
                 "best_in_shortlist": best in short, "shortlist_gap": round(gap_short, 4),
                 "best_gradient": season.df.name[int(rg.idxmax())], "best_static": rs.name.iloc[0],
                 "rank_corr_policies": round(rho, 3)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(M.DATA / "convergence.csv", index=False)
