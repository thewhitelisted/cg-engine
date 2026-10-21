"""Item 10: are the projection-error model's predictive intervals calibrated out of sample?

For each target season t (2024-2026): error model fit on seasons before t (as the live model does),
projections for t from seasons before t. Draw 2000 truths per player from the error model and check
how often the realised season line falls inside the central 50% / 80% / 95% intervals.
Calibrated -> coverage ~= nominal.
"""
import numpy as np
import pandas as pd

import model as M
import projections as Pj

LEVELS = (0.5, 0.8, 0.95)


def coverage(t: int, n=2000):
    s = M.Season(t)
    ev = M.Evaluator(s, proj_unc=False)
    tr = ev.draw_truth(n, np.random.default_rng(t))
    act = Pj.actuals()
    act = act[act.season == t].set_index("id")
    df = s.df
    have = df["id"].isin(act.index).to_numpy()
    y = act.reindex(df["id"])
    rows = []
    rot = (y["gp"] >= 20).to_numpy() & (y["min"] >= 15).to_numpy() & have
    for k in ("pts", "reb", "ast", "stl", "blk", "tov", "tpm"):
        draws, obs = tr["mu"][:, rot, M.SK[k]], y[k].to_numpy()[rot]
        for lv in LEVELS:
            lo, hi = np.quantile(draws, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            rows.append({"season": t, "stat": k, "level": lv, "cover": ((obs >= lo) & (obs <= hi)).mean()})
    fgp_obs = (y["fgm"] / y["fga"]).to_numpy()[rot]
    fgp = (tr["mu"][:, rot, M.SK["fgx"]] / np.maximum(tr["mu"][:, rot, M.SK["fga"]], 1e-6)) + M.LG_FG
    for lv in LEVELS:
        lo, hi = np.quantile(fgp, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        rows.append({"season": t, "stat": "fg%", "level": lv, "cover": ((fgp_obs >= lo) & (fgp_obs <= hi)).mean()})
    # games-played share among players projected for real minutes (outcome not conditioned on)
    gm = (df["espn_min"] >= 20).to_numpy() & have
    share = (y["gp"].fillna(0) / Pj.sgames(t)).to_numpy()[gm]
    for lv in LEVELS:
        lo, hi = np.quantile(tr["avail"][:, gm], [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        rows.append({"season": t, "stat": "gp_share", "level": lv, "cover": ((share >= lo) & (share <= hi)).mean()})
    return rows


if __name__ == "__main__":
    import sys
    if "--boot" in sys.argv:
        M.set_error_draws("boot")
    r = pd.DataFrame([x for t in (2024, 2025, 2026) for x in coverage(t)])
    r.to_csv(M.DATA / "coverage.csv", index=False)
    p = r.pivot_table(index="stat", columns="level", values="cover", aggfunc="mean").round(3)
    print("Out-of-sample interval coverage (nominal 0.5 / 0.8 / 0.95), mean over 2024-2026:")
    print(p.to_string())
