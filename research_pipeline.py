"""Item 2: is the tool's own estimate of its title odds honest?

For N drafts (shipping config candidate: P(champ) objective, kappa 1.182), record the tool's predicted
P(championship) for its chosen pick at round 1 (pick 12) and at its final pick, then score the final
roster on independent simulated true seasons (backtest.SimTruth, different seed). Under a correct
pipeline, mean predicted ~= mean realised; a systematic gap = optimism (optimizer's curse) or pessimism.
"""
import sys
import time

import numpy as np
import pandas as pd

import backtest as B
import harness as H
import model as M
import sim as S


def run(n=20):
    st = B.SimTruth(n_truth=48, seed=99)
    rows, t0 = [], time.time()
    season = M.get_season(2027, "stack")
    for seed in range(n):
        M.set_kappa(1.182)
        d = S.Draft(season, objective="champ")
        rng = np.random.default_rng(70_000 + seed)
        boards = d.boards(rng, 1)[0]
        pred_first = pred_last = None
        for k in range(d.n_picks):
            t = d.order[k]
            avail = d.available()
            if t != d.me:
                d.picks.append(int(np.where(avail, boards[t], np.inf).argmin()))
                continue
            res = d.evaluate(d.candidate_pool(12), n_roll=200, seed=seed * 100 + k)
            if pred_first is None:
                pred_first = res.J.iloc[0]
            pred_last = res.J.iloc[0]
            d.picks.append(int(res.idx.iloc[0]))
        R = np.full((M.N_TEAMS, M.ROSTER), -1)
        for team in range(M.N_TEAMS):
            r = d.roster(team)
            R[team, : len(r)] = r
        realised = st.score(R, metric="champ")
        rows.append({"seed": seed, "pred_pick12": pred_first, "pred_final": pred_last, "realised": realised})
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(M.DATA / "pipeline_calib.csv", index=False)
    for c in ("pred_pick12", "pred_final"):
        d = r[c] - r.realised
        print(f"{c}: mean {r[c].mean():.4f} vs realised {r.realised.mean():.4f}  gap {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
