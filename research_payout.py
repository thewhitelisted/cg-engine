"""2026-10-17 payout change: does maximising expected prize money ($150/$90/$60) pick differently from
maximising P(title)? Same draft states as research_sensitivity, identical draws for both objectives."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import model as M
import sim as S

STATES = [11, 35, 59, 107]
N_ROLL = 1000


def main():
    M.set_kappa(1.182)
    season = M.get_season(M.LIVE_SEASON, "stack_belief")
    order = np.argsort(S.Draft(season).mkt)
    rows = []
    for k in STATES:
        picks = [int(p) for p in order[:k]]
        cands = S.Draft(season, picks=picks, objective="payout").candidate_pool(10)
        res = {}
        for obj in ("champ", "payout"):
            d = S.Draft(season, picks=picks, objective=obj)
            d.evaluate(cands, n_roll=N_ROLL, seed=777)
            res[obj] = d.last_J
        Jc = {c: res["champ"][c].mean() for c in cands}
        Jp = {c: res["payout"][c].mean() for c in cands}
        bc, bp = max(cands, key=Jc.get), max(cands, key=Jp.get)
        gap = res["payout"][bp] - res["payout"][bc]
        rows.append({"pick": k + 1, "best_champ": season.df.name[bc], "best_payout": season.df.name[bp],
                     "regret_$": gap.mean(), "se_$": gap.std(ddof=1) / np.sqrt(N_ROLL),
                     "rank_corr": spearmanr([Jc[c] for c in cands], [Jp[c] for c in cands])[0],
                     "J_payout_best_$": Jp[bp], "P_title_best": Jc[bc]})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(M.DATA / "payout_objective.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
