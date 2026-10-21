"""Candidate 4: sensitivity of draft decisions to hand-set parameters (report only).

At representative draft states (our picks 12, 36, 60, 108 after opponents/us draft by ADP), for each
parameter value in a plausible range: recommended pick, and the regret of the BASE recommendation if
the alternative value were true = J_alt(best_alt) - J_alt(best_base), with its paired se.
Objective: P(championship) with calibrated kappa (the configuration being considered for shipping).
"""
import numpy as np
import pandas as pd

import model as M
import sim as S

PARAMS = {
    "LEARN_FRAC": (M, [0.1, 0.2, 0.35]),
    "INJURY_RECOVERY": (M, [0.3, 0.5, 0.7]),
    "WAIVER_DEPTH": (M, [6, 12, 24]),
    "NOISE_B": (S, [0.10, 0.15, 0.22]),
    "SHARP_LAMBDA": (S, [0.3, 0.6, 0.9]),
    "SHARP_FRAC": (S, [0.25, 0.5, 0.75]),
}
STATES = [11, 35, 59, 107]  # picks made before our pick (0-based index of our pick)
N_ROLL = 300


def state_picks(season, k):
    d = S.Draft(season)
    order = np.argsort(d.mkt)
    return [int(p) for p in order[:k]]


def evaluate(season, picks, cands, sharp_frac=None):
    kw = {} if sharp_frac is None else {"sharp_frac": sharp_frac}
    d = S.Draft(season, picks=list(picks), objective="champ", **kw)
    return d.evaluate(cands, n_roll=N_ROLL, seed=777)


def main():
    M.set_kappa(1.182)
    season = M.get_season(2027)
    rows = []
    for k in STATES:
        picks = state_picks(season, k)
        base_d = S.Draft(season, picks=picks, objective="champ")
        cands = base_d.candidate_pool(10)
        base = evaluate(season, picks, cands)
        best_base = int(base.idx.iloc[0])
        for name, (mod, values) in PARAMS.items():
            orig = getattr(mod, name)
            for v in values:
                if v == orig:
                    continue
                setattr(mod, name, v)
                try:
                    if name == "SHARP_FRAC":
                        res = evaluate(season, picks, cands, sharp_frac=v)
                    else:
                        res = evaluate(season, picks, cands)
                finally:
                    setattr(mod, name, orig)
                best_alt = int(res.idx.iloc[0])
                J = res.set_index("idx").J
                regret = J[best_alt] - J[best_base]
                rows.append({"pick": k + 1, "param": name, "value": v, "base_choice": season.df.name[best_base],
                             "alt_choice": season.df.name[best_alt], "changed": best_alt != best_base,
                             "regret": regret, "se": res.set_index("idx").se_vs_best[best_base]})
            print(f"pick {k + 1}: {name} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(M.DATA / "sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False))
    print("\nmax regret by parameter:", r.groupby("param").regret.max().round(4).to_dict())


if __name__ == "__main__":
    main()
