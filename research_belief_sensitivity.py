"""Registered report: recommended picks at 12 / 36 / 60 under the shipped variant's tau prior x0.5, x1, x2."""
import numpy as np
import pandas as pd

import belief as Bf
import model as M
import sim as S

M.set_kappa(1.182)
base_order = M.Season(2027, "stack").df
rows = []
for pm in (0.5, 1.0, 2.0):
    h = Bf.fit_final(Bf.shipped_variant(), prior_mult=pm)
    Bf.load_for_season = lambda t, h=h: h  # this script only
    season = M.Season(2027, "stack_belief")
    s = season.belief
    for k in (11, 35, 59):
        picks = [int(p) for p in np.argsort(base_order["mkt"].to_numpy())[:k]]
        d = S.Draft(season, picks=list(picks), objective="champ")
        res = d.evaluate_adaptive(d.candidate_pool(14), budget_s=40, batch=200, seed=3)
        summ = h["summary"]
        sds = {f"{comp}_sd": {c: v["pred_sd"] for c, v in (summ[comp].items() if "pred_sd" not in summ[comp]
                                                         else [("main", summ[comp])])} for comp in ("beta", "gamma")}
        rows.append({"prior": f"x{pm}", **{k_: str(v_) for k_, v_ in sds.items()},
                     "pick": k + 1, "best": res.name.iloc[0],
                     "next two (gap ± se)": ", ".join(f"{r.name} {r.J - res.J.iloc[0]:+.3f}±{r.se_vs_best:.3f}"
                                                      for r in res.iloc[1:3].itertuples()),
                     "clear": res.attrs["resolved"]})
        print(rows[-1], flush=True)
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(pd.DataFrame(rows).to_string(index=False))
