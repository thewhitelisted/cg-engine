"""Diagnostic: winner's curse. Do players our model likes more than the market under-deliver?

For each real season: projected value (our model) vs actual season value (same G-score scale), for
players in the drafted pool. Regress (actual - projected) on (market rank - our value rank): a negative
slope means where we disagree with the market, the market is partly right. Also: realised
(actual - projected) value of the players each strategy drafted in the real-season backtest.
"""
import numpy as np
import pandas as pd

import model as M
import projections as Pj

rows = []
for t, clean in ((2024, True), (2025, False), (2026, False)):
    s = M.get_season(t)
    df = s.df
    ev = M.Evaluator(s, proj_unc=False)
    val, scale = M.static_value(df)
    act = Pj.actuals()
    act = act[act.season == t].set_index("id").reindex(df["id"])
    act["fgx"] = act["fgm"] - M.LG_FG * act["fga"]
    act["ftx"] = act["ftm"] - M.LG_FT * act["fta"]
    mu_act = act[M.STATK].fillna(0).to_numpy()
    share = (act["gp"].fillna(0) / Pj.sgames(t)).to_numpy()
    aval = M.value_from(mu_act, share, scale)
    vrank = pd.Series(val).rank(ascending=False).to_numpy()
    pool = df["mkt"].to_numpy() <= 170
    d = pd.DataFrame({"season": t, "clean_adp": clean, "name": df["name"], "proj": val, "actual": aval,
                      "disagree": df["mkt"].to_numpy() - vrank})[pool]
    rows.append(d)
d = pd.concat(rows)
d["resid"] = d.actual - d.proj
print("mean (actual - projected) value over the drafted pool:", d.resid.mean().round(3))
for (t, c), g in d.groupby(["season", "clean_adp"]):
    b = np.polyfit(g.disagree, g.resid, 1)[0]
    q = pd.qcut(g.disagree, 4, labels=["we like LESS", "", " ", "we like MORE"])
    print(f"{t} ({'real ADP' if c else 'proxy market'}): slope {b:+.4f} per rank of disagreement;",
          g.groupby(q, observed=True).resid.mean().round(2).to_dict())
r = pd.read_csv(M.DATA / "realtest_dev_all.csv")
z = np.load(M.DATA / "rosters_dev.npz")
print("\nrealised minus projected value of OUR roster's players, by strategy (mean per player):")
out = {}
for key in z.files:
    t, seed, strat = key.split("_", 2)
    g = d[d.season == int(t)]
    s = M.get_season(int(t)).df
    idx = [p for p in z[key][11] if p >= 0]
    sub = d[(d.season == int(t))].reindex(idx)
    out.setdefault(strat, []).append(sub.resid.mean())
print({k: round(float(np.nanmean(v)), 3) for k, v in sorted(out.items())})
