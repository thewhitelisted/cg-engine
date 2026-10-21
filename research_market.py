"""Candidate 8: fit beta (market shrinkage) on 2023-24; player-level check on the 2021-22 holdout.

    python3 research_market.py fit        # 2023-24 only -> data/market_beta.json
    python3 research_market.py holdout    # 2021-22 player-level MSE (run once, after registration)
"""
import json
import sys

import numpy as np
import pandas as pd

import model as M
import projections as Pj


def frame(t):
    s = M.Season(t)  # MARKET_BETA must be 0 here: we need the unadjusted model value
    df = s.df
    val, scale = M.static_value(df)
    act = Pj.actuals()
    act = act[act.season == t].set_index("id").reindex(df["id"])
    act["fgx"] = act["fgm"] - M.LG_FG * act["fga"]
    act["ftx"] = act["ftm"] - M.LG_FT * act["fta"]
    aval = M.value_from(act[M.STATK].fillna(0).to_numpy(), (act["gp"].fillna(0) / Pj.sgames(t)).to_numpy(), scale)
    mrank = df["mkt"].rank(method="first").astype(int).to_numpy()
    mkt_val = np.sort(val)[::-1][mrank - 1]
    pool = df["mkt"].to_numpy() <= 170
    return pd.DataFrame({"name": df["name"], "model": val, "market": mkt_val, "actual": aval})[pool]


if __name__ == "__main__":
    assert M.MARKET_BETA == 0
    if sys.argv[1] == "fit":
        d = frame(2024)
        x, y = d.market - d.model, d.actual - d.model
        X = np.c_[np.ones(len(d)), x]
        (a, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ [a, beta]
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())[1]
        print(f"2023-24: beta = {beta:.3f} (se {se:.3f}), n={len(d)}")
        (M.DATA / "market_beta.json").write_text(json.dumps({"beta": float(beta), "se": float(se)}))
    else:
        beta = json.loads((M.DATA / "market_beta.json").read_text())["beta"]
        d = frame(2022)
        adj = d.model + beta * (d.market - d.model)
        for name, pred in (("model", d.model), ("market-implied", d.market), (f"blend beta={beta:.2f}", adj)):
            print(f"2021-22 holdout MSE vs actual value: {name:22} {((d.actual - pred) ** 2).mean():.3f}")


def loso():
    """C9 (beta by leave-one-season-out) and C10 (adjust only market rank <= 135) vs shipped beta."""
    shipped = json.loads((M.DATA / "market_beta.json").read_text())["beta"]
    F = {t: frame(t) for t in (2022, 2024, 2026)}
    for t, d in F.items():
        d["mrank"] = d["market"].rank(ascending=False)  # placeholder, replaced below
    # market rank of each player (for C10): recover from Season.mkt
    for t in F:
        s = M.Season(t)
        F[t]["mkt"] = s.df.loc[F[t].index, "mkt"].to_numpy()

    def fit_beta(ds):
        x = np.concatenate([(d.market - d.model).to_numpy() for d in ds])
        y = np.concatenate([(d.actual - d.model).to_numpy() for d in ds])
        X = np.c_[np.ones(len(x)), x]
        return float(np.linalg.lstsq(X, y, rcond=None)[0][1])

    rows = []
    for t, d in F.items():
        b_loso = fit_beta([F[o] for o in F if o != t])
        real = (d.mkt <= 135).to_numpy()
        pred = {
            "model only": d.model,
            f"shipped beta {shipped:.2f}": d.model + shipped * (d.market - d.model),
            f"C9 loso beta {b_loso:.2f}": d.model + b_loso * (d.market - d.model),
            "C10 shipped, real-ADP only": d.model + np.where(real, shipped, 0) * (d.market - d.model),
            "C9+C10": d.model + np.where(real, b_loso, 0) * (d.market - d.model),
        }
        for name, p in pred.items():
            rows.append({"season": t, "variant": name.split(" beta")[0] if name.startswith(("shipped", "C9 loso")) else name,
                         "label": name, "mse": float(((d.actual - p) ** 2).mean())})
    r = pd.DataFrame(rows)
    print(r.pivot_table(index="variant", columns="season", values="mse").round(3).assign(
        pooled=lambda x: x.mean(1).round(3)).to_string())
    print("betas used:", sorted(set(r.label)))


def pooled():
    """Deployment beta: pooled over all clean-ADP seasons, season intercepts, real-ADP players only."""
    xs, ys, Ds = [], [], []
    seasons = (2022, 2024, 2026)
    for i, t in enumerate(seasons):
        d = frame(t)
        d["mkt"] = M.Season(t).df.loc[d.index, "mkt"].to_numpy()
        d = d[d.mkt <= 135]
        xs.append((d.market - d.model).to_numpy())
        ys.append((d.actual - d.model).to_numpy())
        D = np.zeros((len(d), len(seasons)))
        D[:, i] = 1
        Ds.append(D)
        b_t = np.polyfit(xs[-1], ys[-1], 1)[0]
        print(f"  {t}: beta {b_t:.3f} (n={len(d)})")
    x, y, D = np.concatenate(xs), np.concatenate(ys), np.concatenate(Ds)
    X = np.c_[D, x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    se = float(np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())[-1])
    beta = float(coef[-1])
    print(f"pooled beta = {beta:.3f} (se {se:.3f}, 95% CI {beta - 1.96 * se:.3f} to {beta + 1.96 * se:.3f}), n={len(x)}")
    (M.DATA / "market_beta_deployed.json").write_text(json.dumps({"beta": beta, "se": se}))
