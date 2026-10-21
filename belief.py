"""Belief layer: market-informed predictive distributions of player seasons, validated by proper scoring.

Registered in DECISIONS.md ("PROTOCOL AMENDMENT + Belief-layer redesign", 2026-10-05).

Three belief models:
  V0  stack projection only.
  V1  single pooled market weight on total value (role multiplier on per-game rates).
  V2  decomposed hierarchical: market weight on per-game rates (beta) and on GP share (gamma), each varying
      by season with a random-effects distribution estimated across seasons; every simulated season draws
      its own (beta, gamma) with one shared shock.
Evaluation: leave-one-season-out CRPS of each player's actual season value (players with market rank <= 170).

    python3 belief.py evaluate     # the registered comparison -> data/belief_eval.json + report
    python3 belief.py fit          # refit the shipped variant on all clean seasons -> data/belief_final.json
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import model as M
import projections as Pj

CLEAN_SEASONS = (2022, 2024, 2026)
REAL_ADP_RANK = 135
EVAL_RANK = 170
RATE_MIN_GP = 15
PRIOR_TAU = {"beta": 1.0, "gamma": 0.05}
MULT_CLIP = (0.5, 1.6)
S_GRID = np.round(np.arange(0.6, 1.51, 0.1), 2)
N_DRAW = 400
N_HYPER = 4000
VARIANTS = ("V0", "V1", "V2")
HIER = ("V2", "V3", "V4", "V6")  # hierarchical variants
# disagreement components each variant uses (V0 uses none, V1 one pooled weight on D)
COMPONENTS = {"V0": ("D",), "V1": ("D",), "V2": ("D",), "V3": ("Dinfo",), "V4": ("Dinfo", "Dproj"),
              "V6": ("Dinfo", "Dproj_cat")}


def espn_points_rank(df: pd.DataFrame, t: int) -> np.ndarray:
    """Rank (1 = best) of each player in a pure points ranking of ESPN's own projections for season t,
    ESPN default points scoring, season total = points per game x ESPN projected games."""
    e = Pj.espn_proj(t).reindex(df["id"]).fillna(0)
    ppg = (e["pts"] + e["tpm"] + 2 * e["fgm"] - e["fga"] + e["ftm"] - e["fta"] + e["reb"] + 2 * e["ast"]
           + 4 * e["stl"] + 4 * e["blk"] - 2 * e["tov"])
    return (ppg * e["gp"]).rank(ascending=False, method="first").astype(int).to_numpy()


def espn_category_rank(df: pd.DataFrame, t: int) -> np.ndarray:
    """Rank (1 = best) of each player in a 9-category ranking of ESPN's own projections (ESPN per-game lines
    and ESPN projected GP share), scored on the same G-score scale as the model's values."""
    _, scale = M.static_value(df)
    e = Pj.espn_proj(t).reindex(df["id"]).fillna(0)
    mu = np.zeros((len(df), len(M.STATK)))
    mu[:, M.SK["fga"]], mu[:, M.SK["fta"]] = e["fga"], e["fta"]
    mu[:, M.SK["fgx"]] = e["fgm"] - M.LG_FG * e["fga"]
    mu[:, M.SK["ftx"]] = e["ftm"] - M.LG_FT * e["fta"]
    for k in ("tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        mu[:, M.SK[k]] = e[k]
    share = np.clip(e["gp"].to_numpy() / Pj.sgames(t), 0, 1)
    v = M.value_from(mu, share, scale)
    return pd.Series(v).rank(ascending=False, method="first").astype(int).to_numpy()


def disagreement_parts(df: pd.DataFrame, val: np.ndarray, t: int) -> dict:
    """All disagreement components in value units, real-ADP players only (0 otherwise):
    D = M - V (market vs model), Dinfo = M - F (market vs ESPN-projection points rank),
    Dproj = F - V (ESPN-projection points view vs model; mixes format and projection differences),
    Dproj_cat = Fcat - V (ESPN's category view vs model), Dformat = F - Fcat (points vs categories)."""
    srt = np.sort(val)[::-1]
    at = lambda r: srt[np.clip(r - 1, 0, len(val) - 1)]
    Mv = at(df["mkt"].rank(method="first").astype(int).to_numpy())
    F = at(espn_points_rank(df, t))
    Fcat = at(espn_category_rank(df, t))
    real = df["mkt"].to_numpy() <= REAL_ADP_RANK
    z = lambda x: np.where(real, x, 0.0)
    return {"D": z(Mv - val), "Dinfo": z(Mv - F), "Dproj": z(F - val), "Dproj_cat": z(Fcat - val),
            "Dformat": z(F - Fcat)}


def disagreement_matrix(df: pd.DataFrame, val: np.ndarray, t: int, variant: str) -> np.ndarray:
    parts = disagreement_parts(df, val, t)
    return np.stack([parts[c] for c in COMPONENTS[variant]])


def disagreement(df: pd.DataFrame, val: np.ndarray, t: int, variant: str) -> np.ndarray:
    """Market disagreement in value units for real-ADP players (0 otherwise).
    V2: value at market rank minus model value. V3: value at market rank minus value at the rank a pure
    points ranking of ESPN's projections would give (strips format and projection differences)."""
    srt = np.sort(val)[::-1]
    mrank = df["mkt"].rank(method="first").astype(int).to_numpy()
    Mv = srt[np.clip(mrank - 1, 0, len(val) - 1)]
    real = df["mkt"].to_numpy() <= REAL_ADP_RANK
    if variant == "V3":
        F = srt[np.clip(espn_points_rank(df, t) - 1, 0, len(val) - 1)]
        return np.where(real, Mv - F, 0.0)
    return np.where(real, Mv - val, 0.0)


# ---------------------------------------------------------------------------------------------- data
class SeasonData:
    """Per-player quantities for one season; model = stack projection without any market adjustment."""

    def __init__(self, t: int):
        assert M.MARKET_BETA == 0
        s = M.Season(t, "stack")
        df = s.df
        self.t, self.season, self.df = t, s, df
        val, scale = M.static_value(df)
        self.scale, self.V = scale, val
        mrank = df["mkt"].rank(method="first").astype(int).to_numpy()
        self.Mv = np.sort(val)[::-1][np.clip(mrank - 1, 0, len(val) - 1)]
        self.real = df["mkt"].to_numpy() <= REAL_ADP_RANK
        parts = disagreement_parts(df, val, t)
        self.D, self.Dinfo, self.Dproj = parts["D"], parts["Dinfo"], parts["Dproj"]
        self.Dproj_cat, self.Dformat = parts["Dproj_cat"], parts["Dformat"]
        self.mu0 = df[M.STATK].to_numpy(float)
        self.av0 = df["avail"].to_numpy()
        self.A = sum(sign * self.mu0[:, j] * self.av0 / sd for j, m0, sd, sign in scale.values())
        self.eval = df["mkt"].to_numpy() <= EVAL_RANK
        if t >= M.LIVE_SEASON:
            return  # no outcomes yet
        act = Pj.actuals()
        act = act[act.season == t].set_index("id").reindex(df["id"])
        act["fgx"] = act["fgm"] - M.LG_FG * act["fga"]
        act["ftx"] = act["ftm"] - M.LG_FT * act["fta"]
        amu = act[M.STATK].fillna(0).to_numpy()
        self.agp = act["gp"].fillna(0).to_numpy()
        share = self.agp / Pj.sgames(t)
        self.y = M.value_from(amu, share, scale)
        self.r = M.value_from(amu, self.av0, scale) - self.V
        self.g = share - self.av0
        self.rate_ok = self.agp >= RATE_MIN_GP


# ------------------------------------------------------------------------------------- estimation
def ols_slope(x, y):
    """Slope of y on x with intercept; HC1 robust standard error."""
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V = XtXi @ meat @ XtXi * n / (n - k)
    return float(b[1]), float(np.sqrt(V[1, 1]))


def ols_multi(X, y):
    """Coefficients (excluding intercept) of y on columns of X with intercept; HC1 robust se."""
    X = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V = XtXi @ meat @ XtXi * n / (n - k)
    return b[1:].astype(float), np.sqrt(np.diag(V)[1:]).astype(float)


def hier_posterior(b, se, prior_scale, rng, n=N_HYPER):
    """Grid posterior of (mu, tau) for b_s ~ N(mu, tau^2 + se_s^2); flat mu, half-normal(prior_scale) tau.
    Returns samples (mu, tau) of size n."""
    b, se = np.asarray(b, float), np.asarray(se, float)
    span = 4 * se.max() + 3 * prior_scale
    mu_grid = np.linspace(b.min() - span, b.max() + span, 301)
    tau_grid = np.linspace(0, 5 * prior_scale, 301)
    MU, TAU = np.meshgrid(mu_grid, tau_grid, indexing="ij")
    lp = sum(norm.logpdf(bs, MU, np.sqrt(TAU**2 + s**2)) for bs, s in zip(b, se))
    lp = lp + norm.logpdf(TAU, 0, prior_scale)
    p = np.exp(lp - lp.max()).ravel()
    p /= p.sum()
    idx = rng.choice(p.size, n, p=p)
    return MU.ravel()[idx], TAU.ravel()[idx]


def fit_hyper(train: list[SeasonData], variant: str, prior_mult: float = 1.0, seed: int = 0) -> dict:
    if variant == "V0":
        return {"variant": "V0"}
    if variant == "V1":
        xs, ys, Ds = [], [], []
        for i, sd in enumerate(train):
            m = sd.real
            xs.append(sd.D[m])
            ys.append((sd.y - sd.V)[m])
            d = np.zeros((m.sum(), len(train)))
            d[:, i] = 1
            Ds.append(d)
        X = np.c_[np.concatenate(Ds), np.concatenate(xs)]
        beta = float(np.linalg.lstsq(X, np.concatenate(ys), rcond=None)[0][-1])
        return {"variant": "V1", "beta": beta}
    rng = np.random.default_rng(seed)
    comps = COMPONENTS[variant]
    out = {"variant": variant, "per_season": {}}
    for comp, target, mask_fn in (("beta", "r", lambda sd: sd.real & sd.rate_ok), ("gamma", "g", lambda sd: sd.real)):
        est = {c: ([], []) for c in comps}
        for sd in train:
            m = mask_fn(sd)
            X = np.stack([getattr(sd, c)[m] for c in comps], axis=1)
            coefs, ses = ols_multi(X, getattr(sd, target)[m])
            for c, bc, sc in zip(comps, coefs, ses):
                est[c][0].append(bc)
                est[c][1].append(sc)
                out["per_season"].setdefault(sd.t, {}).setdefault(comp, {})[c] = (float(bc), float(sc))
        mus, taus = [], []
        for c in comps:
            mu, tau = hier_posterior(est[c][0], est[c][1], PRIOR_TAU[comp] * prior_mult, rng)
            mus.append(mu)
            taus.append(tau)
        out[f"{comp}_mu"], out[f"{comp}_tau"] = np.array(mus), np.array(taus)  # [K, n]
    return out


# ------------------------------------------------------------------------------------- prediction
def Dmat(sd: "SeasonData", variant: str) -> np.ndarray:
    """Disagreement components [K, P] used by a variant."""
    return np.stack([getattr(sd, c) for c in COMPONENTS[variant]])


def Dof(sd: "SeasonData", variant: str) -> np.ndarray:
    return Dmat(sd, variant)[0]


def mult_from_shift(shift, A):
    """Role multiplier producing a value shift `shift` [..., P] (value is linear in a common rate scale)."""
    safe = np.where(np.abs(A) > 1e-6, A, 1.0)
    m = 1 + shift / safe
    return np.clip(np.where(np.abs(A) > 1e-6, m, 1.0), *MULT_CLIP)


def multiplier(beta, D, A):
    """Role multiplier hitting value shift beta * D (value is linear in a common scale of all rates)."""
    safe = np.where(np.abs(A) > 1e-6, A, 1.0)
    m = 1 + np.multiply.outer(beta, D) / safe
    return np.clip(np.where(np.abs(A) > 1e-6, m, 1.0), *MULT_CLIP)


def market_draws(hyper: dict, n: int, rng):
    """Per-simulated-season weight draws: beta [K, n], gamma [K, n] (one shared shock per season)."""
    v = hyper["variant"]
    if v == "V0":
        return np.zeros((1, n)), np.zeros((1, n))
    if v == "V1":
        return np.full((1, n), hyper["beta"]), np.zeros((1, n))
    assert v in HIER
    bmu, btau = np.atleast_2d(np.asarray(hyper["beta_mu"])), np.atleast_2d(np.asarray(hyper["beta_tau"]))
    gmu, gtau = np.atleast_2d(np.asarray(hyper["gamma_mu"])), np.atleast_2d(np.asarray(hyper["gamma_tau"]))
    eps = rng.standard_normal(n)
    bi = [rng.integers(0, bmu.shape[1], n) for _ in range(bmu.shape[0])]
    gi = [rng.integers(0, gmu.shape[1], n) for _ in range(gmu.shape[0])]
    beta = np.stack([bmu[k][bi[k]] + btau[k][bi[k]] * eps for k in range(bmu.shape[0])])
    gamma = np.stack([gmu[k][gi[k]] + gtau[k][gi[k]] * eps for k in range(gmu.shape[0])])
    return beta, gamma


def predictive_values(sd: SeasonData, hyper: dict, s_v: float, seed: int, n: int = N_DRAW) -> np.ndarray:
    """Draws [n, P] of each player's season value under a belief model."""
    rng = np.random.default_rng(seed)
    err, df = sd.season.err, sd.df
    P = len(df)
    L = np.linalg.cholesky(err["corr"] + 1e-9 * np.eye(len(Pj.ERR_STATS)))
    z = (rng.standard_normal((n, P, len(Pj.ERR_STATS))) @ L.T) * s_v
    raw = {}
    for j, k in enumerate(Pj.ERR_STATS):
        f = df[k].to_numpy()
        raw[k] = np.maximum(f + err["scale"][k] * np.sqrt(np.maximum(f, 0.05)) * z[..., j], 0)
    fgp = (df["fgm"] / df["fga"].clip(lower=0.1)).to_numpy() + err["fg_sd"] * s_v * rng.standard_normal((n, P))
    ftp = (df["ftm"] / df["fta"].clip(lower=0.1)).to_numpy() + err["ft_sd"] * s_v * rng.standard_normal((n, P))
    mu = np.zeros((n, P, len(M.STATK)))
    mu[..., M.SK["fga"]], mu[..., M.SK["fta"]] = raw["fga"], raw["fta"]
    mu[..., M.SK["fgx"]] = raw["fga"] * (np.clip(fgp, 0.3, 0.75) - M.LG_FG)
    mu[..., M.SK["ftx"]] = raw["fta"] * (np.clip(ftp, 0.3, 0.98) - M.LG_FT)
    for k in ("tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        mu[..., M.SK[k]] = raw[k]
    resid = rng.choice(err["gp_resid"], (n, P)) * s_v
    beta, gamma = market_draws(hyper, n, rng)
    Dm = Dmat(sd, hyper["variant"])
    mu = mu * mult_from_shift(beta.T @ Dm, sd.A)[..., None]
    avail = np.clip(sd.av0 + resid + gamma.T @ Dm, 0, 1)
    return M.value_from(mu, avail, sd.scale)


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per column: E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(samples, axis=0)
    n = x.shape[0]
    e1 = np.abs(x - y).mean(0)
    w = 2 * np.arange(1, n + 1) - n - 1
    e2 = 2 * (w[:, None] * x).sum(0) / n**2
    return e1 - 0.5 * e2


def choose_s(train: list[SeasonData], hyper: dict, seed: int = 11) -> float:
    best, best_c = None, np.inf
    for s in S_GRID:
        c = np.mean([crps(predictive_values(sd, hyper, s, seed + k), sd.y)[sd.eval].mean() for k, sd in enumerate(train)])
        if c < best_c:
            best, best_c = float(s), c
    return best


# ------------------------------------------------------------------------------------- evaluation
def score_heldout(sd: SeasonData, hyper: dict, s_v: float, seed: int) -> dict:
    draws = predictive_values(sd, hyper, s_v, seed)
    c = crps(draws, sd.y)
    mean, sdv = draws.mean(0), draws.std(0)
    zres = (sd.y - mean) / np.maximum(sdv, 1e-6)
    m = sd.real
    slope, se = ols_slope(sd.D[m], zres[m])
    slope_i, se_i = ols_slope(sd.Dinfo[m], zres[m])
    slope_p, se_p = ols_slope(sd.Dproj[m], (sd.y - mean)[m])  # over-deviation from ESPN's view (target ~0)
    cov = {}
    for lv in (0.5, 0.8, 0.95):
        lo, hi = np.quantile(draws, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        cov[lv] = float(((sd.y >= lo) & (sd.y <= hi))[sd.eval].mean())
    return {"crps_players": c[sd.eval], "crps": float(c[sd.eval].mean()), "wc_slope": slope, "wc_se": se,
            "wc_slope_info": slope_i, "wc_se_info": se_i, "proj_slope": slope_p, "proj_se": se_p, "coverage": cov}


def run_split(data: dict, held: int, train_seasons, variant: str, prior_mult: float = 1.0) -> dict:
    train = [data[t] for t in train_seasons]
    hyper = fit_hyper(train, variant, prior_mult, seed=held)
    s_v = choose_s(train, hyper)
    res = score_heldout(data[held], hyper, s_v, seed=1000 + held)
    res.update(s_v=s_v, hyper_summary=summarize(hyper))
    return res


def summarize(h: dict) -> dict:
    if h["variant"] == "V0":
        return {}
    if h["variant"] == "V1":
        return {"beta": round(h["beta"], 3)}
    out = {}
    comps = COMPONENTS[h["variant"]]
    for comp in ("beta", "gamma"):
        mu, tau = np.atleast_2d(h[f"{comp}_mu"]), np.atleast_2d(h[f"{comp}_tau"])
        stats = {c: {"mu_mean": round(float(mu[k].mean()), 4), "tau_mean": round(float(tau[k].mean()), 4),
                     "pred_sd": round(float(np.sqrt(mu[k].var() + (tau[k] ** 2).mean())), 4)} for k, c in enumerate(comps)}
        out[comp] = stats[comps[0]] if len(comps) == 1 else stats
    out["per_season"] = {int(t): {comp: {c: [round(x, 4) for x in v] for c, v in d.items()} for comp, d in dd.items()}
                         for t, dd in h["per_season"].items()}
    return out


def evaluate():
    data = {t: SeasonData(t) for t in CLEAN_SEASONS}
    rows, detail = [], {}
    splits = {"loso": [(t, [o for o in CLEAN_SEASONS if o != t]) for t in CLEAN_SEASONS],
              "forward": [(2024, [2022]), (2026, [2022, 2024])]}
    for scheme, sp in splits.items():
        for held, train in sp:
            for v in VARIANTS:
                r = run_split(data, held, train, v)
                detail[(scheme, held, v)] = r
                rows.append({"scheme": scheme, "held_out": held, "variant": v, "crps": r["crps"], "s_v": r["s_v"],
                             "wc_slope": r["wc_slope"], "wc_se": r["wc_se"], **{f"cov{int(k * 100)}": x for k, x in r["coverage"].items()}})
                print(f"{scheme} {held} {v}: CRPS {r['crps']:.4f}  s_v {r['s_v']}  {r['hyper_summary']}", flush=True)
    sens = []
    for pm in (0.5, 2.0):
        for held, train in splits["loso"]:
            r = run_split(data, held, train, "V2", prior_mult=pm)
            sens.append({"prior_mult": pm, "held_out": held, "crps": r["crps"]})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== PRIMARY: leave-one-season-out mean CRPS (lower is better) ===")
    loso = res[res.scheme == "loso"].pivot(index="variant", columns="held_out", values="crps")
    loso["MEAN"] = loso.mean(1)
    print(loso.round(4).to_string())
    winner = loso["MEAN"].idxmin()
    print(f"\nRegistered decision rule -> SHIP {winner}")
    print("\npaired player-level differences (held-out season, variant minus V0; negative = better):")
    for held, _ in splits["loso"]:
        base = detail[("loso", held, "V0")]["crps_players"]
        line = []
        for v in ("V1", "V2"):
            d = detail[("loso", held, v)]["crps_players"] - base
            line.append(f"{v} {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
        print(f"  {held}: " + "   ".join(line))
    print("\nsecondary: forward-chained CRPS")
    print(res[res.scheme == "forward"].pivot(index="variant", columns="held_out", values="crps").round(4).to_string())
    print("\nwinner's-curse check (held-out standardized residual vs disagreement; want CI to include 0):")
    w = res[res.scheme == "loso"][["variant", "held_out", "wc_slope", "wc_se"]].copy()
    w["ci"] = w.apply(lambda r: f"[{r.wc_slope - 1.96 * r.wc_se:+.3f}, {r.wc_slope + 1.96 * r.wc_se:+.3f}]", axis=1)
    print(w.round(3).to_string(index=False))
    print("\ncoverage of predictive intervals (nominal 0.50 / 0.80 / 0.95), LOSO:")
    print(res[res.scheme == "loso"].groupby("variant")[["cov50", "cov80", "cov95"]].mean().round(3).to_string())
    print("\nV2 prior sensitivity (LOSO mean CRPS):")
    s = pd.DataFrame(sens).groupby("prior_mult").crps.mean()
    s.loc[1.0] = loso.loc["V2", "MEAN"]
    print(s.sort_index().round(4).to_string())
    out = {"loso": loso.round(5).to_dict(), "winner": winner, "rows": res.to_dict("records"),
           "prior_sensitivity": s.round(5).to_dict()}
    (M.DATA / "belief_eval.json").write_text(json.dumps(out, default=str, indent=1))


def shipped_variant() -> str:
    """Shipped belief variant; tracked config at the repo root (DECISIONS.md records each change)."""
    p = M.DATA.parent / "belief_shipped.json"
    return json.loads(p.read_text())["variant"]


def load_for_season(t: int) -> dict:
    """Belief hyperparameters of the shipped variant for season t: the final fit, or for a clean season a
    fit that excludes it, or for a historical non-clean season a fit on clean seasons before it."""
    v = shipped_variant()
    if t in CLEAN_SEASONS:
        path = M.DATA / f"belief_loso_{v}_{t}.json"
        if not path.exists():
            fit_final(v, exclude=t, path=path)
    elif t < M.LIVE_SEASON:
        path = M.DATA / f"belief_prior_{v}_{t}.json"
        if not path.exists():
            data = [SeasonData(s) for s in CLEAN_SEASONS if s < t]
            hyper = fit_hyper(data, v, seed=7)
            final = {"variant": v, "s_v": choose_s(data, hyper), "seasons": [s for s in CLEAN_SEASONS if s < t]}
            for k in ("beta_mu", "beta_tau", "gamma_mu", "gamma_tau"):
                final[k] = hyper[k].tolist()
            path.write_text(json.dumps(final))
    else:
        path = M.DATA / f"belief_final_{v}.json"
        if not path.exists():
            fit_final(v, path=path)
    return json.loads(path.read_text())


def fit_final(variant: str | None = None, exclude: int | None = None, path=None, prior_mult: float = 1.0) -> dict:
    """Fit a variant on the clean seasons (optionally excluding one, for descriptive backtests)."""
    if variant is None:
        variant = json.loads((M.DATA / "belief_eval.json").read_text())["winner"]
    seasons = [t for t in CLEAN_SEASONS if t != exclude]
    data = [SeasonData(t) for t in seasons]
    hyper = fit_hyper(data, variant, prior_mult=prior_mult, seed=7)
    s_v = choose_s(data, hyper)
    final = {"variant": variant, "s_v": s_v, "seasons": seasons}
    if variant == "V1":
        final["beta"] = hyper["beta"]
    if variant in HIER:
        for k in ("beta_mu", "beta_tau", "gamma_mu", "gamma_tau"):
            final[k] = hyper[k].tolist()
    final["summary"] = summarize(hyper)
    if path is not None:
        path.write_text(json.dumps(final))
    return final


def evaluate_v3():
    """Registered V3 vs V2 comparison (same splits and seeds as the main comparison)."""
    data = {t: SeasonData(t) for t in CLEAN_SEASONS}
    splits = {"loso": [(t, [o for o in CLEAN_SEASONS if o != t]) for t in CLEAN_SEASONS],
              "forward": [(2024, [2022]), (2026, [2022, 2024])]}
    rows, det = [], {}
    for scheme, sp in splits.items():
        for held, train in sp:
            for v in ("V2", "V3"):
                r = run_split(data, held, train, v)
                det[(scheme, held, v)] = r
                rows.append({"scheme": scheme, "held_out": held, "variant": v, "crps": r["crps"], "s_v": r["s_v"],
                             "wc_D": f"{r['wc_slope']:+.3f}±{r['wc_se']:.3f}",
                             "wc_Dinfo": f"{r['wc_slope_info']:+.3f}±{r['wc_se_info']:.3f}",
                             **{f"cov{int(k * 100)}": x for k, x in r["coverage"].items()}})
                print(scheme, held, v, round(r["crps"], 4), r["hyper_summary"].get("beta"), flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    lo = res[res.scheme == "loso"].pivot(index="variant", columns="held_out", values="crps")
    lo["MEAN"] = lo.mean(axis=1)
    fw = res[res.scheme == "forward"].pivot(index="variant", columns="held_out", values="crps")
    fw["MEAN"] = fw.mean(axis=1)
    print("\nLOSO CRPS:"); print(lo.round(4).to_string())
    print("\nforward-chained CRPS:"); print(fw.round(4).to_string())
    for held in CLEAN_SEASONS:
        d = det[("loso", held, "V3")]["crps_players"] - det[("loso", held, "V2")]["crps_players"]
        print(f"  {held}: V3 - V2 = {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    print("\nwinner's-curse diagnostics and coverage:")
    print(res[res.scheme == "loso"][["variant", "held_out", "wc_D", "wc_Dinfo", "cov50", "cov80", "cov95"]].to_string(index=False))
    a = lo.loc["V3", "MEAN"] < lo.loc["V2", "MEAN"]
    b_ = (lo.loc["V3", list(CLEAN_SEASONS)] < lo.loc["V2", list(CLEAN_SEASONS)]).sum() >= 2
    c = fw.loc["V3", "MEAN"] <= fw.loc["V2", "MEAN"]
    verdict = "V3" if (a and b_ and c) else "V2"
    print(f"\nmultiplicity guard: pooled better={a}, >=2 of 3 seasons={b_}, forward not worse={c} -> SHIP {verdict}")
    (M.DATA / "belief_eval_v3.json").write_text(json.dumps({"loso": lo.round(5).to_dict(), "forward": fw.round(5).to_dict(),
                                                            "verdict": verdict}, default=str, indent=1))
    return verdict


def evaluate_v4():
    """Registered V4 vs V3 comparison (same splits and seeds), plus the D_proj over-deviation diagnostic."""
    data = {t: SeasonData(t) for t in CLEAN_SEASONS}
    splits = {"loso": [(t, [o for o in CLEAN_SEASONS if o != t]) for t in CLEAN_SEASONS],
              "forward": [(2024, [2022]), (2026, [2022, 2024])]}
    rows, det = [], {}
    for scheme, sp in splits.items():
        for held, train in sp:
            for v in ("V3", "V4"):
                r = run_split(data, held, train, v)
                det[(scheme, held, v)] = r
                rows.append({"scheme": scheme, "held_out": held, "variant": v, "crps": r["crps"], "s_v": r["s_v"],
                             "proj_slope": f"{r['proj_slope']:+.3f}±{r['proj_se']:.3f}",
                             **{f"cov{int(k * 100)}": x for k, x in r["coverage"].items()}})
                print(scheme, held, v, round(r["crps"], 4), flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    lo = res[res.scheme == "loso"].pivot(index="variant", columns="held_out", values="crps")
    lo["MEAN"] = lo.mean(axis=1)
    fw = res[res.scheme == "forward"].pivot(index="variant", columns="held_out", values="crps")
    fw["MEAN"] = fw.mean(axis=1)
    print("\nLOSO CRPS:"); print(lo.round(4).to_string())
    print("\nforward-chained CRPS:"); print(fw.round(4).to_string())
    for held in CLEAN_SEASONS:
        d = det[("loso", held, "V4")]["crps_players"] - det[("loso", held, "V3")]["crps_players"]
        print(f"  {held}: V4 - V3 = {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    print("\nover-deviation diagnostic (held-out value residual vs D_proj; target ~0) and coverage:")
    print(res[res.scheme == "loso"][["variant", "held_out", "proj_slope", "cov50", "cov80", "cov95"]].to_string(index=False))
    full = fit_hyper([data[t] for t in CLEAN_SEASONS], "V4", seed=7)
    print("\nV4 fit on all seasons:", json.dumps(summarize(full), indent=1))
    a = lo.loc["V4", "MEAN"] < lo.loc["V3", "MEAN"]
    b_ = (lo.loc["V4", list(CLEAN_SEASONS)] < lo.loc["V3", list(CLEAN_SEASONS)]).sum() >= 2
    c = fw.loc["V4", "MEAN"] <= fw.loc["V3", "MEAN"]
    verdict = "V4" if (a and b_ and c) else "V3"
    print(f"\nmultiplicity guard: pooled better={a}, >=2 of 3 seasons={b_}, forward not worse={c} -> SHIP {verdict}")
    (M.DATA / "belief_eval_v4.json").write_text(json.dumps({"loso": lo.round(5).to_dict(), "forward": fw.round(5).to_dict(),
                                                            "verdict": verdict}, default=str, indent=1))
    return verdict


def evaluate_v5b():
    """Registered (post hoc) V5b stage-2 test: V4 belief on the DARKO-augmented stack (ast, FG%, FT%)
    vs V4 on the current stack. Same splits and seeds as earlier comparisons."""
    splits = {"loso": [(t, [o for o in CLEAN_SEASONS if o != t]) for t in CLEAN_SEASONS],
              "forward": [(2024, [2022]), (2026, [2022, 2024])]}
    det, rows = {}, []
    for label, flag in (("V4", False), ("V5b", True)):
        Pj.set_darko(flag, stats=("ast",))
        data = {t: SeasonData(t) for t in CLEAN_SEASONS}
        for scheme, sp in splits.items():
            for held, train in sp:
                r = run_split(data, held, train, "V4")
                det[(scheme, held, label)] = r
                rows.append({"scheme": scheme, "held_out": held, "variant": label, "crps": r["crps"],
                             "cov50": r["coverage"][0.5], "cov80": r["coverage"][0.8], "cov95": r["coverage"][0.95]})
                print(scheme, held, label, round(r["crps"], 4), flush=True)
    Pj.set_darko(False)
    res = pd.DataFrame(rows)
    lo = res[res.scheme == "loso"].pivot(index="variant", columns="held_out", values="crps")
    lo["MEAN"] = lo.mean(axis=1)
    fw = res[res.scheme == "forward"].pivot(index="variant", columns="held_out", values="crps")
    fw["MEAN"] = fw.mean(axis=1)
    pd.set_option("display.width", 220)
    print("\nLOSO CRPS:"); print(lo.round(4).to_string())
    print("\nforward-chained CRPS:"); print(fw.round(4).to_string())
    for held in CLEAN_SEASONS:
        d = det[("loso", held, "V5b")]["crps_players"] - det[("loso", held, "V4")]["crps_players"]
        print(f"  {held}: V5b - V4 = {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    print("\ncoverage (LOSO):"); print(res[res.scheme == "loso"].groupby("variant")[["cov50", "cov80", "cov95"]].mean().round(3).to_string())
    a = lo.loc["V5b", "MEAN"] < lo.loc["V4", "MEAN"]
    b_ = (lo.loc["V5b", list(CLEAN_SEASONS)] < lo.loc["V4", list(CLEAN_SEASONS)]).sum() >= 2
    c = fw.loc["V5b", "MEAN"] <= fw.loc["V4", "MEAN"]
    verdict = "V5b" if (a and b_ and c) else "V4"
    print(f"\nmultiplicity guard: pooled better={a}, >=2 of 3 seasons={b_}, forward not worse={c} -> SHIP {verdict}")
    return verdict


def evaluate_v6():
    """Registered V6 vs V4 comparison (same splits and seeds), with format diagnostics."""
    data = {t: SeasonData(t) for t in CLEAN_SEASONS}
    splits = {"loso": [(t, [o for o in CLEAN_SEASONS if o != t]) for t in CLEAN_SEASONS],
              "forward": [(2024, [2022]), (2026, [2022, 2024])]}
    rows, det = [], {}
    for scheme, sp in splits.items():
        for held, train in sp:
            for v in ("V4", "V6"):
                r = run_split(data, held, train, v)
                det[(scheme, held, v)] = r
                rows.append({"scheme": scheme, "held_out": held, "variant": v, "crps": r["crps"],
                             **{f"cov{int(k * 100)}": x for k, x in r["coverage"].items()}})
                print(scheme, held, v, round(r["crps"], 4), flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    lo = res[res.scheme == "loso"].pivot(index="variant", columns="held_out", values="crps")
    lo["MEAN"] = lo.mean(axis=1)
    fw = res[res.scheme == "forward"].pivot(index="variant", columns="held_out", values="crps")
    fw["MEAN"] = fw.mean(axis=1)
    print("\nLOSO CRPS:"); print(lo.round(4).to_string())
    print("\nforward-chained CRPS:"); print(fw.round(4).to_string())
    for held in CLEAN_SEASONS:
        d = det[("loso", held, "V6")]["crps_players"] - det[("loso", held, "V4")]["crps_players"]
        print(f"  {held}: V6 - V4 = {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    print("\ncoverage (LOSO):"); print(res[res.scheme == "loso"].groupby("variant")[["cov50", "cov80", "cov95"]].mean().round(3).to_string())
    print("\ndiagnostics: held-out V6 value residual slopes (target ~0):")
    for held in CLEAN_SEASONS:
        sd = data[held]
        train = [data[o] for o in CLEAN_SEASONS if o != held]
        h = fit_hyper(train, "V6", seed=held)
        dr = predictive_values(sd, h, choose_s(train, h), seed=1000 + held)
        resid = sd.y - dr.mean(0)
        m = sd.real
        f, fs = ols_slope(sd.Dformat[m], resid[m])
        p, ps = ols_slope(sd.Dproj_cat[m], resid[m])
        print(f"  {held}: vs D_format {f:+.3f} ± {fs:.3f}   vs D_proj_cat {p:+.3f} ± {ps:.3f}")
    full = fit_hyper([data[t] for t in CLEAN_SEASONS], "V6", seed=7)
    print("\nV6 fit on all seasons:", json.dumps({k: v for k, v in summarize(full).items() if k != "per_season"}, indent=1))
    a = lo.loc["V6", "MEAN"] < lo.loc["V4", "MEAN"]
    b_ = (lo.loc["V6", list(CLEAN_SEASONS)] < lo.loc["V4", list(CLEAN_SEASONS)]).sum() >= 2
    c = fw.loc["V6", "MEAN"] <= fw.loc["V4", "MEAN"]
    verdict = "V6" if (a and b_ and c) else "V4"
    print(f"\nmultiplicity guard: pooled better={a}, >=2 of 3 seasons={b_}, forward not worse={c} -> SHIP {verdict}")
    return verdict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "evaluate":
        evaluate()
    elif cmd == "evaluate_v3":
        evaluate_v3()
    elif cmd == "evaluate_v4":
        evaluate_v4()
    elif cmd == "evaluate_v5b":
        evaluate_v5b()
    elif cmd == "evaluate_v6":
        evaluate_v6()
    elif cmd == "fit":
        v = shipped_variant()
        f = fit_final(v, path=M.DATA / f"belief_final_{v}.json")
        print(json.dumps({k: v for k, v in f.items() if not k.endswith(("_mu", "_tau"))}, indent=1))
