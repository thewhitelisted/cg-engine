"""Projection model: ESPN preseason projection stacked with a Marcel-style history projection.

Marcel: per-minute rates from the previous 3 seasons weighted 5/4/3 by minutes, regressed toward the
league-average rate with REG_MIN phantom minutes; minutes per game from weighted history.
Stack: y = a*ESPN + b*Marcel + c*Marcel*old + d*Marcel*young  (per stat, no intercept), where
old = max(age-30, 0), young = max(24-age, 0). GP: y/season_games = linear in ESPN proj GP share,
last-2 GP shares and age. All coefficients are fit by least squares on seasons strictly
before the target season (or on all available seasons for the live 2027 projection).

    python3 projections.py      # leave-one-season-out CV report
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fgm", "fga", "ftm", "fta"]
SEASON_GAMES = {2020: 72, 2021: 72}  # shortened seasons (2019-20 varied 63-75; 72 is close enough)
REG_MIN = 600                        # phantom minutes at league-average rate
# Candidate V5 (DECISIONS.md 2026-10-06): DARKO preseason inputs for points, assists, FG%, FT%. Off until accepted.
USE_DARKO = False
DARKO_STATS = ("pts", "ast")


# Candidate P1 (DECISIONS.md provenance review): "marcel" = fixed 5/4/3 weights + 24/30 age hinges (shipped);
# "split" = the stack fits its own weight for each of the last three seasons, age as a quadratic in (age - 27).
HIST_MODE = "marcel"


def set_hist_mode(mode: str):
    global HIST_MODE
    HIST_MODE = mode


def set_darko(on: bool, stats=("pts", "ast")):
    """Enable DARKO inputs; `stats` = counting stats that get the DARKO regressor (FG%/FT% blend always on)."""
    global USE_DARKO, DARKO_STATS
    USE_DARKO, DARKO_STATS = on, tuple(stats)
FIRST, LAST = 2020, 2027


def sgames(s):
    return SEASON_GAMES.get(s, 82)


@lru_cache(maxsize=1)
def files():
    return {s: pd.read_csv(DATA / f"players_{s}.csv").set_index("id") for s in range(FIRST, LAST + 1)}


@lru_cache(maxsize=1)
def actuals() -> pd.DataFrame:
    """Long table of actual season lines: id, season, gp, min, per-game stats."""
    rows = []
    F = files()
    for s in range(FIRST, LAST):
        src = F[s + 1] if s == LAST - 1 else F[s]  # act2026 lives in the 2027 file
        cols = [f"act{s}_{k}" for k in ["gp", "min"] + STATS]
        d = src[[c for c in cols if c in src]].copy()
        d.columns = [c.split("_", 1)[1] for c in d.columns]
        d = d[d["gp"].fillna(0) > 0].fillna(0)
        d["season"] = s
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


@lru_cache(maxsize=1)
def birthdates() -> pd.Series:
    b = pd.read_csv(DATA / "birthdates.csv").dropna()
    return pd.to_datetime(b.set_index("id")["dob"]).dt.tz_localize(None)


def espn_proj(t: int) -> pd.DataFrame:
    F = files()
    src = F[t + 1] if t + 1 in F and f"proj{t}_pts" in F[t + 1] and t < LAST else F[t]
    if f"proj{t}_pts" not in src:
        src = F[t]
    cols = {f"proj{t}_{k}": k for k in ["gp", "min"] + STATS}
    d = src[[c for c in cols if c in src]].rename(columns=cols)
    d = d[d["pts"].notna() & (d["min"].fillna(0) > 0)].copy()
    return d.fillna(0)


def marcel(t: int, ids) -> pd.DataFrame:
    """History-only projection for season t from seasons t-1..t-3."""
    A = actuals()
    H = A[(A["season"] < t) & (A["season"] >= t - 3)]
    w = H["season"].map({t - 1: 5, t - 2: 4, t - 3: 3})
    tot_min = H["min"] * H["gp"]
    lg = {k: (H[k] * H["gp"]).sum() / tot_min.sum() for k in STATS}  # league per-minute rates
    out = pd.DataFrame(index=pd.Index(ids, name="id"))
    W = (w * tot_min).groupby(H["id"]).sum()
    for k in STATS:
        num = (w * H[k] * H["gp"]).groupby(H["id"]).sum()  # w * per-min rate * minutes
        rate = (num + REG_MIN * lg[k] * 4) / (W + REG_MIN * 4)
        out[f"rate_{k}"] = rate
    gw = (w * H["gp"]).groupby(H["id"]).sum()
    out["mpg"] = (w * H["gp"] * H["min"]).groupby(H["id"]).sum() / gw
    for lag in (1, 2):
        s = A[A["season"] == t - lag].set_index("id")
        out[f"gp_share{lag}"] = (s["gp"] / sgames(t - lag)).reindex(out.index)
    out["n_hist"] = H.groupby("id").size().reindex(out.index).fillna(0)
    for k in STATS:
        out[k] = out[f"rate_{k}"] * out["mpg"]
    # single-season regressed rates (candidate P1); a missing season falls back to the composite rate
    for lag in (1, 2, 3):
        Hs = H[H["season"] == t - lag]
        mins = (Hs["min"] * Hs["gp"]).groupby(Hs["id"]).sum()
        for k in STATS:
            num = (Hs[k] * Hs["gp"]).groupby(Hs["id"]).sum()
            r = ((num + REG_MIN * lg[k]) / (mins + REG_MIN)).reindex(out.index)
            out[f"{k}_h{lag}"] = r.fillna(out[f"rate_{k}"]) * out["mpg"]
    return out


def _name_key(n: str) -> str:
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


# DARKO full names -> ESPN display names (normalized keys); fixed before any evaluation was run
DARKO_ALIASES = {"alexandresarr": "alexsarr", "nicolasclaxton": "nicclaxton", "carltoncarrington": "bubcarrington",
                 "kenyonmartin": "kjmartin"}


@lru_cache(maxsize=16)
def darko(t: int) -> pd.DataFrame:
    """DARKO preseason snapshot for season t, indexed by ESPN id (per-minute pts/ast, FG%, FT%)."""
    path = DATA / f"darko_{t}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["d_pts_pm", "d_ast_pm", "d_fg", "d_ft"])
    dk = pd.read_csv(path)
    dk["k"] = dk["player_name"].map(_name_key).replace(DARKO_ALIASES)
    dk = dk.drop_duplicates("k", keep=False)
    names = files()[t]["name"]
    ids = pd.Series(names.index, index=names.map(_name_key).to_numpy())
    ids = ids[~ids.index.duplicated(keep=False)]
    dk["id"] = dk["k"].map(ids)
    dk = dk.dropna(subset=["id"]).set_index(dk.dropna(subset=["id"])["id"].astype(int))
    per_min = dk["x_pace"] / 100 / 48
    return pd.DataFrame({"d_pts_pm": dk["x_pts_100"] * per_min, "d_ast_pm": dk["x_ast_100"] * per_min,
                         "d_fg": dk["x_fg_pct"], "d_ft": dk["x_ft_pct"]})


def features(t: int) -> pd.DataFrame:
    e = espn_proj(t)
    m = marcel(t, e.index)
    d = e.add_prefix("e_").join(m.add_prefix("m_"))
    if USE_DARKO:
        dk = darko(t).reindex(d.index)
        d["d_pts"] = dk["d_pts_pm"] * d["e_min"]
        d["d_ast"] = dk["d_ast_pm"] * d["e_min"]
        d["d_fg"], d["d_ft"] = dk["d_fg"], dk["d_ft"]
    bd = birthdates().reindex(d.index)
    d["age"] = ((pd.Timestamp(f"{t - 1}-10-20") - bd).dt.days / 365.25).fillna(25)
    d["old"] = np.maximum(d["age"] - 30, 0)
    d["young"] = np.maximum(24 - d["age"], 0)
    d["has_hist"] = d["m_n_hist"].fillna(0) > 0
    d["season"] = t
    return d


def stack_X(d: pd.DataFrame, k: str) -> np.ndarray:
    m = d[f"m_{k}"].where(d["has_hist"], d[f"e_{k}"]).fillna(0).to_numpy()
    if HIST_MODE == "split":
        a = (d["age"] - 27).to_numpy()
        hs = [d[f"m_{k}_h{lag}"].where(d["has_hist"], d[f"e_{k}"]).fillna(0).to_numpy() for lag in (1, 2, 3)]
        X = np.c_[d[f"e_{k}"].to_numpy(), *hs, m * a, m * a**2]
    else:
        X = np.c_[d[f"e_{k}"].to_numpy(), m, m * d["old"], m * d["young"]]
    if USE_DARKO and k in DARKO_STATS:
        X = np.c_[X, d[f"d_{k}"].fillna(d[f"e_{k}"]).fillna(0).to_numpy()]
    return X


def gp_X(d: pd.DataFrame) -> np.ndarray:
    t = d["season"].to_numpy()
    e = d["e_gp"].to_numpy() / np.vectorize(sgames)(t)
    g1 = d["m_gp_share1"].fillna(-1).to_numpy()
    g2 = d["m_gp_share2"].fillna(-1).to_numpy()
    h1, h2 = (g1 >= 0).astype(float), (g2 >= 0).astype(float)
    return np.c_[np.ones(len(d)), e, np.maximum(g1, 0), h1, np.maximum(g2, 0), h2, d["old"], d["young"]]


def panel(seasons) -> pd.DataFrame:
    A = actuals().set_index(["id", "season"])
    out = []
    for t in seasons:
        d = features(t)
        y = A.xs(t, level="season").reindex(d.index) if t < LAST else None
        if y is not None:
            d = d.join(y.add_prefix("y_"))
            d["y_gp"] = d["y_gp"].fillna(0)
        out.append(d)
    return pd.concat(out)


def rate_mask(d):
    return (d["y_gp"] >= 20) & (d["y_min"] >= 15)


def gp_mask(d):
    return d["e_min"] >= 20


def fit(train: pd.DataFrame, calibrate: bool | None = None) -> dict:
    coef = {}
    r = train[rate_mask(train)]
    for k in STATS:
        coef[k] = np.linalg.lstsq(stack_X(r, k), r[f"y_{k}"].to_numpy(), rcond=None)[0]
    # ESPN's 2022-23 GP projections in the API average ~38 (vs ~70 every other year): unusable
    g = train[gp_mask(train) & (train["season"] != 2023)]
    if len(g) < 100:  # nothing usable to fit on: ESPN share scaled by its typical over-projection
        coef["gp"] = np.array([0, 0.83, 0, 0, 0, 0, 0, 0], float)
    else:
        y = g["y_gp"].to_numpy() / np.vectorize(sgames)(g["season"].to_numpy())
        coef["gp"] = np.linalg.lstsq(gp_X(g), y, rcond=None)[0]
    use = GP_CALIBRATION if calibrate is None else calibrate
    coef["gp_cal"] = gp_calibration(train) if use else (0.0, 1.0)
    if USE_DARKO:
        # blend weight between the stack's shooting % and DARKO's, fit on the training seasons
        pr = predict(r, coef)
        for made, att, dcol, min_att in (("fgm", "fga", "d_fg", 5), ("ftm", "fta", "d_ft", 2)):
            ok = (r[f"y_{att}"] >= min_att) & r[dcol].notna() & (pr[att] > 0)
            y = (r.loc[ok, f"y_{made}"] / r.loc[ok, f"y_{att}"]).to_numpy()
            sp = (pr.loc[ok, made] / pr.loc[ok, att]).to_numpy()
            dp = r.loc[ok, dcol].to_numpy()
            w = float(np.clip(((y - dp) * (sp - dp)).sum() / max(((sp - dp) ** 2).sum(), 1e-12), 0, 1))
            coef[f"w_{made}"] = w
    return coef


# Candidate 7: shrink GP-share predictions with a calibration line learned from inner LOSO predictions
# within the training seasons. Off by default (v1/v2 behaviour); enabled via set_gp_calibration(True).
GP_CALIBRATION = False


def set_gp_calibration(on: bool):
    global GP_CALIBRATION
    GP_CALIBRATION = on
    project.cache_clear() if hasattr(project, "cache_clear") else None


def gp_calibration(train: pd.DataFrame) -> tuple[float, float]:
    """(a, b) with actual_share ~ a + b * predicted_share, from inner leave-one-season-out predictions."""
    seasons = sorted(s for s in train["season"].unique() if s != 2023)
    if len(seasons) < 2:
        return 0.0, 1.0
    preds, acts = [], []
    for s in seasons:
        inner = train[train["season"] != s]
        if len(inner[gp_mask(inner) & (inner["season"] != 2023)]) < 100:
            continue
        te = train[(train["season"] == s) & gp_mask(train)]
        c = fit(inner, calibrate=False)
        preds.append(predict(te, c)["gp_share"].to_numpy())
        acts.append(te["y_gp"].to_numpy() / sgames(s))
    if not preds:
        return 0.0, 1.0
    b, a = np.polyfit(np.concatenate(preds), np.concatenate(acts), 1)
    return float(a), float(b)


def predict(d: pd.DataFrame, coef: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for k in STATS:
        out[k] = np.maximum(stack_X(d, k) @ coef[k], 0)
    out["fgm"] = np.minimum(out["fgm"], out["fga"])
    out["ftm"] = np.minimum(out["ftm"], out["fta"])
    for made, att, dcol in (("fgm", "fga", "d_fg"), ("ftm", "fta", "d_ft")):
        if USE_DARKO and f"w_{made}" in coef and dcol in d:
            w = coef[f"w_{made}"]
            sp = out[made] / out[att].where(out[att] > 0)
            pct = (w * sp + (1 - w) * d[dcol]).where(d[dcol].notna(), sp)
            out[made] = (out[att] * pct).fillna(out[made])
    a, b = coef.get("gp_cal", (0.0, 1.0))
    share = np.clip(a + b * (gp_X(d) @ coef["gp"]), 0.02, 1.0)
    out["gp"] = share * np.vectorize(sgames)(d["season"].to_numpy())
    out["gp_share"] = share
    return out


def train_start(t: int) -> int:
    """First training season: 2022 normally; the 2021-22 holdout can only train on 2020-21."""
    return min(2022, t - 1)


def project(t: int) -> tuple[pd.DataFrame, dict]:
    """Projection for season t using only seasons < t for fitting."""
    train = panel(range(train_start(t), t))
    coef = fit(train)
    return predict(features(t), coef), coef


def cv_report():
    seasons = list(range(2022, 2027))
    P = panel(seasons)
    res = []
    for t in seasons:
        tr, te = P[P["season"] != t], P[P["season"] == t]
        coef = fit(tr)
        pr = predict(te, coef)
        rm = rate_mask(te)
        for k in STATS:
            y = te.loc[rm, f"y_{k}"]
            sst = ((y - y.mean()) ** 2).sum()
            res.append({"season": t, "stat": k,
                        "espn": 1 - ((y - te.loc[rm, f"e_{k}"]) ** 2).sum() / sst,
                        "stack": 1 - ((y - pr.loc[rm, k]) ** 2).sum() / sst})
        gm = gp_mask(te)
        y = te.loc[gm, "y_gp"]
        sst = ((y - y.mean()) ** 2).sum()
        e_cal = te.loc[gm, "e_gp"]
        res.append({"season": t, "stat": "gp",
                    "espn": 1 - ((y - e_cal) ** 2).sum() / sst,
                    "stack": 1 - ((y - pr.loc[gm, "gp"]) ** 2).sum() / sst})
    r = pd.DataFrame(res)
    print("Out-of-sample R^2 (leave-one-season-out), rotation players")
    print(r.pivot_table(index="stat", values=["espn", "stack"], aggfunc="mean").round(3).to_string())
    print("\nstack minus ESPN by season:")
    r["gain"] = r["stack"] - r["espn"]
    print(r.pivot(index="stat", columns="season", values="gain").round(3).to_string())
    print("\nfull-sample coefficients [ESPN, Marcel, Marcel*old, Marcel*young]:")
    coef = fit(P)
    for k in STATS:
        print(f"  {k:4} {np.round(coef[k], 3)}")
    print("  gp  ", np.round(coef["gp"], 3), "[1, espn_share, gp1, has1, gp2, has2, old, young]")


if __name__ == "__main__":
    cv_report()


ERR_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fga", "fta"]


def error_params(seasons) -> dict:
    """Projection-error model from leave-one-season-out residuals within `seasons`.

    Returns per-stat scale (sd of (y - pred)/sqrt(pred)), their correlation matrix, FG%/FT% error sd,
    and the empirical distribution of GP-share residuals (actual - predicted share), so truth draws
    keep the fat left tail from season-ending injuries.
    """
    seasons = list(seasons)
    P = panel(seasons)
    Z, fg, ft, gp, gp_pred = [], [], [], [], []
    for t in seasons:
        tr, te = P[P["season"] != t], P[P["season"] == t]
        if tr.empty:  # single training season (2021-22 holdout): in-sample residuals
            tr = te
        pr = predict(te, fit(tr))
        rm = rate_mask(te)
        Z.append(np.stack([((te.loc[rm, f"y_{k}"] - pr.loc[rm, k]) / np.sqrt(pr.loc[rm, k].clip(0.3))).to_numpy()
                           for k in ERR_STATS], 1))
        m = rm & (te["y_fga"] >= 5)
        fg.append((te.loc[m, "y_fgm"] / te.loc[m, "y_fga"] - pr.loc[m, "fgm"] / pr.loc[m, "fga"]).to_numpy())
        m = rm & (te["y_fta"] >= 2)
        ft.append((te.loc[m, "y_ftm"] / te.loc[m, "y_fta"] - pr.loc[m, "ftm"] / pr.loc[m, "fta"]).to_numpy())
        if t != 2023:
            gm = gp_mask(te)
            gp.append((te.loc[gm, "y_gp"] / sgames(t) - pr.loc[gm, "gp_share"]).to_numpy())
            gp_pred.append(pr.loc[gm, "gp_share"].to_numpy())
    Z = np.concatenate(Z)
    scale = Z.std(0)
    return {"scale": dict(zip(ERR_STATS, scale)), "corr": np.corrcoef(Z.T),
            "fg_sd": float(np.concatenate(fg).std()), "ft_sd": float(np.concatenate(ft).std()),
            "gp_resid": np.concatenate(gp), "gp_pred": np.concatenate(gp_pred),
            "z_rows": Z / scale}  # standardized joint residual vectors, for bootstrap draws


def project_espn(t: int) -> pd.DataFrame:
    """Ablation baseline: ESPN per-game rates as-is, GP share linearly calibrated on prior seasons."""
    d = features(t)
    out = pd.DataFrame({k: d[f"e_{k}"] for k in STATS}, index=d.index)
    tr = panel(range(train_start(t), t))
    g = tr[gp_mask(tr) & (tr["season"] != 2023)]
    X = lambda x: np.c_[np.ones(len(x)), x["e_gp"] / np.vectorize(sgames)(x["season"].to_numpy())]
    b = np.linalg.lstsq(X(g), g["y_gp"] / np.vectorize(sgames)(g["season"].to_numpy()), rcond=None)[0] \
        if len(g) else np.array([0.0, 0.83])
    out["gp_share"] = np.clip(X(d) @ b, 0.02, 1.0)
    out["gp"] = out["gp_share"] * sgames(t)
    return out


def cv_darko_report(seasons=range(2022, 2027)):
    """Registered stage-1 test for V5: LOSO accuracy of pts/ast (R^2) and FG%/FT% (squared error),
    current stack vs DARKO-augmented stack."""
    rows = []
    for flag in (False, True):
        set_darko(flag)
        P = panel(list(seasons))
        for t in seasons:
            tr, te = P[P["season"] != t], P[P["season"] == t]
            coef = fit(tr)
            pr = predict(te, coef)
            rm = rate_mask(te)
            row = {"darko": flag, "season": t}
            for k in ("pts", "ast"):
                y = te.loc[rm, f"y_{k}"]
                row[f"r2_{k}"] = 1 - ((y - pr.loc[rm, k]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            for made, att, mn in (("fgm", "fga", 5), ("ftm", "fta", 2)):
                ok = rm & (te[f"y_{att}"] >= mn)
                y = te.loc[ok, f"y_{made}"] / te.loc[ok, f"y_{att}"]
                p = pr.loc[ok, made] / pr.loc[ok, att]
                row[f"se_{made[:2]}pct"] = float(((y - p) ** 2).mean())
            if flag:
                row["coef_pts_darko"] = float(coef["pts"][-1])
                row["coef_ast_darko"] = float(coef["ast"][-1])
                row["w_fg_stack"], row["w_ft_stack"] = coef.get("w_fgm"), coef.get("w_ftm")
            rows.append(row)
    set_darko(False)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(r.round(5).to_string(index=False))
    pooled = r.groupby("darko")[["r2_pts", "r2_ast", "se_fgpct", "se_ftpct"]].mean()
    print("\npooled (mean over held-out seasons):")
    print(pooled.round(5).to_string())
    better = {"pts": pooled.loc[True, "r2_pts"] > pooled.loc[False, "r2_pts"],
              "ast": pooled.loc[True, "r2_ast"] > pooled.loc[False, "r2_ast"],
              "fg%": pooled.loc[True, "se_fgpct"] < pooled.loc[False, "se_fgpct"],
              "ft%": pooled.loc[True, "se_ftpct"] < pooled.loc[False, "se_ftpct"]}
    print("\nstage 1 (all four must improve):", better, "->", "PASS" if all(better.values()) else "FAIL")
    return all(better.values())


def cv_p1_report(seasons=range(2022, 2027)):
    """Registered P1 stage (a): LOSO R^2 of all 11 per-game stats, shipped history model vs P1."""
    out = {}
    for mode in ("marcel", "split"):
        set_hist_mode(mode)
        P = panel(list(seasons))
        r2 = {k: [] for k in STATS}
        for t in seasons:
            tr, te = P[P["season"] != t], P[P["season"] == t]
            pr = predict(te, fit(tr))
            rm = rate_mask(te)
            for k in STATS:
                y = te.loc[rm, f"y_{k}"]
                r2[k].append(1 - ((y - pr.loc[rm, k]) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out[mode] = {k: float(np.mean(v)) for k, v in r2.items()}
        out[mode + "_by_season"] = r2
    set_hist_mode("marcel")
    t = pd.DataFrame({"shipped": out["marcel"], "P1": out["split"]})
    t["gain"] = t["P1"] - t["shipped"]
    print(t.round(4).to_string())
    mean_gain, worst = t["gain"].mean(), t["gain"].min()
    ok = mean_gain >= 0.003 and worst >= -0.003
    print(f"\nmean gain {mean_gain:+.4f} (need >= +0.003), worst stat {worst:+.4f} (need >= -0.003) -> "
          f"{'PASS stage (a)' if ok else 'FAIL stage (a)'}")
    return ok
