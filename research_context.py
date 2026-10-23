"""Candidates C1-C3 (DECISIONS.md 2026-10-23): game context in the current-week matchup engine (L3).

    C1  opponent and pace: per-game expectation scaled by the opponent's allowed factor (games before that day)
    C2  shared-game and shared-day shocks: residual correlation between expected starts, within and across teams
    C3  teammate absences: expectation scaled up by the minutes of teammates on a 2+ game absence

    python3 research_context.py          # fits on 2024-25, tests on 2025-26 and 2023-24, prints the verdicts

Harness, truth and the baseline are research_live2.py's (random 13-man rosters of rotation players, season means as
projections, capped truth, L3 near-term availability and daily lineup cap).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import model as M
import projections as Pj
import research_live2 as R2

STATK, SK = M.STATK, M.SK
MULT = ["pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fga", "fta"]
KGRID = [0, 5, 10, 20, 40, 80, 160, 320, 640, np.inf]
ARMS = ["base", "C1", "C2", "C3", "C13"]           # C13: C1 and C3 together (registered combination step)
FIT_SEASON, TESTS = 2025, (2026, 2024)
N_BOOT = 2000


# ---------------------------------------------------------------- season context
def context(season, d):
    """Opponent of every team on every day, per-player pool rows by team, and all-player logs for absences."""
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv")
    D = d["X"].shape[1]
    teams = sorted(sch.pro_team_id.unique())
    tix = {t: k for k, t in enumerate(teams)}
    opp = np.full((len(teams), D), -1)
    for (_, gid), grp in sch.groupby(["scoring_period", "game_id"]):
        if len(grp) != 2:
            continue
        a, b = grp.pro_team_id.to_numpy()
        c = int(grp.scoring_period.iloc[0]) - d["day0"]
        if 0 <= c < D:
            opp[tix[a], c], opp[tix[b], c] = tix[b], tix[a]
    team = np.array([tix[t] for t in d["tm"]])
    return dict(opp=opp, team=team, T=len(teams), tix=tix, D=D)


def allowed(d, cx):
    """Cumulative (before each day) actual and expected totals by opponent: A, E [T, D+1, stat], games n [T, D+1]."""
    T, D = cx["T"], cx["D"]
    A, E, n = np.zeros((T, D + 1, len(STATK))), np.zeros((T, D + 1, len(STATK))), np.zeros((T, D + 1))
    for c in range(D):
        A[:, c + 1], E[:, c + 1], n[:, c + 1] = A[:, c], E[:, c], n[:, c]
        n[:, c + 1] += cx["opp"][:, c] >= 0
        on = np.where(d["played"][:, c])[0]
        o = cx["opp"][cx["team"][on], c]
        ok = o >= 0
        np.add.at(A[:, c + 1], o[ok], d["X"][on[ok], c])
        np.add.at(E[:, c + 1], o[ok], d["mu"][on[ok]])
    return A, E, n


def opp_factor(A, E, n, o, c, K):
    """Multiplicative factors [len(o), stat] for opponents o as of day index c (data before c), plus fgx shift."""
    f = np.ones((len(o), len(STATK)))
    for s in MULT:
        j = SK[s]
        w = 0.0 if np.isinf(K[s]) else np.where(n[o, c] > 0, n[o, c] / np.maximum(n[o, c] + K[s], 1e-9), 0.0)
        ratio = np.where(E[o, c, j] > 0, A[o, c, j] / np.maximum(E[o, c, j], 1e-9), 1.0)
        f[:, j] = 1 + w * (ratio - 1)
    w = 0.0 if np.isinf(K["fgx"]) else np.where(n[o, c] > 0, n[o, c] / np.maximum(n[o, c] + K["fgx"], 1e-9), 0.0)
    shift = w * (A[o, c, SK["fgx"]] - E[o, c, SK["fgx"]]) / np.maximum(A[o, c, SK["fga"]], 1e-9)
    return f, shift


def adjust_c1(mu, var, f, shift):
    m, v = mu * f, var * f
    m[..., SK["fgx"]] = mu[..., SK["fgx"]] * f[..., SK["fga"]] + m[..., SK["fga"]] * shift
    m[..., SK["ftx"]] = mu[..., SK["ftx"]] * f[..., SK["fta"]]
    return m, v


def fit_c1(d, cx):
    """K per stat by per-game squared error, predictions from opponent data before each game."""
    A, E, n = allowed(d, cx)
    rows = [(i, c) for c in range(cx["D"]) for i in np.where(d["played"][:, c])[0] if cx["opp"][cx["team"][i], c] >= 0]
    ii, cc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    oo = cx["opp"][cx["team"][ii], cc]
    K, report = {}, {}
    for s in MULT + ["fgx"]:
        sse = {}
        for k in KGRID:
            KK = {t: (k if t == s else np.inf) for t in MULT + ["fgx"]}
            f = np.ones((len(ii), len(STATK))); sh = np.zeros(len(ii))
            for c in np.unique(cc):          # factors as of each game's day
                sel = cc == c
                f[sel], sh[sel] = opp_factor(A, E, n, oo[sel], c, KK)
            m, _ = adjust_c1(d["mu"][ii], d["var"][ii], f, sh)
            sse[k] = float(((d["X"][ii, cc, SK[s]] - m[:, SK[s]]) ** 2).sum())
        K[s] = min(sse, key=sse.get)
        report[s] = (K[s], 1 - sse[K[s]] / sse[np.inf])
    return K, report


def residual_corr(d, cx):
    """Per-stat correlation of per-game residuals for pairs: same-game opponents, same-game teammates, same day."""
    sd = np.sqrt(np.maximum(d["var"], 1e-9))
    Z = (d["X"] - d["mu"][:, None, :]) / sd[:, None, :]
    acc = {k: np.zeros(len(STATK)) for k in ("opp", "tm", "day")}
    cnt = {k: 0 for k in ("opp", "tm", "day")}
    zz, nz = np.zeros(len(STATK)), 0
    for c in range(cx["D"]):
        on = np.where(d["played"][:, c])[0]
        if len(on) < 2:
            continue
        z = Z[on, c]
        zz += (z ** 2).sum(0); nz += len(on)
        tot = (z.sum(0) ** 2 - (z ** 2).sum(0)) / 2
        npairs = len(on) * (len(on) - 1) / 2
        tm = cx["team"][on]
        same_t, same_t_n = np.zeros(len(STATK)), 0
        by_team = {}
        for t in np.unique(tm):
            zt = z[tm == t]
            by_team[t] = zt.sum(0)
            same_t += (zt.sum(0) ** 2 - (zt ** 2).sum(0)) / 2
            same_t_n += len(zt) * (len(zt) - 1) / 2
        opp_s, opp_n = np.zeros(len(STATK)), 0
        for t in by_team:
            o = cx["opp"][t, c]
            if o > t and o in by_team:
                opp_s += by_team[t] * by_team[o]
                opp_n += (tm == t).sum() * (tm == o).sum()
        acc["tm"] += same_t; cnt["tm"] += same_t_n
        acc["opp"] += opp_s; cnt["opp"] += opp_n
        acc["day"] += tot - same_t - opp_s; cnt["day"] += npairs - same_t_n - opp_n
    var_z = zz / nz
    return {k: acc[k] / max(cnt[k], 1) / var_z for k in acc}, cnt


def absences(season, d, cx):
    """x[team, c]: minutes of teammates on a 2+ game absence entering day c, / 240 (all players in the logs)."""
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0]
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv")
    team_days = sch.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
    meta = Pj.files()[season + 1 if season == 2026 else season]
    x = np.zeros((cx["T"], cx["D"]))
    st = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
    for pid, r in st[(st.n >= 10) & (st.mins >= 10)].iterrows():
        tm = meta["pro_team_id"].get(pid)
        if pd.isna(tm) or int(tm) not in cx["tix"]:
            continue
        days = set(g.loc[g.id == pid, "scoring_period"])
        tdays = team_days[int(tm)]
        if not days.issubset(set(tdays)):            # traded mid-season: skip (as the harness does)
            continue
        first = min(days)
        t = cx["tix"][int(tm)]
        run, k = 0, 0
        for c in range(cx["D"]):                     # state entering calendar day c
            day = c + d["day0"]
            while k < len(tdays) and tdays[k] < day:
                if tdays[k] >= first:
                    run = 0 if tdays[k] in days else run + 1
                k += 1
            if day > first and run >= 2:
                x[t, c] += r.mins / 240
    return x


def fit_c3(d, cx, x):
    beta, report = {}, {}
    ii, cc = np.where(d["played"])
    xv = x[cx["team"][ii], cc]
    for s in MULT:
        j = SK[s]
        mu, y = d["mu"][ii, j], d["X"][ii, cc, j]
        b = float((xv * mu * (y - mu)).sum() / max((xv ** 2 * mu ** 2).sum(), 1e-9))
        beta[s] = b
        report[s] = (b, 1 - ((y - mu * (1 + b * xv)) ** 2).sum() / ((y - mu) ** 2).sum())
    return beta, report


def adjust_c3(mu, var, xv, beta):
    """mu, var [n, stat]; xv [n]."""
    m, v = mu.copy(), var.copy()
    for s in MULT:
        up = 1 + beta[s] * xv
        m[:, SK[s]] *= up
        v[:, SK[s]] *= up
    m[:, SK["fgx"]] = mu[:, SK["fgx"]] * (1 + beta["fga"] * xv)
    m[:, SK["ftx"]] = mu[:, SK["ftx"]] * (1 + beta["fta"] * xv)
    return m, v


# ---------------------------------------------------------------- matchup rows
def starts(d, T, cols, upto, rng, p11, S):
    """Expected starts es[day k, player] for the remaining days (L3: near-term availability + lineup cap)."""
    c0 = cols[upto]
    ms = np.array([R2.state(d, i, c0) for i in T])
    jcount = np.zeros(len(T), int)
    out = {}
    for c in cols[upto:]:
        on = np.where(d["sched"][T, c])[0]
        es = np.zeros(len(T))
        if len(on):
            jcount[on] += 1
            p = np.array([R2.p_play(d, T[k], jcount[k], ms[k], p11, S) for k in on])
            if len(on) <= 10:
                es[on] = p
            else:
                draws = rng.random((R2.MC, len(on))) < p
                acc = np.zeros(len(on))
                for dr in draws:
                    pl = on[dr]
                    if len(pl):
                        st = set(R2.starters(d, T[pl]))
                        acc += np.array([T[k] in st for k in on])
                es[on] = acc / R2.MC
        out[c] = es
    return out


def components(d, cx, T, es, c0, arm, P):
    """Remaining mean and variance [stat] of roster T for one arm, plus per-day start-weighted sds for C2."""
    m, v = np.zeros(len(STATK)), np.zeros(len(STATK))
    sdw = {}
    for c, e in es.items():
        on = np.where(e > 0)[0]
        if not len(on):
            continue
        mu, var = d["mu"][T[on]], d["var"][T[on]]
        if arm in ("C1", "C13"):
            o = cx["opp"][cx["team"][T[on]], c]
            f, sh = opp_factor(P["A"], P["E"], P["n"], np.maximum(o, 0), c0, P["K"])
            f[o < 0], sh[o < 0] = 1.0, 0.0
            mu, var = adjust_c1(mu, var, f, sh)
        if arm in ("C3", "C13"):
            mu, var = adjust_c3(mu, var, P["x"][cx["team"][T[on]], c0], P["beta"])
        ee = e[on][:, None]
        m += (ee * mu).sum(0)
        v += (ee * var + (ee * (1 - ee)) * mu ** 2).sum(0)
        sdw[c] = (T[on], ee * np.sqrt(np.maximum(var, 1e-9)))
    return m, v, sdw


def pair_cov(cx, a, b, c, rho, same):
    """Sum over pairs of rho(type) * w_i * w_j for one day; `same` = both from one roster (pairs i < j)."""
    (Ta, wa), (Tb, wb) = a, b
    ta, tb = cx["team"][Ta], cx["team"][Tb]
    ob = cx["opp"][ta, c]
    typ_tm = ta[:, None] == tb[None, :]
    typ_opp = ob[:, None] == tb[None, :]
    R = np.where(typ_tm[..., None], rho["tm"], np.where(typ_opp[..., None], rho["opp"], rho["day"]))
    prod = wa[:, None, :] * wb[None, :, :] * R
    if same:
        iu = np.triu_indices(len(Ta), 1)
        return 2 * prod[iu].sum(0)
    return prod.sum((0, 1))


def cases(d, cx, P, p11, S, days):
    rng = np.random.default_rng(d["seed"])
    rows = []
    for match in range(d["n_match"]):
        A_, B_ = np.split(rng.choice(d["n"], 26, replace=False), 2)
        for w in d["weeks"]:
            cols = np.where(d["week"] == w)[0]
            ta, _ = R2.truth_and_known(d, A_, cols, 0)
            tb, _ = R2.truth_and_known(d, B_, cols, 0)
            if ta[SK["fga"]] == 0 or tb[SK["fga"]] == 0:
                continue
            obs = R2.outcome(ta, tb)
            y = float(obs.sum() > (1 - obs).sum())
            for day in days:
                c0 = cols[day]
                _, ka = R2.truth_and_known(d, A_, cols, day)
                _, kb = R2.truth_and_known(d, B_, cols, day)
                esA, esB = starts(d, A_, cols, day, rng, p11, S), starts(d, B_, cols, day, rng, p11, S)
                rec = {"match": match, "day": day, "y": y, "obs": obs}
                for arm in ARMS:
                    mA, vA, sA = components(d, cx, A_, esA, c0, arm, P)
                    mB, vB, sB = components(d, cx, B_, esB, c0, arm, P)
                    cov = np.zeros(len(STATK))
                    if arm == "C2":
                        for c in set(sA) | set(sB):
                            if c in sA:
                                vA = vA + pair_cov(cx, sA[c], sA[c], c, P["rho"], True)
                            if c in sB:
                                vB = vB + pair_cov(cx, sB[c], sB[c], c, P["rho"], True)
                            if c in sA and c in sB:
                                cov = cov + pair_cov(cx, sA[c], sB[c], c, P["rho"], False)
                    rec[arm] = (ka + mA, np.maximum(vA, 1e-9), kb + mB, np.maximum(vB, 1e-9), cov)
                rows.append(rec)
    return rows


# ---------------------------------------------------------------- scoring (L3 with a cross-team covariance)
def stack(rows, arm, days=None):
    keep = [r for r in rows if days is None or r["day"] in days]
    arr = [np.array([r[arm][k] for r in keep]) for k in range(5)]
    meta = pd.DataFrame({"match": [r["match"] for r in keep], "day": [r["day"] for r in keep], "y": [r["y"] for r in keep]})
    return arr, meta, np.array([r["obs"] for r in keep])


def cat_probs_cov(mA, vA, mB, vB, cov):
    """[N, stat] -> [N, 9]; with cov = 0 identical to model.Evaluator.cat_probs."""
    out = []
    for x, a in (("fgx", "fga"), ("ftx", "fta")):
        fa, fb = mA[:, SK[a]], mB[:, SK[a]]
        var = vA[:, SK[x]] / fa ** 2 + vB[:, SK[x]] / fb ** 2 - 2 * cov[:, SK[x]] / (fa * fb)
        out.append(norm.cdf((mA[:, SK[x]] / fa - mB[:, SK[x]] / fb) / np.sqrt(np.maximum(var, 1e-12))))
    for s_ in ("tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        var = vA[:, SK[s_]] + vB[:, SK[s_]] - 2 * cov[:, SK[s_]]
        p = norm.cdf((mA[:, SK[s_]] - mB[:, SK[s_]]) / np.sqrt(np.maximum(var, 1e-12)))
        out.append(1 - p if s_ == "tov" else p)
    return np.stack(out, axis=1)


def score(st, kc, psi):
    (mA, vA, mB, vB, cov), meta, obs = st
    pc = cat_probs_cov(mA, vA * kc ** 2, mB, vB * kc ** 2, cov * kc ** 2)
    mu, var = pc.sum(1), (pc * (1 - pc)).sum(1) * (1 + psi)
    p = 1 - norm.cdf((4.5 - mu) / np.sqrt(np.maximum(var, 1e-9)))
    out = meta.assign(p=p)
    for i_ in range(9):
        out[f"c{i_}"], out[f"o{i_}"] = pc[:, i_], obs[:, i_]
    return out


def fit_l3(rows, arm):
    st = stack(rows, arm)
    kc = minimize_scalar(lambda k: R2.cat_brier(score(st, k, 0.0)), bounds=(0.6, 2.0), method="bounded",
                         options={"xatol": 0.005}).x
    psi = minimize_scalar(lambda q: (lambda x: ((x.p - x.y) ** 2).mean())(score(st, kc, q)), bounds=(0.0, 3.0),
                          method="bounded", options={"xatol": 0.01}).x
    return kc, psi


def calibration(x):
    P = np.concatenate([x[f"c{i}"].to_numpy() for i in range(9)])
    O = np.concatenate([x[f"o{i}"].to_numpy() for i in range(9)])
    return pd.DataFrame({"pred": P, "obs": O, "bin": pd.cut(P, np.linspace(0, 1, 11), include_lowest=True)}).groupby(
        "bin", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))


def params(season, d, cx, fitted=None):
    """Season-specific inputs (opponent totals, absences) plus fitted parameters (from 2024-25)."""
    A, E, n = allowed(d, cx)
    P = {"A": A, "E": E, "n": n, "x": absences(season, d, cx)}
    if fitted:
        P.update(fitted)
    return P


def main():
    M.set_kappa(1.0)
    p11, S = R2.fit_p11(FIT_SEASON)
    fit = R2.load(FIT_SEASON, n_match=150, seed=5)
    cx = context(FIT_SEASON, fit)
    K, rep1 = fit_c1(fit, cx)
    rho, npairs = residual_corr(fit, cx)
    P = params(FIT_SEASON, fit, cx)
    beta, rep3 = fit_c3(fit, cx, P["x"])
    fitted = {"K": K, "rho": rho, "beta": beta}
    P.update(fitted)
    print("C1 K by stat and first-stage SSE reduction:", {s: (k, round(r, 4)) for s, (k, r) in rep1.items()}, flush=True)
    print("C2 residual correlations (pairs):", {k: dict(zip(STATK, np.round(v, 4))) for k, v in rho.items()}, npairs, flush=True)
    print("C3 beta and first-stage SSE reduction:", {s: (round(b, 3), round(r, 4)) for s, (b, r) in rep3.items()}, flush=True)
    frows = cases(fit, cx, P, p11, S, days=range(1, 7))
    L3 = {arm: fit_l3(frows, arm) for arm in ARMS}
    print("refit kappa_c, psi on 2024-25:", {a: (round(k, 3), round(q, 3)) for a, (k, q) in L3.items()}, flush=True)

    passed = {}
    for season in TESTS:
        test = R2.load(season, n_match=300, seed=3)
        tcx = context(season, test)
        TP = params(season, test, tcx, fitted)
        rows = cases(test, tcx, TP, p11, S, days=range(0, 7))
        sc = {arm: score(stack(rows, arm), *L3[arm]) for arm in ARMS}
        err = {arm: (x.p - x.y) ** 2 for arm, x in sc.items()}
        base_day = sc["base"].assign(e=err["base"]).groupby("day").e.mean()
        print(f"\n=== test season {season}: baseline Brier pooled {err['base'].mean():.4f}; by day {base_day.round(4).to_dict()}", flush=True)
        rng = np.random.default_rng(11)
        matches = sc["base"].match.to_numpy()
        um = np.unique(matches)
        for arm in ARMS[1:]:
            diff = (err[arm] - err["base"]).to_numpy()
            per = pd.DataFrame({"m": matches, "d": diff}).groupby("m").d.agg(["sum", "size"]).loc[um]
            boots = []
            for _ in range(N_BOOT):
                pick = rng.integers(0, len(um), len(um))
                boots.append(per["sum"].to_numpy()[pick].sum() / per["size"].to_numpy()[pick].sum())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            by_day = sc[arm].assign(e=err[arm]).groupby("day").e.mean()
            worst_day = float((by_day - base_day).max())
            cal = calibration(sc[arm])
            gap = float((cal[cal.n >= 200].pred - cal[cal.n >= 200].obs).abs().max())
            ok = diff.mean() < 0 and hi < 0 and worst_day <= 0.002 and gap <= 0.04
            passed.setdefault(arm, []).append(ok)
            print(f"{arm}: pooled {err[arm].mean():.4f} (diff {diff.mean():+.5f}, 95% CI {lo:+.5f} to {hi:+.5f}); "
                  f"worst day vs base {worst_day:+.4f}; worst calibration gap {gap:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
            print(f"   by day {by_day.round(4).to_dict()}", flush=True)
    for arm in ARMS[1:]:
        print(f"{arm}: {'SHIP' if all(passed[arm]) else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
