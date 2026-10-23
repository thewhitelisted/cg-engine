"""Candidate L2 (DECISIONS.md 2026-10-21): near-term availability + daily lineup cap in the live matchup model.

    python3 research_live2.py      # fits p11 and kappa_L2 on 2024-25, tests L2 vs L1b on 2025-26 (capped truth)
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar

import model as M
import projections as Pj
import research_availability as RA
import research_lineup as RL

STATK = M.STATK
KAPPA_PRE, KAPPA_L1B = 1.182, 1.087
CAT_STATS = ["tpm", "reb", "ast", "stl", "blk", "tov", "pts"]
SLOTS = RL.SLOTS
MC = 30


def load(season, n_match, seed):
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0].copy()
    g["fgx"] = g.fgm - M.LG_FG * g.fga
    g["ftx"] = g.ftm - M.LG_FT * g.fta
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv")
    team_days = sch.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
    meta = Pj.files()[season + 1 if season == 2026 else season]
    st = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
    pool = []
    for pid in st[(st.n >= 40) & (st.mins >= 24)].index:
        tm = meta["pro_team_id"].get(pid)
        if pd.isna(tm) or int(tm) not in team_days.index:
            continue
        if set(g.loc[g.id == pid, "scoring_period"]).issubset(set(team_days[int(tm)])):
            pool.append((pid, int(tm), str(meta["elig"].get(pid, "UTIL"))))
    ids = np.array([p for p, _, _ in pool])
    g = g[g.id.isin(ids)]
    mu = g.groupby("id")[STATK].mean().loc[ids].to_numpy()
    days = np.arange(g.scoring_period.min(), g.scoring_period.max() + 1)
    day0, D = days[0], len(days)
    X = np.zeros((len(ids), D, len(STATK)))
    played = np.zeros((len(ids), D), bool)
    pidx = {p: i for i, p in enumerate(ids)}
    for r in g.itertuples(index=False):
        X[pidx[r.id], r.scoring_period - day0] = [getattr(r, k) for k in STATK]
        played[pidx[r.id], r.scoring_period - day0] = True
    sched = np.zeros((len(ids), D), bool)
    for i, (_, t, _) in enumerate(pool):
        d = team_days[t] - day0
        sched[i, d[(d >= 0) & (d < D)]] = True
    share = played.sum(1) / np.maximum(sched.sum(1), 1)
    E = np.array([[RL.eligible(e, s) for s in SLOTS] for _, _, e in pool], float)
    # season value (G-score style on per-game means over the pool) to decide who a manager benches
    val = np.zeros(len(ids))
    for s, sign in (("fgx", 1), ("ftx", 1), ("tpm", 1), ("reb", 1), ("ast", 1), ("stl", 1), ("blk", 1), ("tov", -1), ("pts", 1)):
        c = mu[:, M.SK[s]]
        val += sign * (c - c.mean()) / c.std()
    week = days // 7
    weeks = [w for w in np.unique(week) if (week == w).sum() == 7]
    gm = sched.sum(1) / len(np.unique(week)) * share
    return dict(mu=mu, var=M.per_game_var(mu), X=X, played=played, sched=sched, share=share, E=E, val=val,
                week=week, weeks=weeks, gm=gm, n=len(ids), n_match=n_match, seed=seed,
                ids=ids, tm=np.array([t for _, t, _ in pool]), day0=day0)


def starters(d, players):
    """Indices (subset of `players`) who start: best value, 10 slots with eligibility."""
    if len(players) <= 10 and d["E"][players].any(1).all():
        return players
    W = d["E"][players] * (10 + d["val"][players][:, None])
    a, b = linear_sum_assignment(-W)
    return players[a][d["E"][players][a, b] > 0]


def truth_and_known(d, T, cols, upto):
    tot, known = np.zeros(len(STATK)), np.zeros(len(STATK))
    for k, c in enumerate(cols):
        on = T[d["played"][T, c]]
        if len(on):
            s = d["X"][starters(d, on), c].sum(0)
            tot += s
            if k < upto:
                known += s
    return tot, known


def state(d, i, c):
    """Player i before day c: misses in a row at the end (0 = played his team's last game)."""
    past = np.where(d["sched"][i, :c])[0]
    m = 0
    for x in past[::-1]:
        if d["played"][i, x]:
            break
        m += 1
    return m


def p_play(d, i, j, m, rho, S):
    pi = min(max(d["share"][i], 1e-3), 0.999)
    if m == 0:                                   # L2b: P(miss next | played) = rho (1 - pi)
        return pi + (1 - pi) * (1 - rho) ** j
    mm = min(m, len(S) - 2)
    back = 1 - S[min(mm + j, len(S) - 1)] / max(S[mm], 1e-12)
    return back * pi


def remaining(d, T, cols, upto, rng, model, p11=None, S=None):
    """Mean and variance of the rest of the week for roster T under 'L1b' or 'L2'."""
    mu, var = d["mu"][T], d["var"][T]
    m, v = np.zeros(len(STATK)), np.zeros(len(STATK))
    if model == "L1b":
        n_rem = d["sched"][T][:, cols[upto:]].sum(1)
        a = d["share"][T]
        ea = (n_rem * a)[:, None]
        return (ea * mu).sum(0), (ea * var + (n_rem * a * (1 - a))[:, None] * mu ** 2).sum(0)
    c0 = cols[upto]
    ms = np.array([state(d, i, c0) for i in T])
    jcount = np.zeros(len(T), int)
    for c in cols[upto:]:
        on = np.where(d["sched"][T, c])[0]
        if not len(on):
            continue
        jcount[on] += 1
        p = np.array([p_play(d, T[k], jcount[k], ms[k], p11, S) for k in on])
        if len(on) <= 10:
            es = p
        else:
            draws = rng.random((MC, len(on))) < p
            es = np.zeros(len(on))
            for dr in draws:
                pl = on[dr]
                if len(pl):
                    st = set(starters(d, T[pl]))
                    es += np.array([T[k] in st for k in on])
            es /= MC
        m += (es[:, None] * mu[on]).sum(0)
        v += (es[:, None] * var[on] + (es * (1 - es))[:, None] * mu[on] ** 2).sum(0)
    return m, v


def outcome(ta, tb):
    o = []
    for x, a_ in (("fgx", "fga"), ("ftx", "fta")):
        o.append(float(ta[M.SK[x]] / ta[M.SK[a_]] > tb[M.SK[x]] / tb[M.SK[a_]]))
    for s in CAT_STATS:
        o.append(float(ta[M.SK[s]] < tb[M.SK[s]]) if s == "tov" else float(ta[M.SK[s]] > tb[M.SK[s]]))
    return np.array(o)


def cases(d, models, p11=None, S=None, days=range(0, 7)):
    """Per (match, week, day): component means/variances for each model, plus outcomes."""
    rng = np.random.default_rng(d["seed"])
    rows = []
    for _ in range(d["n_match"]):
        A, B = np.split(rng.choice(d["n"], 26, replace=False), 2)
        for w in d["weeks"]:
            cols = np.where(d["week"] == w)[0]
            ta, _ = truth_and_known(d, A, cols, 0)
            tb, _ = truth_and_known(d, B, cols, 0)
            if ta[M.SK["fga"]] == 0 or tb[M.SK["fga"]] == 0:
                continue
            obs = outcome(ta, tb)
            y = float(obs.sum() > (1 - obs).sum())
            for day in days:
                rec = {"day": day, "y": y, "obs": obs}
                for model in models:
                    if day == 0 and model != "pre":
                        continue
                    if model == "pre":
                        if day != 0:
                            continue
                        prs = []
                        for T in (A, B):
                            g_ = d["gm"][T][:, None]
                            prs.append(((g_ * d["mu"][T]).sum(0), (g_ * d["var"][T] + M.GAMES_VAR_RATIO * g_ * d["mu"][T] ** 2).sum(0)))
                    else:
                        prs = []
                        for T in (A, B):
                            _, kn = truth_and_known(d, T, cols, day)
                            mm, vv = remaining(d, T, cols, day, rng, model, p11, S)
                            prs.append((kn + mm, vv))
                    rec[model] = prs
                rows.append(rec)
    return rows


def score(rows, model, kappa, day_filter=None):
    out = []
    for r in rows:
        if model not in r or (day_filter is not None and r["day"] not in day_filter):
            continue
        (mA, vA), (mB, vB) = r[model]
        pc = M.Evaluator.cat_probs(mA, np.maximum(vA, 1e-9) * kappa**2, mB, np.maximum(vB, 1e-9) * kappa**2)
        out.append({"day": r["day"], "p": float(M.Evaluator.matchup_win(pc)), "y": r["y"],
                    **{f"c{i}": pc[i] for i in range(9)}, **{f"o{i}": r["obs"][i] for i in range(9)}})
    return pd.DataFrame(out)


def score_l3(rows, kc, psi, day_filter=None):
    """L3: category probabilities with kappa_c; matchup by over-dispersed normal on categories won."""
    from scipy.stats import norm
    out = []
    for r in rows:
        if "L2" not in r or (day_filter is not None and r["day"] not in day_filter):
            continue
        (mA, vA), (mB, vB) = r["L2"]
        pc = M.Evaluator.cat_probs(mA, np.maximum(vA, 1e-9) * kc**2, mB, np.maximum(vB, 1e-9) * kc**2)
        mu, var = pc.sum(), (pc * (1 - pc)).sum() * (1 + psi)
        p = float(1 - norm.cdf((4.5 - mu) / np.sqrt(max(var, 1e-9))))
        out.append({"day": r["day"], "p": p, "y": r["y"], **{f"c{i}": pc[i] for i in range(9)}, **{f"o{i}": r["obs"][i] for i in range(9)}})
    return pd.DataFrame(out)


def cat_brier(x):
    return np.mean([((x[f"c{i}"] - x[f"o{i}"]) ** 2).mean() for i in range(9)])


def run_l3():
    M.set_kappa(1.0)
    rho, S = fit_p11(2025)
    fit = load(2025, n_match=150, seed=5)
    frows = cases(fit, ["L2"], rho, S, days=range(1, 7))
    kc = minimize_scalar(lambda k: cat_brier(score_l3(frows, k, 0.0)), bounds=(0.6, 2.0), method="bounded",
                         options={"xatol": 0.005}).x
    psi = minimize_scalar(lambda q: ((lambda x: ((x.p - x.y) ** 2).mean())(score_l3(frows, kc, q))), bounds=(0.0, 3.0),
                          method="bounded", options={"xatol": 0.01}).x
    print(f"L3 fit on 2024-25: rho {rho:.3f}  kappa_c {kc:.3f}  psi {psi:.3f}", flush=True)
    verdicts = []
    for season in (2026, 2024):
        test = load(season, n_match=300, seed=3)
        rows = cases(test, ["pre", "L1b", "L2"], rho, S)
        pre, l1, l3 = score(rows, "pre", KAPPA_PRE), score(rows, "L1b", KAPPA_L1B), score_l3(rows, kc, psi)
        b = lambda x: ((x.p - x.y) ** 2)
        pre_b = b(pre).mean()
        by1, by3 = l1.assign(e=b(l1)).groupby("day").e.mean(), l3.assign(e=b(l3)).groupby("day").e.mean()
        P = np.concatenate([l3[f"c{i}"].to_numpy() for i in range(9)])
        O = np.concatenate([l3[f"o{i}"].to_numpy() for i in range(9)])
        cal = pd.DataFrame({"pred": P, "obs": O, "bin": pd.cut(P, np.linspace(0, 1, 11), include_lowest=True)}).groupby(
            "bin", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
        ok = (b(l3).mean() < b(l1).mean()) and all(by3[dd] < pre_b for dd in by3.index) and \
            ((cal.n < 200) | ((cal.pred - cal.obs).abs() <= 0.04)).all()
        verdicts.append(ok)
        print(f"\n=== test season {season}: pre-week {pre_b:.4f}", flush=True)
        print("L1b by day:", by1.round(4).to_dict())
        print("L3  by day:", by3.round(4).to_dict())
        print(f"pooled days 1-6: L1b {b(l1).mean():.4f}  L3 {b(l3).mean():.4f}   worst calibration gap "
              f"{(cal[cal.n >= 200].pred - cal[cal.n >= 200].obs).abs().max():.3f}")
        print(cal.round(3).T.to_string(), flush=True)
        print("season", season, "PASS" if ok else "FAIL", flush=True)
    print("L3:", "SHIP" if all(verdicts) else "REJECT", flush=True)


def fit_p11(season):
    """L2b: rho = (1 - p11) / (1 - mean share) on the season's rows, plus the absence survival curve."""
    rows = RA.season_rows(season)
    pairs = [b for r in rows for a, b in zip(r["seq"][:-1], r["seq"][1:]) if a]
    p11 = float(np.mean(pairs))
    share = float(np.mean([r["seq"].mean() for r in rows]))
    return (1 - p11) / (1 - share), RA.km_survival(RA.spells(rows))


if __name__ == "__main__":
    import sys
    if "--l3" in sys.argv:
        run_l3()
        sys.exit()
    M.set_kappa(1.0)
    p11, S = fit_p11(2025)                      # holds rho under L2b
    print(f"rho (2024-25) = {p11:.3f}")
    if "--replicate" in sys.argv:               # frozen model, test on 2023-24 (DECISIONS.md L2b replication)
        k, test_season = 1.193, 2024
    else:
        fit = load(2025, n_match=150, seed=5)
        frows = cases(fit, ["L2"], p11, S, days=range(1, 7))
        def fb(kk):
            x = score(frows, "L2", kk)
            return ((x.p - x.y) ** 2).mean()
        k = minimize_scalar(fb, bounds=(0.7, 2.0), method="bounded", options={"xatol": 0.01}).x
        test_season = 2026
    print(f"kappa_L2 = {k:.3f}; test season {test_season}")
    test = load(test_season, n_match=300, seed=3)
    rows = cases(test, ["pre", "L1b", "L2"], p11, S)
    pre, l1, l2 = score(rows, "pre", KAPPA_PRE), score(rows, "L1b", KAPPA_L1B), score(rows, "L2", k)
    b = lambda x: ((x.p - x.y) ** 2)
    pre_b = b(pre).mean()
    by1, by2 = l1.assign(e=b(l1)).groupby("day").e.mean(), l2.assign(e=b(l2)).groupby("day").e.mean()
    print(f"pre-week Brier {pre_b:.4f}")
    print("L1b by day:", by1.round(4).to_dict())
    print("L2  by day:", by2.round(4).to_dict())
    print(f"pooled days 1-6: L1b {b(l1).mean():.4f}  L2 {b(l2).mean():.4f}")
    P = np.concatenate([l2[f"c{i}"].to_numpy() for i in range(9)])
    O = np.concatenate([l2[f"o{i}"].to_numpy() for i in range(9)])
    cal = pd.DataFrame({"pred": P, "obs": O, "bin": pd.cut(P, np.linspace(0, 1, 11), include_lowest=True)}).groupby(
        "bin", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
    print(cal.round(3).T.to_string())
    ok = (b(l2).mean() < b(l1).mean()) and all(by2[dd] < pre_b for dd in by2.index) and \
        ((cal.n < 200) | ((cal.pred - cal.obs).abs() <= 0.04)).all()
    print("SHIP" if ok else "REJECT")
