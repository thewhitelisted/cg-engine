"""Round R (DECISIONS.md 2026-10-23): five registered candidates. Nothing here changes production.

    python3 research_round_r.py r1      # early-season projection factor
    python3 research_round_r.py r5      # trades: how much pre-trade games should count
    python3 research_round_r.py r2      # back-to-backs (availability, production) in the matchup harness
    python3 research_round_r.py r3      # opponent factors for future weeks
    python3 research_round_r.py r4      # fantasy-playoff rest rule

Shared: fit 2024-25, primary tests 2025-26 and 2023-24, secondary 2021-22; 99% cluster-bootstrap intervals.
Every game is predicted from information before its day; projections are forward-chained (seasons < t only).
"""
import json
import sys

import numpy as np
import pandas as pd

import model as M

FIT, PRIMARY, SECONDARY = 2025, (2026, 2024), (2022,)
BLEND = ["pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fga", "fta"]
VALUE = ["fgx", "ftx", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"]
K_GAMES = json.loads((M.DATA / "inseason_K.json").read_text())
N_BOOT, LEVEL = 2000, 0.99
TAUS = [1, 2, 3, 4, 6, 8, 12]


# ---------------------------------------------------------------- shared helpers
def season_frame(t):
    """Forward-chained projections, value-scale spreads, and game logs with dates and season week."""
    s = M.get_season(t)
    df = s.df.set_index("id")
    _, scale = M.static_value(s.df)
    sd = {M.STATK[j]: sdv for _, (j, _, sdv, _) in scale.items()}
    sch = pd.read_csv(M.DATA / f"schedule_{t}.csv", parse_dates=["date"])
    date_of = sch.groupby("scoring_period").date.first()
    g = pd.read_csv(M.DATA / f"gamelogs_{t}.csv")
    g = g[(g["min"] > 0) & g.id.isin(df.index)].copy()
    g["date"] = g.scoring_period.map(date_of)
    g["week"] = (g.date - sch.date.min()).dt.days // 7 + 1
    g = g.sort_values(["id", "date"])
    g["fgx"] = g.fgm - M.LG_FG * g.fga
    g["ftx"] = g.ftm - M.LG_FT * g.fta
    return df, sd, g, sch


def blend(proj, sums, n, c=None):
    """Production in-season blend (waiver.updated_rates) for arrays of player-games.
    proj: DataFrame rows aligned to the games; sums: dict of season sums before the game; n: games before;
    c: optional dict stat -> multiplier on the projection (counting stats and attempts)."""
    out = {}
    for k in BLEND:
        p = proj[k].to_numpy() * (c[k] if c is not None else 1.0)
        out[k] = (K_GAMES[k] * p + sums[k]) / (K_GAMES[k] + n)
    for x, a, made, lg in (("fgx", "fga", "fgm", M.LG_FG), ("ftx", "fta", "ftm", M.LG_FT)):
        pa = proj[a].to_numpy()
        ppct = proj[made].to_numpy() / np.maximum(pa, 0.1)
        prior = K_GAMES[a] * pa
        pct = (prior * ppct + sums[made]) / np.maximum(prior + sums[a], 1e-6)
        out[x] = out[a] * (pct - lg)
    return out


def se(g, pred, sd):
    return sum(((g[k].to_numpy() - pred[k]) / sd[k]) ** 2 for k in VALUE)


def cum_before(g, cols, by="id"):
    """Season sums of `cols` and games count before each row (rows sorted by player and date)."""
    sums = {k: (g.groupby(by)[k].cumsum() - g[k]).to_numpy() for k in cols}
    n = g.groupby(by).cumcount().to_numpy().astype(float)
    return sums, n


def boot_diff(diff, cluster, seed=7):
    """Mean of per-game differences with a cluster-bootstrap interval at LEVEL."""
    d = pd.DataFrame({"c": cluster, "d": diff}).groupby("c").d.agg(["sum", "size"])
    s, n = d["sum"].to_numpy(), d["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    b = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(b, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
    return float(diff.mean()), float(lo), float(hi)


def verdict(per_season, pooled):
    ok = all(m < 0 for m in per_season.values()) and pooled[2] < 0
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- R1 early season
def r1_rows(t):
    df, sd, g, _ = season_frame(t)
    gp = g.groupby("id").size()
    keep = df.index[(df.espn_min >= 20) & df.index.isin(gp[gp >= 20].index)]
    g = g[g.id.isin(keep)].reset_index(drop=True)
    sums, n = cum_before(g, BLEND + ["fgm", "ftm"])
    proj = df.loc[g.id]
    return g, proj, sums, n, sd


def r1_predict(g, proj, sums, n, delta=None, tau=None):
    if delta is None:
        return blend(proj, sums, n)
    e = np.exp(-(g.week.to_numpy() - 1) / tau)
    return blend(proj, sums, n, {k: 1 + delta[k] * e for k in BLEND})


def r1_fit(g, proj, sums, n, tau):
    """delta_s by least squares on each blended stat's own error (prediction is linear in delta_s)."""
    e = np.exp(-(g.week.to_numpy() - 1) / tau)
    base = blend(proj, sums, n)
    delta = {}
    for k in BLEND:
        a = K_GAMES[k] * proj[k].to_numpy() * e / (K_GAMES[k] + n)
        delta[k] = float((a * (g[k].to_numpy() - base[k])).sum() / (a ** 2).sum())
    return delta


def run_r1():
    fit = r1_rows(FIT)
    best = None
    for tau in TAUS:
        d = r1_fit(*fit[:4], tau)
        err = se(fit[0], r1_predict(*fit[:4], d, tau), fit[4]).mean()
        print(f"  fit tau {tau:>2}: SE {err:.4f}", flush=True)
        if best is None or err < best[0]:
            best = (err, tau, d)
    _, tau, delta = best
    print(f"R1 fit on 2024-25: tau {tau} weeks, delta {json.dumps({k: round(v, 3) for k, v in delta.items()})}", flush=True)
    per, pooled_d, pooled_c = {}, [], []
    for t in PRIMARY + SECONDARY:
        g, proj, sums, n, sd = r1_rows(t)
        b, c = se(g, r1_predict(g, proj, sums, n), sd), se(g, r1_predict(g, proj, sums, n, delta, tau), sd)
        diff = c - b
        m, lo, hi = boot_diff(diff, g.id.to_numpy())
        early = g.week.to_numpy() <= 6
        me, loe, hie = boot_diff(diff[early], g.id.to_numpy()[early])
        print(f"{t}: games {len(g)} players {g.id.nunique()} | SE base {b.mean():.4f} cand {c.mean():.4f} diff {m:+.5f} "
              f"[{lo:+.5f}, {hi:+.5f}] | weeks 1-6 diff {me:+.5f} [{loe:+.5f}, {hie:+.5f}]", flush=True)
        if t in PRIMARY:
            per[t] = m
            pooled_d.append(diff); pooled_c.append(np.char.add(str(t) + "_", g.id.astype(str).to_numpy()))
        gm = g.assign(pmin=proj["espn_min"].to_numpy())
        mins = gm.groupby("week")["min"].mean() / gm.groupby("week").pmin.mean()
        print(f"   minutes actual / projected by week 1-8: {mins.loc[1:8].round(3).to_dict()}", flush=True)
    pooled = boot_diff(np.concatenate(pooled_d), np.concatenate(pooled_c))
    print(f"R1 pooled primary diff {pooled[0]:+.5f} [{pooled[1]:+.5f}, {pooled[2]:+.5f}] -> {verdict(per, pooled)}", flush=True)
    return {"tau": tau, "delta": delta}


# ---------------------------------------------------------------- R5 trades
def split_team(days, sched):
    """(old team, new team, index of first post-trade game) if the player's games split cleanly across two teams."""
    full = [t for t, s in sched.items() if all(d in s for d in days)]
    if full:
        return None
    pre = {t: next((i for i, d in enumerate(days) if d not in s), len(days)) for t, s in sched.items()}
    suf = {t: next((i for i, d in enumerate(reversed(days)) if d not in s), len(days)) for t, s in sched.items()}
    A = max(pre, key=pre.get)
    B = max((t for t in suf if t != A), key=suf.get)
    if pre[A] + suf[B] < len(days):
        return None                                   # needs more than two teams, or data gaps
    return A, B, pre[A]                               # ambiguous shared days go to the old team


def r5_rows(t):
    df, sd, g, sch = season_frame(t)
    sched = {tm: set(x) for tm, x in sch.groupby("pro_team_id").scoring_period}
    rows = []
    for pid, gp in g.groupby("id"):
        sp = gp.scoring_period.tolist()
        sp_ = split_team(sp, sched)
        if sp_ is None or gp["min"].mean() < 15:
            continue
        _, _, k = sp_
        if k < 5 or len(sp) - k < 5:
            continue
        rows.append(gp.assign(post=np.arange(len(gp)) >= k))
    if not rows:
        return None
    g = pd.concat(rows).reset_index(drop=True)
    pre = g.assign(**{c: g[c].where(~g.post, 0) for c in BLEND + ["fgm", "ftm"]})
    post = g.assign(**{c: g[c].where(g.post, 0) for c in BLEND + ["fgm", "ftm"]})
    s_pre, _ = cum_before(pre, BLEND + ["fgm", "ftm"])
    s_post, _ = cum_before(post, BLEND + ["fgm", "ftm"])
    n_pre = (g.groupby("id").cumcount() - g.groupby("id").post.cumsum() + g.post).to_numpy().astype(float)
    n_post = (g.groupby("id").post.cumsum() - g.post).to_numpy().astype(float)
    proj = df.loc[g.id]
    return g, proj, s_pre, s_post, n_pre, n_post, sd


def r5_predict(rows, w):
    g, proj, s_pre, s_post, n_pre, n_post, _ = rows
    sums = {k: w * s_pre[k] + s_post[k] for k in s_pre}
    return blend(proj, sums, w * n_pre + n_post)


def r5_eval(rows, w):
    g, sd = rows[0], rows[-1]
    m = g.post.to_numpy()
    return se(g[m], {k: v[m] for k, v in r5_predict(rows, w).items()}, sd), g.id.to_numpy()[m]


def run_r5():
    fit = r5_rows(FIT)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    errs = {w: r5_eval(fit, w)[0].mean() for w in grid}
    w_star = float(min(errs, key=errs.get))
    print(f"R5 fit on 2024-25: players {fit[0].id.nunique()}, post-trade games {int(fit[0].post.sum())}; "
          f"best w {w_star} (SE {errs[w_star]:.4f} vs all-games {errs[1.0]:.4f}, post-only {errs[0.0]:.4f})", flush=True)
    res = {}
    for name, w in (("R5-post", 0.0), ("R5-w", w_star)):
        per, pd_, pc_ = {}, [], []
        for t in PRIMARY + SECONDARY:
            rows = r5_rows(t)
            base, cl = r5_eval(rows, 1.0)
            cand, _ = r5_eval(rows, w)
            diff = cand - base
            m, lo, hi = boot_diff(diff, cl)
            print(f"{name} {t}: players {rows[0].id.nunique()}, post games {len(diff)} | SE base {base.mean():.4f} "
                  f"cand {cand.mean():.4f} diff {m:+.5f} [{lo:+.5f}, {hi:+.5f}]", flush=True)
            if t in PRIMARY:
                per[t] = m; pd_.append(diff); pc_.append(np.char.add(str(t) + "_", cl.astype(str)))
        pooled = boot_diff(np.concatenate(pd_), np.concatenate(pc_))
        res[name] = verdict(per, pooled)
        print(f"{name} pooled primary diff {pooled[0]:+.5f} [{pooled[1]:+.5f}, {pooled[2]:+.5f}] -> {res[name]}", flush=True)
    return {"w": w_star, **res}


# ---------------------------------------------------------------- harness (R2, R3)
def harness(season, n_match, seed):
    import research_context as RC
    import research_live2 as R2
    import projections as Pj
    d = R2.load(season, n_match=n_match, seed=seed)
    cx = RC.context(season, d)
    D = cx["D"]
    played_team = cx["opp"] >= 0                                         # team plays on day c
    b2b = np.zeros_like(played_team)
    b2b[:, 1:] = played_team[:, 1:] & played_team[:, :-1]               # second night of a back-to-back
    pmin = Pj.espn_proj(season)["min"].reindex(d["ids"]).fillna(0).to_numpy()
    tier = np.where(pmin >= 30, 0, 1)
    team = cx["team"]
    bi = b2b[team]                                                       # [player, day]
    f = (bi & d["sched"]).sum(1) / np.maximum(d["sched"].sum(1), 1)
    return d, cx, bi, f, tier


def fit_r2(d, bi, tier):
    sched, played = d["sched"], d["played"]
    r = []
    for t in (0, 1):
        m = tier == t
        b2, all_ = (bi & sched)[m], sched[m]
        r.append(float(((played[m] & b2).sum() / b2.sum()) / ((played[m] & all_).sum() / all_.sum())))
    msk = played & bi
    mm = {}
    for k in ("pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fga", "fta"):
        j = M.SK[k]
        mm[k] = float(d["X"][..., j][msk].sum() / np.broadcast_to(d["mu"][:, None, j], msk.shape)[msk].sum())
    return r, mm


def r2_cases(d, cx, bi, f, tier, P, p11, S, days, arms):
    """Matchup rows for each arm with common random numbers (same rosters, same lineup draws)."""
    import research_context as RC
    import research_live2 as R2
    rng = np.random.default_rng(d["seed"])
    rows = []
    for match in range(d["n_match"]):
        A_, B_ = np.split(rng.choice(d["n"], 26, replace=False), 2)
        for w in d["weeks"]:
            cols = np.where(d["week"] == w)[0]
            ta, _ = R2.truth_and_known(d, A_, cols, 0)
            tb, _ = R2.truth_and_known(d, B_, cols, 0)
            if ta[M.SK["fga"]] == 0 or tb[M.SK["fga"]] == 0:
                continue
            obs = R2.outcome(ta, tb)
            y = float(obs.sum() > (1 - obs).sum())
            for day in days:
                c0 = cols[day]
                rec = {"match": match, "day": day, "y": y, "obs": obs}
                sub = int(rng.integers(1 << 30))
                for arm in arms:
                    out = []
                    for T, k_ in ((A_, 0), (B_, 1)):
                        _, kn = R2.truth_and_known(d, T, cols, day)
                        es = r2_starts(d, T, cols, day, np.random.default_rng(sub + k_), p11, S,
                                       bi, f, tier, P["r"] if arm in ("R2a", "R2ab") else None)
                        m, v = r2_components(d, T, es, bi, f, P["m"] if arm in ("R2b", "R2ab") else None)
                        out += [kn + m, np.maximum(v, 1e-9)]
                    rec[arm] = (out[0], out[1], out[2], out[3], np.zeros(len(M.STATK)))
                rows.append(rec)
    return rows


def r2_starts(d, T, cols, upto, rng, p11, S, bi, f, tier, r):
    import research_live2 as R2
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
            if r is not None:
                rt = np.array([r[tier[T[k]]] for k in on])
                fk = f[T[on]]
                adj = np.where(bi[T[on], c], rt, (1 - fk * rt) / np.maximum(1 - fk, 1e-9))
                p = np.clip(p * adj, 0, 1)
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


def r2_components(d, T, es, bi, f, mm):
    m, v = np.zeros(len(M.STATK)), np.zeros(len(M.STATK))
    for c, e in es.items():
        on = np.where(e > 0)[0]
        if not len(on):
            continue
        mu, var = d["mu"][T[on]].copy(), d["var"][T[on]].copy()
        if mm is not None:
            fk, is_b = f[T[on]], bi[T[on], c]
            base = mu.copy()
            for k, mk in mm.items():
                adj = np.where(is_b, mk, (1 - fk * mk) / np.maximum(1 - fk, 1e-9))
                mu[:, M.SK[k]] = base[:, M.SK[k]] * adj
                var[:, M.SK[k]] *= adj
            adjg = np.where(is_b, mm["fga"], (1 - fk * mm["fga"]) / np.maximum(1 - fk, 1e-9))
            adjt = np.where(is_b, mm["fta"], (1 - fk * mm["fta"]) / np.maximum(1 - fk, 1e-9))
            mu[:, M.SK["fgx"]] = base[:, M.SK["fgx"]] * adjg
            mu[:, M.SK["ftx"]] = base[:, M.SK["ftx"]] * adjt
        ee = e[on][:, None]
        m += (ee * mu).sum(0)
        v += (ee * var + (ee * (1 - ee)) * mu ** 2).sum(0)
    return m, v


def judge(rows_by_season, arms, L3):
    """Per-season and pooled-primary Brier differences vs base, cluster = match, plus calibration."""
    import research_context as RC
    out = {}
    for arm in arms:
        per, dd, cl = {}, [], []
        for t, rows in rows_by_season.items():
            sb = RC.score(RC.stack(rows, "base"), *L3["base"])
            sc = RC.score(RC.stack(rows, arm), *L3[arm])
            diff = ((sc.p - sc.y) ** 2 - (sb.p - sb.y) ** 2).to_numpy()
            m, lo, hi = boot_diff(diff, sc.match.to_numpy())
            cal = RC.calibration(sc)
            gap = float((cal[cal.n >= 200].pred - cal[cal.n >= 200].obs).abs().max())
            byday = (sc.assign(e=(sc.p - sc.y) ** 2).groupby("day").e.mean() - sb.assign(e=(sb.p - sb.y) ** 2).groupby("day").e.mean())
            print(f"{arm} {t}: base {((sb.p - sb.y) ** 2).mean():.4f} cand {((sc.p - sc.y) ** 2).mean():.4f} diff {m:+.5f} "
                  f"[{lo:+.5f}, {hi:+.5f}] worst calibration {gap:.3f} worst day {byday.max():+.4f}", flush=True)
            if t in PRIMARY:
                per[t] = m if gap <= 0.04 else 1.0
                dd.append(diff); cl.append(np.char.add(str(t) + "_", sc.match.astype(str).to_numpy()))
        pooled = boot_diff(np.concatenate(dd), np.concatenate(cl))
        out[arm] = verdict(per, pooled)
        print(f"{arm} pooled primary diff {pooled[0]:+.5f} [{pooled[1]:+.5f}, {pooled[2]:+.5f}] -> {out[arm]}", flush=True)
    return out


def run_r2(arms=("R2a", "R2b")):
    import research_context as RC
    import research_live2 as R2
    M.set_kappa(1.0)
    p11, S = R2.fit_p11(FIT)
    d, cx, bi, f, tier = harness(FIT, 150, 5)
    r, mm = fit_r2(d, bi, tier)
    print(f"R2 fit on 2024-25: play-rate ratio on second nights r = {np.round(r, 3).tolist()} (tiers >=30 / 20-30 projected min); "
          f"production ratio m = {json.dumps({k: round(v, 3) for k, v in mm.items()})}", flush=True)
    P = {"r": r, "m": mm}
    all_arms = ("base",) + tuple(arms)
    frows = r2_cases(d, cx, bi, f, tier, P, p11, S, range(1, 7), all_arms)
    L3 = {a: RC.fit_l3(frows, a) for a in all_arms}
    print("refit kappa_c, psi:", {a: (round(k, 3), round(q, 3)) for a, (k, q) in L3.items()}, flush=True)
    rows = {}
    for t in PRIMARY + SECONDARY:
        dt, cxt, bit, ft, tt = harness(t, 300, 3)
        rows[t] = r2_cases(dt, cxt, bit, ft, tt, P, p11, S, range(0, 7), all_arms)
    return judge(rows, arms, L3), P


# ---------------------------------------------------------------- R3 opponents in future weeks
def r3_cases(d, cx, P, arms, horizons=(1, 2, 3, 4)):
    import research_context as RC
    import research_live2 as R2
    rng = np.random.default_rng(d["seed"])
    rows = []
    wk = list(d["weeks"])
    for match in range(d["n_match"]):
        A_, B_ = np.split(rng.choice(d["n"], 26, replace=False), 2)
        for w in wk:
            c0 = np.where(d["week"] == w)[0][0]                     # prediction made at the start of week w
            for h in horizons:
                if w + h not in wk:
                    continue
                cols = np.where(d["week"] == w + h)[0]
                ta, _ = R2.truth_and_known(d, A_, cols, 0)
                tb, _ = R2.truth_and_known(d, B_, cols, 0)
                if ta[M.SK["fga"]] == 0 or tb[M.SK["fga"]] == 0:
                    continue
                obs = R2.outcome(ta, tb)
                rec = {"match": match, "day": h, "y": float(obs.sum() > (1 - obs).sum()), "obs": obs}
                for arm in arms:
                    out = []
                    for T in (A_, B_):
                        m, v = np.zeros(len(M.STATK)), np.zeros(len(M.STATK))
                        for c in cols:
                            on = np.where(d["sched"][T, c])[0]
                            if not len(on):
                                continue
                            mu, var = d["mu"][T[on]], d["var"][T[on]]
                            if arm == "R3":
                                o = cx["opp"][cx["team"][T[on]], c]
                                fct, sh = RC.opp_factor(P["A"], P["E"], P["n"], np.maximum(o, 0), c0, P["K"])
                                fct[o < 0], sh[o < 0] = 1.0, 0.0
                                mu, var = RC.adjust_c1(mu, var, fct, sh)
                            a = d["share"][T[on]][:, None]
                            m += (a * mu).sum(0)
                            v += (a * var + a * (1 - a) * mu ** 2).sum(0)
                        out += [m, np.maximum(v, 1e-9)]
                    rec[arm] = (out[0], out[1], out[2], out[3], np.zeros(len(M.STATK)))
                rows.append(rec)
    return rows


def run_r3():
    import research_context as RC
    import research_live2 as R2
    M.set_kappa(1.0)
    fit = R2.load(FIT, n_match=150, seed=5)
    cx = RC.context(FIT, fit)
    K, _ = RC.fit_c1(fit, cx)
    A, E, n = RC.allowed(fit, cx)
    arms = ("base", "R3")
    frows = r3_cases(fit, cx, {"A": A, "E": E, "n": n, "K": K}, arms)
    L3 = {a: RC.fit_l3(frows, a) for a in arms}
    print(f"R3: K {K}; refit kappa_c, psi {L3}", flush=True)
    rows = {}
    for t in PRIMARY + SECONDARY:
        d = R2.load(t, n_match=300, seed=3)
        cxt = RC.context(t, d)
        A, E, n = RC.allowed(d, cxt)
        rows[t] = r3_cases(d, cxt, {"A": A, "E": E, "n": n, "K": K}, arms)
        sc, sb = RC.score(RC.stack(rows[t], "R3"), *L3["R3"]), RC.score(RC.stack(rows[t], "base"), *L3["base"])
        print(f"   {t} by horizon (cand - base):", (sc.assign(e=(sc.p - sc.y) ** 2).groupby("day").e.mean()
              - sb.assign(e=(sb.p - sb.y) ** 2).groupby("day").e.mean()).round(5).to_dict(), flush=True)
    return judge(rows, ("R3",), L3)


# ---------------------------------------------------------------- R4 fantasy-playoff rest
WINDOW = {2022: "2022-03-07", 2024: "2024-03-11", 2025: "2025-03-10", 2026: "2026-03-09"}
R4_SEASONS = (2022, 2024, 2025, 2026)
R4_PRIMARY = (2026, 2025, 2024)
K_GP = 34


def team_results(season, teams=None):
    """Each team's regular-season results (date, win) from ESPN's public team schedules; cached."""
    import requests
    f = M.DATA / f"team_results_{season}.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    rows = []
    for tm in (teams if teams is not None else pd.read_csv(M.DATA / f"schedule_{season}.csv").pro_team_id.unique()):
        j = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{int(tm)}/schedule",
                         params={"season": season}, timeout=30).json()
        for e in j.get("events", []):
            c = e["competitions"][0]
            me = next((x for x in c["competitors"] if int(x["team"]["id"]) == int(tm)), None)
            if me is not None and me.get("winner") is not None:
                rows.append({"pro_team_id": int(tm), "date": e["date"][:10], "win": bool(me["winner"])})
    out = pd.DataFrame(rows)
    out.to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["date"])


def r4_rows(season):
    s = M.get_season(season)
    pre_share = s.df.set_index("id")["avail"]
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv", parse_dates=["date"])
    date_of = sch.groupby("scoring_period").date.first()
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0].copy()
    g["date"] = g.scoring_period.map(date_of)
    meta = __import__("projections").files()[season + 1 if season == 2026 else season]
    lo = pd.Timestamp(WINDOW[season]); hi = lo + pd.Timedelta(days=20)
    cut = pd.Timestamp(f"{season}-02-10")
    rec = pd.read_csv(M.DATA / "team_records.csv")
    wp_prev = rec[rec.season == season - 1].set_index("pro_team_id").win_pct
    res = team_results(season)
    wp_now = res[res.date < cut].groupby("pro_team_id").win.mean()
    tdays = {t: set(x) for t, x in sch.groupby("pro_team_id").scoring_period}
    out = []
    for pid, gp in g.groupby("id"):
        tm = meta["pro_team_id"].get(pid)
        if pd.isna(tm) or int(tm) not in tdays or gp["min"].mean() < 25 or not set(gp.scoring_period) <= tdays[int(tm)]:
            continue
        tm = int(tm)
        ts = sch[sch.pro_team_id == tm]
        before, win = ts[ts.date < cut], ts[(ts.date >= lo) & (ts.date <= hi)]
        outside = ts[(ts.date < lo) | (ts.date > hi)]
        pb = gp[gp.date < cut]
        if len(pb) < 20 or len(win) == 0 or pid not in pre_share.index:
            continue
        share_now = (K_GP * pre_share[pid] + len(pb)) / (K_GP + len(before))
        out.append({"season": season, "id": pid, "team": tm, "pred0": min(share_now, 1.0),
                    "y": gp.date.between(lo, hi).sum() / len(win), "w": len(win),
                    "part_win": gp.date.between(lo, hi).sum() / len(win),
                    "part_out": (~gp.date.between(lo, hi)).sum() / len(outside),
                    "wp_prev": wp_prev.get(tm), "wp_now": wp_now.get(tm)})
    return pd.DataFrame(out)


def run_r4():
    D = pd.concat([r4_rows(t) for t in R4_SEASONS]).dropna(subset=["wp_prev", "wp_now"])
    print("R4 players by season:", D.groupby("season").size().to_dict(), flush=True)
    per = {"R4-1": {}, "R4-2": {}}
    pooled = {"R4-1": ([], []), "R4-2": ([], []), "R4-2v1": ([], [])}
    for t in R4_SEASONS:
        tr, te = D[D.season != t], D[D.season == t].copy()
        base_part = tr.part_out.mean()
        err = {}
        for name, col in (("R4-1", "wp_prev"), ("R4-2", "wp_now")):
            X = np.c_[np.ones(len(tr)), tr[col]]
            a, b = np.linalg.lstsq(X, tr.part_win - tr.part_out, rcond=None)[0]
            mult = np.clip(1 + (a + b * te[col]) / base_part, 0.5, 1.05)
            te[name] = te.pred0 * mult
        for name in ("pred0", "R4-1", "R4-2"):
            err[name] = te.w * (te.y - te[name]) ** 2 / te.w.mean()
        line = f"{t}: n {len(te)} | weighted sq. error R4-0 {err['pred0'].mean():.4f}  R4-1 {err['R4-1'].mean():.4f}  R4-2 {err['R4-2'].mean():.4f}"
        for name, ref in (("R4-1", "pred0"), ("R4-2", "pred0"), ("R4-2v1", "R4-1")):
            cand = "R4-2" if name == "R4-2v1" else name
            diff = (err[cand] - err[ref]).to_numpy()
            m, lo, hi = boot_diff(diff, te.team.to_numpy())
            line += f" | {cand} vs {ref} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            if t in R4_PRIMARY:
                if name in per:
                    per[name][t] = m
                if name == "R4-2v1":
                    per.setdefault("R4-2v1", {})[t] = m
                pooled[name][0].append(diff); pooled[name][1].append(np.char.add(str(t) + "_", te.team.astype(str).to_numpy()))
        print(line, flush=True)
    for name in ("R4-1", "R4-2", "R4-2v1"):
        pl = boot_diff(np.concatenate(pooled[name][0]), np.concatenate(pooled[name][1]))
        print(f"{name} pooled primary {pl[0]:+.4f} [{pl[1]:+.4f}, {pl[2]:+.4f}] -> {verdict(per[name], pl)}", flush=True)
    X = np.c_[np.ones(len(D)), D.wp_prev]
    print("drop vs prior win pct, all seasons:", np.round(np.linalg.lstsq(X, D.part_win - D.part_out, rcond=None)[0], 4),
          "| production a, b from rest_window.csv")


# ---------------------------------------------------------------- R6 rest-rule confirmation (unseen seasons)
R6_WINDOW = {2018: "2018-03-05", 2019: "2019-03-11", 2023: "2023-03-06"}


def r6_rows(season):
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv", parse_dates=["date"])
    date_of = sch.groupby("scoring_period").date.first()
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0].copy()
    g["date"] = g.scoring_period.map(date_of)
    teams = sorted(sch.pro_team_id.unique())
    prev = team_results(season - 1, teams).groupby("pro_team_id").win.mean()
    lo = pd.Timestamp(R6_WINDOW[season]); hi = lo + pd.Timedelta(days=20)
    cut = pd.Timestamp(f"{season}-02-10")
    tdays = {t: set(x) for t, x in sch.groupby("pro_team_id").scoring_period}
    out = []
    for pid, gp in g.groupby("id"):
        days = set(gp.scoring_period)
        tm = [t for t in teams if days <= tdays[t]]
        if len(tm) != 1 or gp["min"].mean() < 25:
            continue
        tm = tm[0]
        ts = sch[sch.pro_team_id == tm]
        before, win = ts[ts.date < cut], ts[(ts.date >= lo) & (ts.date <= hi)]
        pb = gp[gp.date < cut]
        if len(pb) < 20 or len(win) == 0:
            continue
        out.append({"season": season, "id": pid, "team": tm, "pred0": min(len(pb) / len(before), 1.0),
                    "y": gp.date.between(lo, hi).sum() / len(win), "w": len(win), "wp_prev": prev.get(tm)})
    return pd.DataFrame(out)


def run_r6():
    rw = pd.read_csv(M.DATA / "rest_window.csv")
    a, b = np.linalg.lstsq(np.c_[np.ones(len(rw)), rw.wp_prev], rw["drop"], rcond=None)[0]
    print(f"R6: production rule frozen: a {a:.4f}, b {b:.4f}, REST_BASE {M.REST_BASE}", flush=True)
    dd, cl = [], []
    for t in (2018, 2019, 2023):
        d = r6_rows(t).dropna(subset=["wp_prev"])
        mult = np.clip(1 + (a + b * d.wp_prev) / M.REST_BASE, 0.5, 1.05)
        e0 = d.w * (d.y - d.pred0) ** 2 / d.w.mean()
        e1 = d.w * (d.y - d.pred0 * mult) ** 2 / d.w.mean()
        diff = (e1 - e0).to_numpy()
        m = diff.mean()
        print(f"{t}: players {len(d)} teams {d.team.nunique()} | window share actual {np.average(d.y, weights=d.w):.3f} "
              f"pred no-rule {np.average(d.pred0, weights=d.w):.3f} rule {np.average(d.pred0 * mult, weights=d.w):.3f} | "
              f"error no-rule {e0.mean():.4f} rule {e1.mean():.4f} diff {m:+.4f}", flush=True)
        dd.append(diff); cl.append(np.char.add(str(t) + "_", d.team.astype(str).to_numpy()))
    global LEVEL
    keep, LEVEL = LEVEL, 0.95
    m, lo, hi = boot_diff(np.concatenate(dd), np.concatenate(cl))
    LEVEL = keep
    out = "VALIDATED (keep)" if hi < 0 else "REMOVE" if lo > 0 else "INCONCLUSIVE (keep, unvalidated)"
    print(f"R6 pooled diff {m:+.4f} [95%: {lo:+.4f}, {hi:+.4f}] -> {out}", flush=True)


# ---------------------------------------------------------------- P1 precision of kappa_c and psi (engine L4)
def run_p1():
    import research_context as RC
    import research_live2 as R2
    M.set_kappa(1.0)
    RC.ARMS = ["C13"]
    p11, S = R2.fit_p11(FIT)
    fitted = {}
    rows_all = []
    for seed in (11, 12, 13, 14, 15):
        d = R2.load(FIT, n_match=300, seed=seed)
        cx = RC.context(FIT, d)
        if not fitted:
            K, _ = RC.fit_c1(d, cx)
            P0 = RC.params(FIT, d, cx)
            beta, _ = RC.fit_c3(d, cx, P0["x"])
            fitted = {"K": K, "beta": beta, "rho": None}
        P = RC.params(FIT, d, cx, fitted)
        rows = RC.cases(d, cx, P, p11, S, days=range(1, 7))
        kc, psi = RC.fit_l3(rows, "C13")
        print(f"P1 draw seed {seed}: kappa_c {kc:.3f} psi {psi:.3f}", flush=True)
        for r in rows:
            r["match"] = f"{seed}_{r['match']}"
        rows_all += rows
    kc, psi = RC.fit_l3(rows_all, "C13")
    print(f"P1 pooled 1,500 matches: kappa_c {kc:.3f} psi {psi:.3f} (production 1.041 / 0.350)", flush=True)
    per, dd, cl = {}, [], []
    for t in PRIMARY:
        d = R2.load(t, n_match=300, seed=3)
        cx = RC.context(t, d)
        rows = RC.cases(d, cx, RC.params(t, d, cx, fitted), p11, S, days=range(0, 7))
        st = RC.stack(rows, "C13")
        sb, sc = RC.score(st, 1.041, 0.350), RC.score(st, kc, psi)
        diff = ((sc.p - sc.y) ** 2 - (sb.p - sb.y) ** 2).to_numpy()
        m, lo, hi = boot_diff(diff, sc.match.to_numpy())
        cal = RC.calibration(sc)
        gap = float((cal[cal.n >= 200].pred - cal[cal.n >= 200].obs).abs().max())
        print(f"P1 {t}: production {((sb.p - sb.y) ** 2).mean():.5f} refit {((sc.p - sc.y) ** 2).mean():.5f} diff {m:+.6f} "
              f"[{lo:+.6f}, {hi:+.6f}] worst calibration {gap:.3f}", flush=True)
        per[t] = m if gap <= 0.04 else 1.0
        dd.append(diff); cl.append(np.char.add(str(t) + "_", sc.match.astype(str).to_numpy()))
    pooled = boot_diff(np.concatenate(dd), np.concatenate(cl))
    print(f"P1 pooled diff {pooled[0]:+.6f} [{pooled[1]:+.6f}, {pooled[2]:+.6f}] -> {verdict(per, pooled)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "r2ab":
        run_r2(("R2ab",))
    else:
        {"r1": run_r1, "r5": run_r5, "r2": run_r2, "r3": run_r3, "r4": run_r4, "r6": run_r6, "p1": run_p1}[which]()
