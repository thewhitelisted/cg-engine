"""Candidate L1 (DECISIONS.md 2026-10-20): validate the live (mid-week) matchup model on 2025-26 box scores."""
import numpy as np
import pandas as pd

import model as M
import projections as Pj

KAPPA = 1.182
STATK = M.STATK
CAT_STATS = ["tpm", "reb", "ast", "stl", "blk", "tov", "pts"]


def load(season=2026, n_match=600, seed=3):
    g = pd.read_csv(M.DATA / f"gamelogs_{season}.csv")
    g = g[g["min"] > 0].copy()
    g["fgx"] = g.fgm - M.LG_FG * g.fga
    g["ftx"] = g.ftm - M.LG_FT * g.fta
    sch = pd.read_csv(M.DATA / f"schedule_{season}.csv")
    team_days = sch.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
    team = Pj.files()[season + 1 if season == 2026 else season]["pro_team_id"]
    st = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
    pool = []
    for pid in st[(st.n >= 40) & (st.mins >= 24)].index:
        tm = team.get(pid)
        if pd.isna(tm) or int(tm) not in team_days.index:
            continue
        if set(g.loc[g.id == pid, "scoring_period"]).issubset(set(team_days[int(tm)])):
            pool.append((pid, int(tm)))
    ids = np.array([p for p, _ in pool]); tms = np.array([t for _, t in pool])
    g = g[g.id.isin(ids)]
    mu = g.groupby("id")[STATK].mean().loc[ids].to_numpy()
    var_pg = M.per_game_var(mu)
    a = np.array([st.loc[p, "n"] / len(team_days[t]) for p, t in pool])
    days = np.arange(g.scoring_period.min(), g.scoring_period.max() + 1)
    D = len(days)
    day0 = days[0]
    # daily actual stats [P, D, S] and scheduled games [P, D]
    X = np.zeros((len(ids), D, len(STATK)))
    pidx = {p: i for i, p in enumerate(ids)}
    for r in g.itertuples(index=False):
        X[pidx[r.id], r.scoring_period - day0] = [getattr(r, k) for k in STATK]
    sched = np.zeros((len(ids), D))
    for i, t in enumerate(tms):
        sched[i, team_days[t] - day0] = 1
    week = (days // 7)
    weeks = [w for w in np.unique(week) if (week == w).sum() == 7]
    gm = np.array([sched[i].sum() / len(np.unique(week)) for i in range(len(ids))]) * a   # avg games/week x avail
    return dict(mu=mu, var=var_pg, a=a, X=X, sched=sched, week=week, weeks=weeks, gm=gm, n=len(ids), seed=seed, n_match=n_match)


def team_cov(m_by_player, a_sel):
    return m_by_player.sum(0)


def cat_probs(mA, vA, mB, vB, kappa=KAPPA):
    return M.Evaluator.cat_probs(mA, vA * kappa**2, mB, vB * kappa**2)


def run(d, kappa_live=KAPPA, days=range(0, 7)):
    rng = np.random.default_rng(d["seed"])
    rows = []
    for _ in range(d["n_match"]):
        A, B = np.split(rng.choice(d["n"], 26, replace=False), 2)
        for w in d["weeks"]:
            cols = np.where(d["week"] == w)[0]
            act = d["X"][:, cols]                                   # [P, 7, S]
            ta, tb = act[A].sum((0, 1)), act[B].sum((0, 1))
            if ta[M.SK["fga"]] == 0 or tb[M.SK["fga"]] == 0:
                continue
            obs = outcome(ta, tb)
            for day in days:
                pr = []
                for T_ in (A, B):
                    known = act[T_, :day].sum((0, 1))
                    nrem = d["sched"][T_][:, cols[day:]].sum(1)
                    if day == 0:                                     # pre-week production structure
                        g_ = d["gm"][T_][:, None]
                        m = (g_ * d["mu"][T_]).sum(0)
                        v = (g_ * d["var"][T_] + M.GAMES_VAR_RATIO * g_ * d["mu"][T_] ** 2).sum(0)
                    else:
                        ea = (nrem * d["a"][T_])[:, None]
                        m = known + (ea * d["mu"][T_]).sum(0)
                        v = (ea * d["var"][T_] + (nrem * d["a"][T_] * (1 - d["a"][T_]))[:, None] * d["mu"][T_] ** 2).sum(0)
                    pr.append((m, v))
                k_ = KAPPA if day == 0 else kappa_live
                pc = cat_probs(pr[0][0], np.maximum(pr[0][1], 1e-9), pr[1][0], np.maximum(pr[1][1], 1e-9), k_)
                pw = float(M.Evaluator.matchup_win(pc))
                won, lost = obs.sum(), (1 - obs).sum()
                y = 1.0 if won > lost else 0.0
                rows.append({"day": day, "p": pw, "y": y, **{f"c{i}": pc[i] for i in range(9)}, **{f"o{i}": obs[i] for i in range(9)}})
    return pd.DataFrame(rows)


def outcome(ta, tb):
    o = []
    for x, a_ in (("fgx", "fga"), ("ftx", "fta")):
        o.append(float(ta[M.SK[x]] / ta[M.SK[a_]] > tb[M.SK[x]] / tb[M.SK[a_]]))
    for s in CAT_STATS:
        o.append(float(ta[M.SK[s]] < tb[M.SK[s]]) if s == "tov" else float(ta[M.SK[s]] > tb[M.SK[s]]))
    return np.array(o)


def fit_kappa_live():
    """L1b: kappa for the live checkpoints, fit on 2024-25 by matchup Brier (days 1-6)."""
    from scipy.optimize import minimize_scalar
    d = load(season=2025, n_match=300, seed=5)
    def brier(k):
        r = run(d, k, days=range(1, 7))
        return ((r.p - r.y) ** 2).mean()
    return minimize_scalar(brier, bounds=(0.7, 2.0), method="bounded", options={"xatol": 0.01}).x


if __name__ == "__main__":
    import sys
    k_live = fit_kappa_live() if "--l1b" in sys.argv else KAPPA
    print(f"kappa_live = {k_live:.3f}")
    d = load()
    r = run(d, k_live)
    pd.set_option("display.width", 200)
    br = r.groupby("day").apply(lambda x: ((x.p - x.y) ** 2).mean(), include_groups=False)
    cb = r.groupby("day").apply(lambda x: np.mean([((x[f"c{i}"] - x[f"o{i}"]) ** 2).mean() for i in range(9)]), include_groups=False)
    print("matchup Brier by day (0 = pre-week model):", br.round(4).to_dict())
    print("category Brier by day:", cb.round(4).to_dict())
    live = r[r.day > 0]
    P = np.concatenate([live[f"c{i}"].to_numpy() for i in range(9)])
    O = np.concatenate([live[f"o{i}"].to_numpy() for i in range(9)])
    bins = pd.cut(P, np.linspace(0, 1, 11), include_lowest=True)
    cal = pd.DataFrame({"pred": P, "obs": O, "bin": bins}).groupby("bin", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
    print("\ncategory calibration, live checkpoints pooled:")
    print(cal.round(3).T.to_string())
    ok1 = all(br[day] < br[0] for day in range(1, 7))
    ok2 = ((cal.n < 200) | ((cal.pred - cal.obs).abs() <= 0.04)).all()
    print(f"\nrule (1) live beats pre-week at every day: {ok1}; rule (2) calibration within 0.04: {ok2} -> {'SHIP' if ok1 and ok2 else 'REJECT'}")
