"""Simulated-truth backtest for the live 2026-27 season + opponent-model sensitivity.

Rosters are drafted on projections, then scored on simulated true seasons drawn from the fitted
projection-error model (correlated stat errors, empirical games-played misses), with every team
cutting busts for waiver players. Complements realtest.py (real seasons, but only 3 of them).

    python3 backtest.py strategies [n_drafts]        # tool variants vs baselines
    python3 backtest.py opponents [n_drafts]         # cost of assuming the wrong opponent behaviour
"""
import sys
import time

import numpy as np
import pandas as pd

import harness as H
import league as L
import model as M
import sim as S

T_LIVE = 2027


class SimTruth:
    def __init__(self, n_truth=24, seed=7, proj="stack_belief"):
        # truth is drawn from the validated belief layer (season-varying market weights)
        self.season = M.get_season(T_LIVE, proj)
        self.d = S.Draft(self.season)
        self.ev = M.Evaluator(self.season, proj_unc=False)
        self.truth = self.ev.draw_truth(n_truth, np.random.default_rng(seed))
        self.n = n_truth

    SCORE_KAPPA = 1.182  # calibrated (DECISIONS.md, candidate 2), same for every strategy

    def score(self, R, me=11, metric="J"):
        prev = M.MATCHUP_SD_KAPPA
        M.set_kappa(self.SCORE_KAPPA)
        try:
            return self._score(R, me, metric)
        finally:
            M.set_kappa(prev)

    def _score(self, R, me, metric):
        Rb = np.broadcast_to(R, (self.n,) + R.shape).copy()
        mS, vS = self.ev.truth_team_stats(self.truth, Rb, waivers=True)
        if metric == "champ":
            return float(L.p_champ_model(mS, vS, me, np.random.default_rng(4321), K=400).mean())
        opp = [t for t in range(M.N_TEAMS) if t != me]
        J = self.d._J(mS[:, me : me + 1], vS[:, me : me + 1], mS[:, opp], vS[:, opp])
        return float(J.mean())


def strategies(n_drafts=24, strats=None):
    st = SimTruth()
    strats = strats or list(H.TOOL_VARIANTS) + list(H.BASELINES)
    rows, rosters, t0 = [], {}, time.time()
    for seed in range(n_drafts):
        for s in strats:
            R = H.run_draft(T_LIVE, s, seed)
            rosters[f"{seed}_{s}"] = R
            rows.append({"seed": seed, "strategy": s, "champ": st.score(R, metric="champ"), "J": st.score(R)})
        print(f"draft {seed + 1}/{n_drafts} {time.time() - t0:.0f}s", flush=True)
        tag = "" if strats is None else "_" + "_".join(strats)
        pd.DataFrame(rows).to_csv(M.DATA / f"simtest{tag}.csv", index=False)
        np.savez(M.DATA / f"rosters_sim{tag}.npz", **rosters)
    if strats is None or "tool" in strats:
        report(pd.DataFrame(rows))


def report(res, ref="tool"):
    for metric in ("champ", "J"):
        p = res.pivot(index="seed", columns="strategy", values=metric)
        print(f"\nsimulated-truth {metric}, slot 12, 2026-27:")
        for s in p.columns:
            d = p[s] - p[ref]
            print(f"  {s:15} {p[s].mean():.4f}   minus {ref}: {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")


def opponents(n_drafts=16):
    """Regret of assuming default opponents when they truly behave differently."""
    st = SimTruth()
    scenarios = [(0.0, 1.0), (1.0, 1.0), (0.5, 0.5), (0.5, 1.8), (0.0, 1.8), (1.0, 0.5)]
    rows, t0 = [], time.time()
    for sf, nm in scenarios:
        for seed in range(n_drafts):
            kw = dict(opp_sharp=sf, opp_noise=nm)
            j_def = st.score(H.run_draft(T_LIVE, "tool", seed, **kw))
            j_orc = st.score(H.run_draft(T_LIVE, "tool", seed, assume=dict(sharp_frac=sf, noise_mult=nm), **kw))
            j_adp = st.score(H.run_draft(T_LIVE, "adp", seed, **kw))
            rows.append({"sharp": sf, "noise": nm, "seed": seed, "default": j_def, "oracle": j_orc, "adp": j_adp})
        print(f"scenario sharp={sf} noise={nm} done {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(M.DATA / "opponent_sensitivity.csv", index=False)
    res["regret"] = res["oracle"] - res["default"]
    g = res.groupby(["sharp", "noise"])
    out = g[["default", "oracle", "adp", "regret"]].mean()
    out["regret_se"] = g["regret"].std() / np.sqrt(n_drafts)
    print("\nP(win wk) when true opponents differ from the tool's assumption (sharp=0.5, noise=1):")
    print(out.round(3).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "strategies"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (24 if mode == "strategies" else 16)
    extra = sys.argv[3:]
    if mode == "strategies":
        strategies(n, extra or None)
    else:
        opponents(n)
