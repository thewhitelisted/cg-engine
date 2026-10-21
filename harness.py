"""Shared draft runner for backtests: one full 12-team snake draft with a given strategy in our slot.

Opponents always use the noisy-board model with the given (true) behaviour parameters; boards are
seeded by `seed` only, so every strategy faces identical opponent tendencies.
"""
import numpy as np

import model as M
import sim as S

TOOL_VARIANTS = {
    # name: (projection source, Draft kwargs)
    "tool": ("stack", dict(mode="mc", waivers=True)),
    "tool_nowaiver": ("stack", dict(mode="mc", waivers=False)),
    "tool_analytic": ("stack", dict(mode="analytic")),
    "tool_espnproj": ("espn", dict(mode="mc", waivers=True)),
    # candidate 1 (DECISIONS.md): optimise P(championship) directly
    "tool_champ": ("stack", dict(mode="mc", waivers=True, objective="champ")),
    # candidate 2: + calibrated matchup sd (kappa fit on 2024-25, validated on 2025-26)
    "tool_champ_k": ("stack", dict(mode="mc", waivers=True, objective="champ", kappa=1.182)),
    # candidate 8: + market-informed projections (beta fit on 2023-24)
    "tool_champ_k_mkt": ("stack_mkt", dict(mode="mc", waivers=True, objective="champ", kappa=1.182)),
    # candidate 10: market adjustment only for players with a real ADP
    "tool_champ_k_mkt10": ("stack_mkt10", dict(mode="mc", waivers=True, objective="champ", kappa=1.182)),
    # belief-layer redesign (V2): season-varying market weights carried into every simulated season
    "tool_belief": ("stack_belief", dict(mode="mc", waivers=True, objective="champ", kappa=1.182)),
}
BASELINES = ("value", "adp")


def run_draft(t: int, strategy: str, seed: int, my_slot: int = 12, n_roll: int = 120,
              opp_sharp: float = S.SHARP_FRAC, opp_noise: float = 1.0, assume: dict | None = None):
    """Returns final rosters [T, ROSTER] as indices into the stack-projection Season(t).df."""
    proj, kw = TOOL_VARIANTS.get(strategy, ("stack", {}))
    kw = dict(kw)
    prev_kappa = M.MATCHUP_SD_KAPPA
    M.set_kappa(kw.pop("kappa", 1.0))
    try:
        return _run_draft(t, strategy, seed, my_slot, n_roll, opp_sharp, opp_noise, assume, proj, kw)
    finally:
        M.set_kappa(prev_kappa)


def _run_draft(t, strategy, seed, my_slot, n_roll, opp_sharp, opp_noise, assume, proj, kw):
    base = M.get_season(t, "stack")
    season = M.get_season(t, proj)
    d = S.Draft(season, my_slot=my_slot, **kw, **(assume or {}))
    # map between this season's df and the stack df (same players, possibly different order)
    to_base = base.df.reset_index().set_index("id")["index"].reindex(season.df["id"]).to_numpy()
    ref = S.Draft(base, my_slot=my_slot) if season is not base else d
    rng = np.random.default_rng(10_000 + seed + 977 * t)
    boards = ref.boards(rng, 1, sharp_frac=opp_sharp, noise_mult=opp_noise)[0]   # [T, P_base]
    from_base = season.df.reset_index().set_index("id")["index"].reindex(base.df["id"]).to_numpy()
    static = ref.Z.sum(1)
    for k in range(d.n_picks):
        t_ = d.order[k]
        avail_b = np.ones(len(base.df), bool)
        picked_b = [to_base[p] for p in d.picks if p >= 0]
        avail_b[picked_b] = False
        if t_ != d.me:
            pb = int(np.where(avail_b, boards[t_], np.inf).argmin())
            p = from_base[pb]
            p = -1 if np.isnan(p) else int(p)
        elif strategy in TOOL_VARIANTS:
            res = d.evaluate(d.candidate_pool(12), n_roll=n_roll, seed=seed * 100 + k)
            p = int(res["idx"].iloc[0])
        elif strategy == "adp":
            p = int(from_base[np.where(avail_b, ref.mkt, np.inf).argmin()])
        elif strategy == "value":
            p = int(from_base[np.where(avail_b, static, -np.inf).argmax()])
        else:
            raise ValueError(strategy)
        d.picks.append(p)
    R = np.full((M.N_TEAMS, M.ROSTER), -1)
    for team in range(M.N_TEAMS):
        r = [int(to_base[p]) for p in d.roster(team) if p >= 0]
        R[team, : len(r)] = r
    return R
