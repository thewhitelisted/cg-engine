"""League-format simulation: P(championship) for The Cool Guys format.

Regular season: N_REG one-week matchups against randomly paired opponents; standings by total
categories won (ESPN "H2H Each Category"). Top 8 make the playoffs, bracket 1v8 / 4v5 / 2v7 / 3v6
(no reseeding), three one-week rounds in the fantasy-playoff weeks; more categories won advances
(ties -> higher seed).

Two entry points:
  p_champ_model(...)  model-based, from team weekly mean/var  -> used by sim.Draft (objective="champ")
  payout_model(...)   expected prize money (PRIZES for 1st/2nd/3rd) -> sim.Draft (objective="payout")
  p_champ_real(...)   from realised category results           -> used by realtest.py
"""
import numpy as np

import model as M

N_REG = 19
N_PLAYOFF = 8
BRACKET = [(0, 7), (3, 4), (1, 6), (2, 5)]  # seed indices; winners of pairs 0/1 and 2/3 meet
PRIZES = (150.0, 90.0, 60.0)                # 1st / 2nd / 3rd (3rd: semifinal losers meet in the final week)


def _random_pairings(rng, lead_shape, n_weeks, T):
    """Random perfect matchings: (a, b) each [*lead_shape, n_weeks, T//2]."""
    perm = np.argsort(rng.random(tuple(lead_shape) + (n_weeks, T)), axis=-1)
    return perm[..., 0::2], perm[..., 1::2]


def p_champ_model(mS, vS, me: int, rng, K: int = 64) -> np.ndarray:
    """mS, vS: team weekly totals [R, T, 4, S] (week 0 = avg regular week, 1-3 = playoff weeks).
    Returns P(we win the title) per rollout [R], averaged over K simulated seasons."""
    champ, _, _, _ = _places_model(mS, vS, rng, K)
    return (champ == me).mean(1)


def payout_model(mS, vS, me: int, rng, K: int = 64, prizes=PRIZES, pairs=None, tot0=None) -> np.ndarray:
    """Expected prize money per rollout [R] (same simulation as p_champ_model, same rng use)."""
    places = _places_model(mS, vS, rng, K, pairs, tot0)[:3]
    return sum(z * (pl == me).mean(1) for z, pl in zip(prizes, places))


def outcomes_by_rollout(mS, vS, rng, K: int = 64, prizes=PRIZES, pairs=None, tot0=None, per_week=False, week0_pc=None) -> dict:
    """Per-rollout, per-team outcomes [R, T]: expected prize money, P(title), P(playoffs). Used for paired
    comparisons of roster changes (same truth draws and rng for both sides)."""
    T = mS.shape[1]
    champ, second, third, seeds = _places_model(mS, vS, rng, K, pairs, tot0, per_week, week0_pc=week0_pc)
    team = np.arange(T)
    dollars = sum(z * (pl[..., None] == team).mean(1) for z, pl in zip(prizes, (champ, second, third)))
    return {"dollars": dollars, "title": (champ[..., None] == team).mean(1),
            "playoffs": (seeds[..., None] == team).any(-2).mean(1)}


def playoff_detail(mS, vS, rng, K: int = 64, pairs=None, tot0=None, per_week=False, week0_pc=None, prizes=PRIZES) -> dict:
    """Everything about the playoffs, per team: seed distribution, round-by-round survival, and who each team
    meets in each round with its win rate against them. Same simulation (and rng use) as season_outcomes."""
    T = mS.shape[1]
    d = _places_model(mS, vS, rng, K, pairs, tot0, per_week, detail=True, week0_pc=week0_pc)
    seeds, games = d["seeds"], d["games"]
    team = np.arange(T)
    p_seed = np.stack([(seeds[..., k] == t).mean() for t in team for k in range(N_PLAYOFF)]).reshape(T, N_PLAYOFF)
    out = {"p_seed": p_seed, "p_playoffs": p_seed.sum(1)}

    def meet(rounds):
        face, beat = np.zeros((T, T)), np.zeros((T, T))
        for x, y, w in rounds:
            x, y, w = x.ravel(), y.ravel(), w.ravel()
            np.add.at(face, (x, y), 1); np.add.at(face, (y, x), 1)
            np.add.at(beat, (x, y), w == x); np.add.at(beat, (y, x), w == y)
        n = seeds.shape[0] * seeds.shape[1]
        return face / n, np.divide(beat, face, out=np.full((T, T), np.nan), where=face > 0)

    for name, key in (("qf", "qf"), ("sf", "sf"), ("final", "final")):
        out[f"face_{name}"], out[f"beat_{name}"] = meet(games[key])
    reach = lambda rounds: np.array([np.mean([((x == t) | (y == t)).mean() for x, y, _ in rounds]) * len(rounds) for t in team])
    out["p_semis"], out["p_final"] = reach(games["sf"]), reach(games["final"])
    out["p_1st"] = np.array([(d["champ"] == t).mean() for t in team])
    out["p_2nd"] = np.array([(d["second"] == t).mean() for t in team])
    out["p_3rd"] = np.array([(d["third"] == t).mean() for t in team])
    out["exp_dollars"] = prizes[0] * out["p_1st"] + prizes[1] * out["p_2nd"] + prizes[2] * out["p_3rd"]
    tot = d["tot"]
    out["exp_cats"] = tot.mean((0, 1))
    out["title_by_seed"] = np.array([[np.mean(d["champ"][seeds[..., k] == t] == t) if (seeds[..., k] == t).any() else np.nan
                                      for k in range(N_PLAYOFF)] for t in team])
    out["pw_playoff"] = d["pw"].mean(0)                                                 # [T, T, 3] by playoff week
    return out


def season_outcomes(mS, vS, rng, K: int = 64, prizes=PRIZES, pairs=None, tot0=None, per_week=False, week0_pc=None) -> dict:
    """Per-team outcome probabilities [T], pooled over rollouts and simulated seasons. pairs: the remaining
    regular-season schedule as team-index pairs [weeks, T/2, 2]; tot0: categories won so far [T]."""
    T = mS.shape[1]
    champ, second, third, seeds = _places_model(mS, vS, rng, K, pairs, tot0, per_week, week0_pc=week0_pc)
    out = {k: np.array([(pl == t).mean() for t in range(T)]) for k, pl in
           (("p_1st", champ), ("p_2nd", second), ("p_3rd", third))}
    out["p_playoffs"] = np.array([(seeds == t).any(-1).mean() for t in range(T)])
    out["p_top_seed"] = np.array([(seeds[..., 0] == t).mean() for t in range(T)])
    out["exp_dollars"] = sum(z * out[k] for z, k in zip(prizes, ("p_1st", "p_2nd", "p_3rd")))
    return out


def _places_model(mS, vS, rng, K, pairs=None, tot0=None, per_week=False, detail=False, week0_pc=None):
    """Simulate K seasons per rollout; returns (champion, runner-up, third) [R, K] and playoff seeds [R, K, 8].

    Week layout of mS / vS [R, T, W, S]:
      W = 4:              week 0 = a typical regular-season week (used for every regular week), 1-3 = playoffs
      per_week=True, W = len(pairs) + 3: one column per remaining regular-season matchup (its own NBA games),
                          then the 3 playoff weeks
    """
    R, T, W = mS.shape[:3]
    ev = M.Evaluator.cat_probs
    po = slice(W - 3, W)
    pw = M.Evaluator.matchup_win(ev(mS[:, :, None, po], vS[:, :, None, po], mS[:, None, :, po], vS[:, None, :, po]))  # [R,T,T,3]
    if per_week:
        assert pairs is not None and W == len(pairs) + 3, "per-week layout needs one column per remaining week"
        P = np.asarray(pairs, int).reshape(-1, T // 2, 2)                                # [n, T/2, 2]
        wk = np.arange(len(P))[:, None]
        a0, b0 = P[..., 0], P[..., 1]
        pc = ev(mS[:, a0, wk], vS[:, a0, wk], mS[:, b0, wk], vS[:, b0, wk])            # [R, n, T/2, 9]
        if week0_pc is not None:                     # the week in progress: live category odds (live.py)
            pc[:, 0] = np.asarray(week0_pc)[None]
        m = np.broadcast_to(pc.sum(-1)[:, None], (R, K) + a0.shape)
        v = np.broadcast_to((pc * (1 - pc)).sum(-1)[:, None], (R, K) + a0.shape)
        a = np.broadcast_to(a0[None, None], m.shape)
        b = np.broadcast_to(b0[None, None], m.shape)
    else:
        pc = ev(mS[:, :, None, 0], vS[:, :, None, 0], mS[:, None, :, 0], vS[:, None, :, 0])   # [R, T, T, 9]
        cmean, cvar = pc.sum(-1), (pc * (1 - pc)).sum(-1)
        if pairs is None:
            a, b = _random_pairings(rng, (R, K), N_REG, T)                                # [R, K, N, T/2]
        else:                                                # the league's remaining schedule [n, T/2, 2]
            pairs = np.asarray(pairs, int).reshape(-1, T // 2, 2)
            a = np.broadcast_to(pairs[None, None, ..., 0], (R, K) + pairs.shape[:2])
            b = np.broadcast_to(pairs[None, None, ..., 1], (R, K) + pairs.shape[:2])
        ri = np.arange(R)[:, None, None, None]
        m, v = cmean[ri, a, b], cvar[ri, a, b]
    ca = np.clip(m + np.sqrt(v) * rng.standard_normal(m.shape), 0, 9)
    tot = np.zeros((R, K, T)) + (0 if tot0 is None else np.asarray(tot0, float))
    rk = np.broadcast_to(np.arange(R)[:, None, None, None], a.shape)
    kk = np.broadcast_to(np.arange(K)[None, :, None, None], a.shape)
    np.add.at(tot, (rk, kk, a), ca)
    np.add.at(tot, (rk, kk, b), 9 - ca)
    seeds = np.argsort(-(tot + 1e-6 * rng.random(tot.shape)), axis=-1)[..., :N_PLAYOFF]  # [R, K, 8]

    u = rng.random((R, K, 3, 4))
    if detail:
        champ, second, third, games = _bracket_vec(seeds, pw, u, detail=True)
        return {"champ": champ, "second": second, "third": third, "seeds": seeds, "games": games, "tot": tot, "pw": pw}
    return (*_bracket_vec(seeds, pw, u), seeds)


def _bracket_vec(seeds, pw, u, detail=False):
    """Vectorised bracket for model probabilities. seeds [R, K, 8], pw [R, T, T, 3], u [R, K, 3, 4].
    Returns (champion, runner-up, third); third is the winner of the semifinal losers' game. With detail,
    also every game as (team x, team y, winner) by round."""
    R = seeds.shape[0]
    ri = np.arange(R)[:, None]
    def game(x, y, rnd, slot):
        return np.where(u[:, :, rnd, slot] < pw[ri, x, y, rnd], x, y)
    w = [game(seeds[..., a], seeds[..., b], 0, i) for i, (a, b) in enumerate(BRACKET)]
    s1, s2 = game(w[0], w[1], 1, 0), game(w[2], w[3], 1, 1)
    l1, l2 = np.where(s1 == w[0], w[1], w[0]), np.where(s2 == w[2], w[3], w[2])
    champ = game(s1, s2, 2, 0)
    third = game(l1, l2, 2, 1)
    second = np.where(champ == s1, s2, s1)
    if detail:
        qf = [(seeds[..., a], seeds[..., b], w[i]) for i, (a, b) in enumerate(BRACKET)]
        games = {"qf": qf, "sf": [(w[0], w[1], s1), (w[2], w[3], s2)], "final": [(s1, s2, champ)], "third": [(l1, l2, third)]}
        return champ, second, third, games
    return champ, second, third


def p_champ_real(cats: np.ndarray, reg_idx, po_idx, me: int, rng, K: int = 500) -> float:
    """cats [W, T, T]: categories team a won vs team b in week w (real totals). P(title) over K schedules.
    Playoff game: more categories won advances; equal -> higher seed."""
    W, T, _ = cats.shape
    reg_idx, po_idx = np.asarray(reg_idx), np.asarray(po_idx)
    n_reg = min(N_REG, len(reg_idx))
    wk = np.stack([rng.choice(reg_idx, n_reg, replace=len(reg_idx) < N_REG) for _ in range(K)])  # [K, n]
    a, b = _random_pairings(rng, (K,), n_reg, T)
    ca = cats[wk[..., None], a, b]
    cb = cats[wk[..., None], b, a]
    tot = np.zeros((K, T))
    kk = np.broadcast_to(np.arange(K)[:, None, None], a.shape)
    np.add.at(tot, (kk, a), ca)
    np.add.at(tot, (kk, b), cb)
    seeds = np.argsort(-(tot + 1e-6 * rng.random(tot.shape)), axis=-1)[:, :N_PLAYOFF]

    def game(x, y, rnd):
        w = po_idx[rnd]
        cx, cy = cats[w, x, y], cats[w, y, x]
        sx = np.argmax(seeds == x[:, None], axis=1)
        sy = np.argmax(seeds == y[:, None], axis=1)
        return np.where((cx > cy) | ((cx == cy) & (sx < sy)), x, y)

    w = [game(seeds[:, i], seeds[:, j], 0) for i, j in BRACKET]
    s1, s2 = game(w[0], w[1], 1), game(w[2], w[3], 1)
    champ = game(s1, s2, 2)
    return float((champ == me).mean())
