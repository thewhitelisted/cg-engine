"""Monte Carlo draft engine: evaluate candidate picks by simulating the rest of the draft.

Opponents: each gets a private big board = market rank (ESPN ADP / ESPN rank), for "sharp"
opponents blended with our value rank, plus per-player noise. They take the best available
from their board. Which opponents are sharp is re-drawn every rollout.

Us: in rollouts we draft by category-weighted value, weights = gradient of P(win matchup) w.r.t.
each category at our projected end-of-draft team (H-score style; produces punts endogenously).

Scoring a rollout (mode="mc", default): draw a simulated true season (correlated projection errors,
empirical games-played misses), let every team cut busts for waiver players (model.LEARN_FRAC),
then compute our mean P(win weekly matchup) vs the 11 opponents, weighted REG_WEIGHT on an
average regular-season week and the rest on the 3 fantasy-playoff weeks. Truth draws and opponent
boards are shared across candidates (common random numbers).
mode="analytic" scores with projection uncertainty as extra variance instead (faster, no waivers).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import league as L
import model as M

REG_WEIGHT = 0.5
SHARP_FRAC = 0.5
SHARP_LAMBDA = 0.6            # sharp opponent board = (1-l)*market + l*value rank
NOISE_A, NOISE_B = 2.0, 0.15  # board noise sd = A + B * rank


def pick_owner(k: int, n_teams: int = M.N_TEAMS) -> int:
    r, i = divmod(k, n_teams)
    return i if r % 2 == 0 else n_teams - 1 - i


@dataclass
class Draft:
    season: M.Season
    my_slot: int = 12                          # 1-based
    picks: list = field(default_factory=list)  # player indices in pick order (-1 = unmodeled player)
    mode: str = "mc"                           # "mc" or "analytic"
    waivers: bool = True
    sharp_frac: float = SHARP_FRAC             # what we ASSUME about opponents
    noise_mult: float = 1.0
    objective: str = "matchup"                 # "matchup" (v1: weekly win prob), "champ" (P(title)) or "payout" ($)
    policy: str = "gradient"                   # our future picks in rollouts: "gradient" weights or "static" value
    champ_k: int = 48                          # simulated seasons per rollout for objective="champ"

    def __post_init__(self):
        self.df = self.season.df
        self.ev = M.Evaluator(self.season, proj_unc=True)
        self.P = len(self.df)
        self.n_picks = M.N_TEAMS * M.ROSTER
        self.me = self.my_slot - 1
        self.order = np.array([pick_owner(k) for k in range(self.n_picks)])
        val, _ = M.static_value(self.df)
        self.Z = np.stack([M.static_value.last_by_cat[c] for c in M.CATS], axis=1)  # [P, 9]
        self.vrank = np.argsort(np.argsort(-val)) + 1.0
        self.mkt = self.df["mkt"].to_numpy(float)
        self.avg_pool = np.argsort(self.mkt)[: self.n_picks]
        self._dz = self._z_units()

    # ---- state helpers -------------------------------------------------
    def roster(self, team: int) -> list[int]:
        return [p for k, p in enumerate(self.picks) if self.order[k] == team]

    def available(self) -> np.ndarray:
        a = np.ones(self.P, bool)
        a[[p for p in self.picks if p >= 0]] = False
        return a

    def my_next_picks(self, start: int | None = None) -> list[int]:
        start = len(self.picks) if start is None else start
        return [k for k in range(start, self.n_picks) if self.order[k] == self.me]

    def boards(self, rng, R: int, sharp_frac=None, noise_mult=None) -> np.ndarray:
        """Opponent big boards [R, T, P] (lower = better)."""
        sf = self.sharp_frac if sharp_frac is None else sharp_frac
        nm = self.noise_mult if noise_mult is None else noise_mult
        sharp = rng.random((R, M.N_TEAMS)) < sf
        basis = np.where(sharp[..., None], (1 - SHARP_LAMBDA) * self.mkt + SHARP_LAMBDA * self.vrank, self.mkt)
        return basis + rng.standard_normal((R, M.N_TEAMS, self.P)) * (NOISE_A + NOISE_B * basis) * nm

    # ---- category weights (H-score gradient) ---------------------------
    def need_weights(self, my_players: list[int]) -> np.ndarray:
        """d P(win matchup) / d z_cat at a projected final team, normalised to mean 1."""
        ev = self.ev
        n_left = M.ROSTER - len(my_players)
        avail = self.available()
        avail[[p for p in my_players if p >= 0]] = False
        fut = self.my_next_picks(len(self.picks) + 1)[:n_left] if n_left else []
        cand = np.where(avail)[0]
        cand = cand[np.argsort(self.mkt[cand])]
        filler = [cand[min(max(k - len(self.picks), 0), len(cand) - 1)] for k in fut]
        team = np.array(list(my_players) + filler + [-1] * (n_left - len(filler)))[None]
        mA, vA = ev.team_stats(team)
        mB = ev.wmu[self.avg_pool].mean(0) * M.ROSTER
        vB = ev.wvar[self.avg_pool].mean(0) * M.ROSTER
        base = self._J(mA, vA, mB[None], vB[None])
        g = np.zeros(9)
        for c, (stat_i, shift) in enumerate(self._dz):
            mA2 = mA.copy()
            mA2[..., stat_i] += shift
            g[c] = (self._J(mA2, vA, mB[None], vB[None]) - base).item()
        g = np.maximum(g, 1e-4 * max(g.max(), 1e-9))
        return g / g.mean()

    def _z_units(self):
        """Per category: (stat index, weekly-mean shift equal to +0.25 z for one player)."""
        weekly = self.ev.wmu.mean(axis=1)
        out = []
        for c, s in enumerate(["fgx", "ftx", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"]):
            x, y = self.Z[:, c], weekly[:, M.SK[s]]
            out.append((M.SK[s], 0.25 * np.cov(x, y)[0, 1] / x.var()))
        return out

    def _J(self, mA, vA, mB, vB):
        w = self.ev.matchup_win(self.ev.cat_probs(mA, vA, mB, vB))  # [..., week]
        return REG_WEIGHT * w[..., 0] + (1 - REG_WEIGHT) * w[..., 1:].mean(-1)

    # ---- rollouts --------------------------------------------------------
    def _rollout(self, c, R, boards, avail0):
        """Finish the draft in R rollouts after we take candidate c. Returns rosters [R, T, ROSTER]."""
        k0 = len(self.picks)
        T, P = M.N_TEAMS, self.P
        myval = self.Z @ (self.need_weights(self.roster(self.me) + [c]) if self.policy == "gradient" else np.ones(9))
        taken = np.broadcast_to(~avail0, (R, P)).copy()
        taken[:, c] = True
        rost = np.full((R, T, M.ROSTER), -1, int)
        counts = np.zeros((R, T), int)
        for t in range(T):
            r = self.roster(t) + ([c] if t == self.me else [])
            rost[:, t, : len(r)] = r
            counts[:, t] = len(r)
        ar = np.arange(R)
        for k in range(k0 + 1, self.n_picks):
            t = self.order[k]
            if t == self.me:
                p = np.where(taken, -np.inf, myval[None, :]).argmax(1)
            else:
                p = np.where(taken, np.inf, boards[:, t, :]).argmin(1)
            taken[ar, p] = True
            rost[ar, t, counts[ar, t]] = p
            counts[ar, t] += 1
        return rost

    def score_rosters(self, rost, truth=None, champ_seed=None):
        """Our J per rollout [R] and reg-week category win probs [R, 11, 9] for final rosters [R, T, n]."""
        if truth is None:
            mS, vS = self.ev.team_stats(rost)
        else:
            mS, vS = self.ev.truth_team_stats(truth, rost, waivers=self.waivers)
        opp = [t for t in range(M.N_TEAMS) if t != self.me]
        me_m, me_v = mS[:, self.me : self.me + 1], vS[:, self.me : self.me + 1]
        if self.objective == "champ":
            J = L.p_champ_model(mS, vS, self.me, np.random.default_rng(champ_seed), K=self.champ_k)
        elif self.objective == "payout":
            J = L.payout_model(mS, vS, self.me, np.random.default_rng(champ_seed), K=self.champ_k)
        else:
            J = self._J(me_m, me_v, mS[:, opp], vS[:, opp]).mean(1)
        cats = self.ev.cat_probs(me_m[:, :, 0], me_v[:, :, 0], mS[:, opp, 0], vS[:, opp, 0])
        return J, cats

    def evaluate(self, candidates: list[int], n_roll: int = 200, seed: int = 0) -> pd.DataFrame:
        """Score each candidate for the current pick (must be ours)."""
        assert self.order[len(self.picks)] == self.me, "not our pick"
        rng = np.random.default_rng(seed)
        boards = self.boards(rng, n_roll)
        truth = self.ev.draw_truth(n_roll, rng) if self.mode == "mc" else None
        avail0 = self.available()
        rows, Js = [], []
        for c in candidates:
            rost = self._rollout(c, n_roll, boards, avail0)
            J, cats = self.score_rosters(rost, truth, champ_seed=seed + 1)
            Js.append(J)
            rows.append({"idx": c, "J": J.mean(), "cats_won": cats.sum(-1).mean(),
                         **{f"p_{cat}": cats[..., i].mean() for i, cat in enumerate(M.CATS)}})
        out = pd.DataFrame(rows)
        # paired standard error of each candidate's gap to the best (CRN makes these much tighter)
        Js = np.array(Js)
        self.last_J = dict(zip(candidates, Js))  # per-rollout values, for adaptive refinement
        best = Js[out["J"].idxmax()]
        out["se_vs_best"] = (Js - best).std(1) / np.sqrt(n_roll)
        out["name"] = self.df.loc[out["idx"], "name"].to_numpy()
        out["weights"] = [self.need_weights(self.roster(self.me) + [c]) for c in candidates]
        return out.sort_values("J", ascending=False).reset_index(drop=True)

    def survival(self, n_roll: int = 300, seed: int = 1) -> np.ndarray:
        """P(player still available at our next turn (skipping the 2nd pick of a back-to-back))."""
        k0 = len(self.picks)
        nxt = [k for k in self.my_next_picks(k0 + 1) if any(self.order[j] != self.me for j in range(k0, k))]
        if not nxt:
            return np.zeros(self.P)
        rng = np.random.default_rng(seed)
        boards = self.boards(rng, n_roll)
        taken = np.broadcast_to(~self.available(), (n_roll, self.P)).copy()
        ar = np.arange(n_roll)
        for k in range(k0, nxt[0]):
            t = self.order[k]
            if t == self.me:
                continue
            p = np.where(taken, np.inf, boards[:, t, :]).argmin(1)
            taken[ar, p] = True
        return 1 - taken.mean(0)

    def candidate_pool(self, k: int = 14) -> list[int]:
        avail = np.where(self.available())[0]
        mine = self.roster(self.me)
        w = self.need_weights(mine) if mine else np.ones(9)
        v = self.Z[avail] @ w
        by_val = avail[np.argsort(-v)][:k]
        by_mkt = avail[np.argsort(self.mkt[avail])][: k // 2]
        return list(dict.fromkeys(list(by_val) + list(by_mkt)))

    def evaluate_adaptive(self, candidates, budget_s: float = 30.0, batch: int = 200,
                          z_stop: float = 2.0, z_prune: float = 3.0, seed: int = 0) -> pd.DataFrame:
        """Add rollout batches (fresh seeds, CRN within each batch) until the best candidate leads the
        runner-up by z_stop paired standard errors, or the time budget runs out. Candidates trailing the
        leader by more than z_prune se are dropped. Returns the final table with `resolved` flag."""
        import time

        t0 = time.time()
        alive = list(candidates)
        acc = {c: [] for c in candidates}
        meta = {}
        b = 0
        while True:
            res = self.evaluate(alive, n_roll=batch, seed=seed + 1000 * b)
            for c in alive:
                acc[c].append(self.last_J[c])
            for _, r in res.iterrows():
                meta[r["idx"]] = r
            b += 1
            n = len(acc[alive[0]]) * batch
            J = {c: np.concatenate(acc[c]) for c in alive}
            best = max(alive, key=lambda c: J[c].mean())
            z = {c: (J[best] - J[c]).mean() / max((J[best] - J[c]).std() / np.sqrt(n), 1e-12) for c in alive}
            runner = max((c for c in alive if c != best), key=lambda c: J[c].mean(), default=None)
            resolved = runner is None or z[runner] >= z_stop
            elapsed = time.time() - t0
            per_batch = elapsed / b
            if resolved or elapsed + per_batch > budget_s:
                break
            alive = [c for c in alive if c == best or z[c] < z_prune]
        # Report every candidate on the leader's scale: leader's full-sample mean + paired gap measured
        # on the batches both were evaluated on (pruned candidates stop early; raw means over different
        # batches are not comparable because batches share common random shocks).
        Jbest = np.concatenate(acc[best])
        rows = []
        for c in candidates:
            Jc = np.concatenate(acc[c])
            m = len(Jc)
            diff = Jc - Jbest[:m]
            r = meta[c].copy()
            r["J"] = Jbest.mean() + (0.0 if c == best else diff.mean())
            r["se_vs_best"] = 0.0 if c == best else diff.std() / np.sqrt(m)
            r["n_roll"] = m
            rows.append(r)
        out = pd.DataFrame(rows).sort_values("J", ascending=False).reset_index(drop=True)
        out.attrs.update(resolved=bool(resolved), seconds=round(elapsed, 1), batches=b)
        return out
