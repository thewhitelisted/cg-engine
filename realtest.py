"""Out-of-sample backtest on REAL seasons (2023-24, 2024-25, 2025-26).

For season t: the model is rebuilt from data before t (model.Season(t)); opponents draft from that
season's preseason ADP (2025-26: ESPN-projection proxy). Each strategy drafts in slot 12 against the
same simulated opponents. Rosters are then scored on actual weekly box scores: every week, our team vs
each of the 11 opponents, categories decided by real totals, matchup = more categories won (tie 0.5).

In-season management, identical for every team and strategy, using only games already played:
  * at weeks 4, 8, 12, 16 each team (random order) swaps a rostered player for the best free agent
    when the FA's season-to-date value exceeds his by > SWAP_MARGIN (up to 2 swaps per team);
  * in weeks a rostered player plays 0 games, INJURY_RECOVERY of his team's games are streamed
    with a replacement-level line (IR + waiver streaming).

Score = 0.5 * regular-season matchup win rate + 0.5 * fantasy-playoff-week win rate.

    python3 realtest.py [n_drafts] [strategies...]
"""
import sys
import time

import numpy as np
import pandas as pd

import harness as H
import league as L
import model as M

SWAP_WEEKS = (4, 8, 12, 16)
SWAP_MARGIN = 0.5
MAX_SWAPS = 2


class RealSeason:
    def __init__(self, t: int):
        self.t = t
        self.s = M.get_season(t, "stack")
        df = self.s.df
        g = pd.read_csv(M.DATA / f"gamelogs_{t}.csv")
        g = g[g["min"] > 0]
        sch = pd.read_csv(M.DATA / f"schedule_{t}.csv", parse_dates=["date"])
        g["date"] = g.scoring_period.map(sch.groupby("scoring_period").date.first())
        g["wk"] = (g["date"] - self.s.season_start).dt.days // 7
        g["fgx"] = g.fgm - M.LG_FG * g.fga
        g["ftx"] = g.ftm - M.LG_FT * g.fta
        self.weeks = sorted(set(self.s.reg_weeks) | set(self.s.playoff_weeks))
        W = len(self.weeks)
        widx = {w: i for i, w in enumerate(self.weeks)}
        pidx = {pid: i for i, pid in enumerate(df["id"])}
        g = g[g.id.isin(pidx) & g.wk.isin(widx)]
        tot = g.groupby(["id", "wk"])[M.STATK].sum()
        n = g.groupby(["id", "wk"]).size()
        self.T = np.zeros((len(df), W, len(M.STATK)))   # actual weekly totals
        self.N = np.zeros((len(df), W))                  # games played
        for (pid, wk), row in tot.iterrows():
            self.T[pidx[pid], widx[wk]] = row.to_numpy()
            self.N[pidx[pid], widx[wk]] = n[(pid, wk)]
        # scheduled team games per week, for injury streaming
        tg = self.s.wg.reindex(columns=self.weeks, fill_value=0)
        self.G = np.stack([df["pro_team_id"].map(tg[w]).fillna(0).to_numpy() for w in self.weeks], 1)
        self.is_po = np.array([w in self.s.playoff_weeks for w in self.weeks])
        ev = M.Evaluator(self.s, proj_unc=False)
        self.repl_mu, self.vscale = ev.repl_mu, ev.vscale

    def to_date_value(self, upto: int) -> np.ndarray:
        """Season-to-date value (per-game line x share of team games played) using weeks < upto."""
        tot, n, g = self.T[:, :upto].sum(1), self.N[:, :upto].sum(1), self.G[:, :upto].sum(1)
        mu = tot / np.maximum(n, 1)[:, None]
        return M.value_from(mu, n / np.maximum(g, 1), self.vscale)

    def score(self, rosters: np.ndarray, me: int, rng) -> dict:
        """rosters [T, n] -> our real matchup results."""
        rosters = rosters.copy()
        T_teams, W = rosters.shape[0], len(self.weeks)
        weekly_rost = np.repeat(rosters[None], W, axis=0)   # roster in effect each week
        drafted = set(rosters[rosters >= 0].tolist())
        for sw in SWAP_WEEKS:
            if sw >= W:
                continue
            val = self.to_date_value(sw)
            fa = [p for p in np.argsort(-val) if p not in drafted]
            for team in rng.permutation(T_teams):
                for _ in range(MAX_SWAPS):
                    slots = [i for i in range(rosters.shape[1]) if rosters[team, i] >= 0]
                    worst = min(slots, key=lambda i: val[rosters[team, i]])
                    if fa and val[fa[0]] > val[rosters[team, worst]] + SWAP_MARGIN:
                        new = fa.pop(0)
                        drafted.add(new)
                        rosters[team, worst] = new
                    else:
                        break
            weekly_rost[sw:] = rosters
        idx = np.where(weekly_rost < 0, 0, weekly_rost)                    # [W, T, n]
        wi = np.arange(W)[:, None, None]
        tot = self.T[idx, wi]                                              # [W, T, n, S]
        out_inj = (self.N[idx, wi] == 0)
        stream = out_inj * self.G[idx, wi] * M.INJURY_RECOVERY
        tot = tot + stream[..., None] * self.repl_mu
        tot = np.where((weekly_rost < 0)[..., None], 0, tot).sum(2)       # [W, T, S]
        k = M.SK
        fg = tot[..., k["fgx"]] / np.maximum(tot[..., k["fga"]], 1)
        ft = tot[..., k["ftx"]] / np.maximum(tot[..., k["fta"]], 1)
        stats = np.stack([fg, ft] + [tot[..., k[s]] for s in ("tpm", "reb", "ast", "stl", "blk")]
                         + [-tot[..., k["tov"]], tot[..., k["pts"]]], -1)   # [W, T, 9], higher = better
        # cats[w, a, b] = categories a won vs b in week w (category ties count half)
        sa, sb = stats[:, :, None, :], stats[:, None, :, :]
        cats = (sa > sb).sum(-1) + 0.5 * (sa == sb).sum(-1)               # [W, T, T]
        opp = [o for o in range(T_teams) if o != me]
        wins, losses = (stats[:, me : me + 1] > stats[:, opp]).sum(-1), (stats[:, me : me + 1] < stats[:, opp]).sum(-1)
        res = (wins > losses) + 0.5 * (wins == losses)
        reg, po = res[~self.is_po].mean(), res[self.is_po].mean()
        champ = L.p_champ_real(cats, np.where(~self.is_po)[0], np.where(self.is_po)[0], me,
                               np.random.default_rng(12345), K=500)
        return {"champ": champ, "J": 0.5 * reg + 0.5 * po, "reg": reg, "po": po, "cats": wins[~self.is_po].mean()}


DEV_SEASONS = (2024, 2025, 2026)
HOLDOUT_SEASONS = (2022,)


def main(n_drafts=16, strategies=None, seasons=DEV_SEASONS, tag="dev"):
    """Draft + score; rosters saved to data/rosters_{tag}.npz so any metric can be re-scored later."""
    strategies = strategies or list(H.TOOL_VARIANTS) + list(H.BASELINES)
    rows, rosters = [], {}
    t0 = time.time()
    for t in seasons:
        rs = RealSeason(t)
        for seed in range(n_drafts):
            for strat in strategies:
                R = H.run_draft(t, strat, seed)
                rosters[f"{t}_{seed}_{strat}"] = R
                r = rs.score(R, me=11, rng=np.random.default_rng(seed))
                rows.append({"season": t, "seed": seed, "strategy": strat, **r})
            print(f"{t} draft {seed + 1}/{n_drafts}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(M.DATA / f"realtest_{tag}.csv", index=False)
            np.savez(M.DATA / f"rosters_{tag}.npz", **rosters)
    report(pd.DataFrame(rows))


def report(res: pd.DataFrame, ref: str = "tool"):
    for metric, label in (("champ", "P(championship)"), ("J", "weekly matchup win rate")):
        print(f"\nREAL seasons, slot 12 — {label}")
        piv = res.pivot_table(index="strategy", columns="season", values=metric)
        piv["mean"] = piv.mean(1)
        print(piv.sort_values("mean", ascending=False).round(3).to_string())
        print(f"paired difference vs '{ref}' (pooled over seasons; se from draft-to-draft variation):")
        p = res.pivot_table(index=["season", "seed"], columns="strategy", values=metric)
        if ref not in p:
            continue
        for s in p.columns:
            if s != ref:
                d = p[s] - p[ref]
                print(f"  {s:15} {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    holdout = "--holdout" in args
    tag = next((a.split("=", 1)[1] for a in args if a.startswith("--tag=")), "holdout" if holdout else "dev")
    args = [a for a in args if a != "--holdout" and not a.startswith("--tag=")]
    n = int(args[0]) if args else 16
    main(n, args[1:] or None, seasons=HOLDOUT_SEASONS if holdout else DEV_SEASONS, tag=tag)
