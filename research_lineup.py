"""Candidate 5: games lost to the daily 10-starter cap (with position eligibility) on the 2026-27 schedule.

For drafted rosters, each day: players whose team plays and who are available (Bernoulli(avail));
fill PG, SG, SF, PF, C, G, F, UTIL x3 by maximum bipartite matching; unfilled-but-playing = lost game.
Model it only if losses exceed 3% and differ across roster types (DECISIONS.md).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import harness as H
import model as M

SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL"]


def eligible(elig: str, slot: str) -> bool:
    pos = set(str(elig).split("/"))
    if slot == "UTIL":
        return True
    if slot == "G":
        return bool(pos & {"PG", "SG"})
    if slot == "F":
        return bool(pos & {"SF", "PF"})
    return slot in pos


def lost_fraction(roster, df, sched, rng, n_sims=3):
    elig = df.loc[roster, "elig"].tolist()
    team = df.loc[roster, "pro_team_id"].to_numpy()
    avail = df.loc[roster, "avail"].to_numpy()
    E = np.array([[eligible(e, s) for s in SLOTS] for e in elig], float)
    lost = played = 0
    for _ in range(n_sims):
        for _, day in sched.groupby("date"):
            plays = np.isin(team, day["pro_team_id"].to_numpy()) & (rng.random(len(roster)) < avail)
            n = plays.sum()
            if n == 0:
                continue
            played += n
            r, c = linear_sum_assignment(-E[plays])
            lost += n - E[plays][r, c].sum()
    return lost / max(played, 1)


if __name__ == "__main__":
    s = M.get_season(2027)
    df = s.df
    sched = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    sched = sched[sched.date < s.playoff_dates[1]]
    rng = np.random.default_rng(0)
    rows = []
    for strat in ("tool", "adp", "value"):
        for seed in range(6):
            R = H.run_draft(2027, strat, seed, n_roll=60)
            for team in range(M.N_TEAMS):
                rows.append({"strategy": strat, "team": team, "mine": team == 11,
                             "lost": lost_fraction(R[team], df, sched, rng, n_sims=2)})
        print(strat, "done", flush=True)
    r = pd.DataFrame(rows)
    print(f"\nshare of player-games lost to the 10-starter cap: mean {r.lost.mean():.3%}, "
          f"p10 {r.lost.quantile(.1):.3%}, p90 {r.lost.quantile(.9):.3%}")
    print(r[r.mine].groupby("strategy").lost.agg(["mean", "std"]).round(4).to_string())
