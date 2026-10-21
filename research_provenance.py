"""Data estimates for parameters that were previously assumed (DECISIONS.md, provenance review)."""
import json

import numpy as np
import pandas as pd

import model as M
import projections as Pj

LAG, MIN_SPELL = 2, 4


def injury_recovery():
    covered = missed = 0
    spells_all = []
    for t in (2024, 2025, 2026):
        g = pd.read_csv(M.DATA / f"gamelogs_{t}.csv")
        g = g[g["min"] > 0]
        sch = pd.read_csv(M.DATA / f"schedule_{t}.csv")
        team_periods = sch.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
        team = Pj.files()[t + 1 if t == 2026 else t]["pro_team_id"]
        stats = g.groupby("id").agg(n=("pts", "size"), mins=("min", "mean"))
        rot = stats[(stats.n >= 20) & (stats.mins >= 20)].index
        for pid in rot:
            tm = team.get(pid)
            if pd.isna(tm) or int(tm) not in team_periods.index:
                continue
            tp = team_periods[int(tm)]
            played = set(g.loc[g.id == pid, "scoring_period"])
            if not played.issubset(set(tp)):
                continue  # changed teams mid-season: schedule attribution unreliable
            run = 0
            for p in tp:
                if p in played:
                    if run:
                        spells_all.append(run)
                    run = 0
                else:
                    run += 1
            if run:
                spells_all.append(run)
    s = np.array(spells_all)
    missed = s.sum()
    covered = np.where(s >= MIN_SPELL, np.maximum(s - LAG, 0), 0).sum()
    return covered / missed, len(s), missed, np.mean(s >= MIN_SPELL)


if __name__ == "__main__":
    phi, n_spells, n_missed, share_long = injury_recovery()
    K = json.loads((M.DATA / "inseason_K.json").read_text())
    k_med = float(np.median(list(K.values())))
    s27 = M.get_season(2027, "stack")
    gpw = float(s27.wg[s27.reg_weeks].to_numpy().mean())
    learn = k_med / gpw / 22
    rho = float(np.mean([0.299, 0.192]))  # research_teammates.py, 2024-25 and 2025-26
    print(f"injury recovery phi = {phi:.3f}  ({n_spells} absence spells, {n_missed} missed games; "
          f"{share_long:.0%} of spells are >= {MIN_SPELL} games)  [model uses 0.5; picks unchanged over 0.3-0.7]")
    print(f"bust identification: median K = {k_med:.1f} games, {gpw:.2f} games/week -> {k_med / gpw:.1f} weeks "
          f"-> LEARN_FRAC = {learn:.3f}  [model uses 0.20; picks unchanged over 0.10-0.35]")
    print(f"teammate games correlation rho = {rho:.3f}  [model uses 0.25]")
    print(f"games per team-week (2026-27 full regular-season weeks) = {gpw:.2f}  [G-score uses 3.5]")
