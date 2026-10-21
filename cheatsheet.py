"""Pre-draft cheat sheet data: rankings, value-vs-market gaps, and simulated drafts from our slot.

    python3 cheatsheet.py [n_drafts]      -> data/cheatsheet.json
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

import model as M
import sim as S


def simulate(season, n_drafts, my_slot=12, n_roll=200):
    d0 = S.Draft(season, my_slot=my_slot)
    my_picks = d0.my_next_picks(0)
    avail_at = np.zeros((len(my_picks), d0.P))
    taken_by_me = Counter()
    by_round = [Counter() for _ in my_picks]
    cat_probs = []
    for seed in range(n_drafts):
        d = S.Draft(season, my_slot=my_slot, objective="champ")
        boards = d.boards(np.random.default_rng(50_000 + seed), 1)[0]
        for k in range(d.n_picks):
            t = d.order[k]
            avail = d.available()
            if t == d.me:
                j = my_picks.index(k)
                avail_at[j] += avail
                res = d.evaluate(d.candidate_pool(14), n_roll=n_roll, seed=seed * 100 + k)
                p = int(res["idx"].iloc[0])
                by_round[j][p] += 1
                taken_by_me[p] += 1
            else:
                p = int(np.where(avail, boards[t], np.inf).argmin())
            d.picks.append(p)
        R = np.full((M.N_TEAMS, M.ROSTER), -1)
        for t in range(M.N_TEAMS):
            r = d.roster(t)
            R[t, : len(r)] = r
        m, v = d.ev.team_stats(R)
        opp = [t for t in range(M.N_TEAMS) if t != d.me]
        cat_probs.append(d.ev.cat_probs(m[d.me : d.me + 1, 0], v[d.me : d.me + 1, 0], m[opp, 0], v[opp, 0]).mean(0))
        print(f"sim {seed + 1}/{n_drafts}", flush=True)
    return my_picks, avail_at / n_drafts, by_round, taken_by_me, np.array(cat_probs)


def main(n_drafts=60):
    from draft import LIVE_CONFIG
    M.set_kappa(LIVE_CONFIG["kappa"])
    season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
    df = season.df.copy()
    val, _ = M.static_value(df)
    z = pd.DataFrame(M.static_value.last_by_cat)
    df = pd.concat([df, z], axis=1)
    df["value"] = val
    df["vrank"] = df["value"].rank(ascending=False).astype(int)
    my_picks, avail_at, by_round, taken, cp = simulate(season, n_drafts)

    top = df.sort_values("value", ascending=False).head(220).copy()
    players = []
    for i, r in top.iterrows():
        players.append({
            "name": r["name"], "team": r["team"], "pos": r["elig"], "injury": r["injury"],
            "adp": None if r["adp"] >= 135 else round(float(r["adp"]), 1),
            "espn_rank": None if pd.isna(r["espn_rank"]) else int(r["espn_rank"]),
            "vrank": int(r["vrank"]), "value": round(float(r["value"]), 2),
            "espn_gp": int(r["espn_gp"]), "gp": round(float(r["gp"]), 0), "min": round(float(r["espn_min"]), 1),
            "po_mult": round(float(r["po_mult"]), 2),
            "cats": {c: round(float(r[c]), 2) for c in M.CATS},
            "line": {s: round(float(r[s]), 1) for s in ["pts", "reb", "ast", "stl", "blk", "tov", "tpm"]}
                    | {"fg": round(float(r["fgm"] / max(r["fga"], 0.1)), 3), "ft": round(float(r["ftm"] / max(r["fta"], 0.1)), 3)},
            "avail": [round(float(avail_at[j, i]), 2) for j in range(len(my_picks))],
            "mine": round(taken[i] / n_drafts, 2),
        })
    rounds = []
    for j, k in enumerate(my_picks):
        common = by_round[j].most_common(6)
        rounds.append({"pick": k + 1, "round": k // M.N_TEAMS + 1,
                       "picks": [{"name": df.at[p, "name"], "freq": round(c / n_drafts, 2)} for p, c in common]})
    out = {
        "my_picks": [k + 1 for k in my_picks],
        "rounds": rounds,
        "players": players,
        "cat_profile": {c: round(float(x), 3) for c, x in zip(M.CATS, cp.mean(0))},
        "punt_freq": {c: round(float((cp[:, i] < 0.3).mean()), 2) for i, c in enumerate(M.CATS)},
        "n_drafts": n_drafts,
    }
    (M.DATA / "cheatsheet.json").write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ["cat_profile", "punt_freq"]}, indent=1))
    for r in rounds:
        print(r["pick"], ", ".join(f"{p['name']} {p['freq']:.0%}" for p in r["picks"]))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
