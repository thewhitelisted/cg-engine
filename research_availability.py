"""Candidate A1 (DECISIONS.md 2026-10-20): in-season availability. Estimates the participation blend weight
K_gp and the absence-spell survival curve, and validates V0 / V1 / V2 leave-one-season-out.

    python3 research_availability.py        # prints the validation table, writes data/availability.json
"""
import json

import numpy as np
import pandas as pd

import model as M
import projections as Pj

SEASONS = (2024, 2025, 2026)
CHECKPOINTS = (10, 20, 30, 40, 50, 60)
MIN_PROJ_MIN = 20
ABSENT_M = 2


def season_rows(t):
    """Per player on one team all season: preseason share and the played/missed sequence over team games."""
    g = pd.read_csv(M.DATA / f"gamelogs_{t}.csv")
    g = g[g["min"] > 0]
    sch = pd.read_csv(M.DATA / f"schedule_{t}.csv")
    team_periods = sch.groupby("pro_team_id")["scoring_period"].apply(lambda s: np.sort(s.unique()))
    team = Pj.files()[t + 1 if t == 2026 else t]["pro_team_id"]
    df = M.get_season(t, "stack").df
    out = []
    played_by = g.groupby("id")["scoring_period"].apply(set)
    for _, r in df[df["espn_min"] >= MIN_PROJ_MIN].iterrows():
        pid = int(r["id"])
        tm = team.get(pid)
        if pd.isna(tm) or int(tm) not in team_periods.index:
            continue
        tp = team_periods[int(tm)]
        played = played_by.get(pid, set())
        if not played.issubset(set(tp)):
            continue                      # changed teams mid-season
        out.append({"id": pid, "season": t, "prior": float(r["avail"]),
                    "seq": np.array([p in played for p in tp], bool)})
    return out


def spells(rows):
    """(length, completed) for every run of consecutive missed team games."""
    s = []
    for r in rows:
        run = 0
        for x in r["seq"]:
            if x:
                if run:
                    s.append((run, True))
                run = 0
            else:
                run += 1
        if run:
            s.append((run, False))        # still out at season end: censored
    return s


def km_survival(sp, max_len=90):
    """Kaplan-Meier S(m) = P(spell lasts at least m games), m = 0..max_len (S(0) = S(1) = 1)."""
    L = np.array([l for l, _ in sp])
    done = np.array([c for _, c in sp])
    S = np.ones(max_len + 2)
    surv = 1.0
    for m in range(1, max_len + 1):
        at_risk = (L >= m).sum()
        ended = ((L == m) & done).sum()
        S[m] = surv                              # P(length >= m)
        if at_risk:
            surv *= 1 - ended / at_risk
    S[max_len + 1] = surv
    return S


def trailing_run(seq):
    n = 0
    for x in seq[::-1]:
        if x:
            break
        n += 1
    return n


def predict(r, c, K, S, variant):
    seq = r["seq"]
    rem = len(seq) - c
    if variant == "V0":
        return r["prior"]
    if variant == "V1":
        return (K * r["prior"] + seq[:c].sum()) / (K + c)
    m = trailing_run(seq[:c])
    if m < ABSENT_M:
        return (K * r["prior"] + seq[:c].sum()) / (K + c)
    c0 = c - m                                   # share while not injured: data before the current absence
    base = (K * r["prior"] + seq[:c0].sum()) / (K + c0)
    j = np.arange(1, rem + 1)
    idx = np.minimum(m + j, len(S) - 1)
    p_out = np.where(S[m] > 0, S[idx] / max(S[m], 1e-12), 1.0)
    return float(np.mean((1 - p_out) * base))


def fit_K(rows, grid=np.r_[np.arange(1, 40), np.arange(40, 401, 5)]):
    best, bk = np.inf, None
    for K in grid:
        e = [(predict(r, c, K, None, "V1") - r["seq"][c:].mean()) ** 2 for r in rows for c in CHECKPOINTS if c < len(r["seq"])]
        if np.mean(e) < best:
            best, bk = np.mean(e), K
    return float(bk)


def main():
    data = {t: season_rows(t) for t in SEASONS}
    res = []
    for hold in SEASONS:
        train = [r for t in SEASONS if t != hold for r in data[t]]
        K = fit_K(train)
        S = km_survival(spells(train))
        for r in data[hold]:
            for c in CHECKPOINTS:
                if c >= len(r["seq"]):
                    continue
                y = r["seq"][c:].mean()
                m = trailing_run(r["seq"][:c])
                row = {"season": hold, "c": c, "absent": m >= ABSENT_M, "K": K}
                for v in ("V0", "V1", "V2"):
                    row[v] = (predict(r, c, K, S, v) - y) ** 2
                res.append(row)
    R = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print("n player-checkpoints:", len(R), " currently absent:", int(R.absent.sum()))
    print("\nMSE by held-out season (all players):")
    print(R.groupby("season")[["V0", "V1", "V2"]].mean().round(5).to_string())
    print("\nMSE by held-out season (currently absent, m >= 2):")
    print(R[R.absent].groupby("season")[["V0", "V1", "V2"]].mean().round(5).to_string())
    print("\npooled all:", R[["V0", "V1", "V2"]].mean().round(5).to_dict())
    print("pooled absent:", R[R.absent][["V0", "V1", "V2"]].mean().round(5).to_dict())
    print("K_gp by fold:", R.groupby("season").K.first().to_dict())
    # final estimates on all three seasons, for live use
    allrows = [r for t in SEASONS for r in data[t]]
    out = {"K_gp": fit_K(allrows), "survival": km_survival(spells(allrows)).round(5).tolist(),
           "absent_m": ABSENT_M, "n_spells": len(spells(allrows))}
    (M.DATA / "availability.json").write_text(json.dumps(out))
    print("final K_gp", out["K_gp"], "spells", out["n_spells"], "S(4..12):", np.round(out["survival"][4:13], 3).tolist())


if __name__ == "__main__":
    main()
