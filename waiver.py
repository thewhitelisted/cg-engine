"""In-season pickup / streaming advisor.

Per-game rates are the preseason projection updated with season-to-date stats:
    rate = (K * projection + n * to_date_mean) / (K + n)      K per stat from research_inseason.py
Your roster: data/my_roster.json (list of names) if present, else your picks in data/draft_state.json.
Opponents: the other 11 rosters recorded at the draft. Players already on a drafted roster are excluded
from suggestions; check that a suggested player is actually a free agent in ESPN before adding.

    python3 waiver.py                # best add/drop swaps for this week and rest of season
    python3 waiver.py --week 5       # a specific week
"""
import json
import sys

import numpy as np
import pandas as pd

import fetch
import model as M
import track
from draft import LIVE_CONFIG

M.set_kappa(LIVE_CONFIG["kappa"])
K_GAMES = json.loads((M.DATA / "inseason_K.json").read_text())


def updated_rates(season, g=None):
    """Blend each projection with season-to-date per-game stats. g: game logs (fetched when None)."""
    df = season.df
    mu = df[M.STATK].to_numpy(float).copy()
    if g is None:
        try:
            g = fetch.fetch_gamelogs(M.LIVE_SEASON, limit=700)
        except Exception:
            return mu, np.zeros(len(df))
    g = g[g["min"] > 0]
    if g.empty:
        return mu, np.zeros(len(df))
    td = g.groupby("id")[["pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fga", "fta", "fgm", "ftm"]].agg(["sum", "count"])
    n = df["id"].map(td[("pts", "count")]).fillna(0).to_numpy()
    for k in ("pts", "reb", "ast", "stl", "blk", "tov", "tpm", "fga", "fta"):
        tot = df["id"].map(td[(k, "sum")]).fillna(0).to_numpy()
        Kk = K_GAMES[k]
        mu[:, M.SK[k]] = (Kk * mu[:, M.SK[k]] + tot) / (Kk + n)
    # FG%/FT%: shrink the percentage toward the projection with the attempt-volume K
    for x, a, made, lg in (("fgx", "fga", "fgm", M.LG_FG), ("ftx", "fta", "ftm", M.LG_FT)):
        proj_pct = (df[made] / df[a].clip(lower=0.1)).to_numpy()
        att = df["id"].map(td[(a, "sum")]).fillna(0).to_numpy()
        mk = df["id"].map(td[(made, "sum")]).fillna(0).to_numpy()
        prior_att = K_GAMES[a] * df[a].to_numpy()
        pct = (prior_att * proj_pct + mk) / np.maximum(prior_att + att, 1e-6)
        mu[:, M.SK[x]] = mu[:, M.SK[a]] * (pct - lg)
    return mu, n


def main(week=None):
    season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
    df = season.df
    today = (pd.Timestamp.today().normalize() - season.season_start).days // 7
    week = today if week is None else week
    ev = M.Evaluator(season, proj_unc=True)
    mu, n = updated_rates(season)
    fresh = fetch.flatten(fetch.fetch_players(M.LIVE_SEASON, limit=600), M.LIVE_SEASON).set_index("id")
    out_now = df["id"].map(fresh["injury"]).isin(["OUT", "INJURY_RESERVE", "SUSPENSION"]).to_numpy()
    R, me = track.rosters(season)
    rostered = set(R[R >= 0].tolist())
    mine = [p for p in R[me] if p >= 0]

    def J_for(rosters, g, avail):
        ev.wmu, ev.wvar, ev.own = ev.weekly(mu, avail, g)
        m, v = ev.team_stats(rosters)                                  # [N, W, S]
        opp = [t for t in range(M.N_TEAMS) if t != me]
        mo, vo = ev.team_stats(R[opp])                                 # [11, W, S]
        p = M.Evaluator.cat_probs(m[:, None], v[:, None], mo[None], vo[None])
        return M.Evaluator.matchup_win(p).mean(1)                      # [N, W]

    g_week = df["pro_team_id"].map(season.wg[week] if week in season.wg else 0).fillna(0).to_numpy()
    g_week = np.repeat(g_week[:, None], 4, 1)
    avail_week = np.where(out_now, 0.0, ev.avail)
    rest_weeks = [w for w in season.wg.columns if w >= week]
    g_rest = np.repeat(df["pro_team_id"].map(season.wg[rest_weeks].mean(1)).fillna(0).to_numpy()[:, None], 4, 1)

    cands = [p for p in np.argsort(-M.value_from(mu, ev.avail, ev.vscale)) if p not in rostered][:60]
    base = np.array(mine + [-1] * (M.ROSTER + 2 - len(mine)))
    trials, labels = [base], [(None, None)]
    for c in cands:
        for i, d in enumerate(mine):
            r = base.copy()
            r[i] = c
            trials.append(r)
            labels.append((c, d))
    T = np.array(trials)
    jw = J_for(T, g_week, avail_week)[:, 0]
    jr = J_for(T, g_rest, ev.avail)[:, 0]
    res = pd.DataFrame({"add": [df.name[c] if c is not None else "" for c, _ in labels],
                        "drop": [df.name[d] if d is not None else "" for _, d in labels],
                        "this_week": jw - jw[0], "rest_of_season": jr - jr[0]}).iloc[1:]
    res["add_games_this_week"] = [int(g_week[c, 0]) for c, _ in labels[1:]]
    pd.set_option("display.width", 200)
    print(f"Week {week}. Your P(win) vs your 11 opponents: this week {jw[0]:.3f}, typical rest-of-season week {jr[0]:.3f}")
    print(f"Games played so far feeding the update: median {np.median(n[mine]):.0f} for your players.\n")
    print("Best moves for THIS WEEK (streaming):")
    print(res.sort_values("this_week", ascending=False).head(8).round(4).to_string(index=False))
    print("\nBest moves for the REST OF THE SEASON:")
    print(res.sort_values("rest_of_season", ascending=False).head(8).round(4).to_string(index=False))
    hurt = [df.name[p] for p in mine if out_now[p]]
    if hurt:
        print(f"\nOUT on your roster: {', '.join(hurt)}. Move them to IR (2 slots) and add from the list above "
              "instead of dropping them, unless 'rest_of_season' for the drop is positive.")
    print("\nCheck each 'add' is a free agent in ESPN; you have 3 adds per matchup.")


if __name__ == "__main__":
    wk = int(sys.argv[sys.argv.index("--week") + 1]) if "--week" in sys.argv else None
    main(wk)
