"""League dashboard data: category win chances, matchup matrix, season outcomes and rosters for all 12
teams from the live rosters in data/league_2027.json (ESPN sync). Writes data/dashboard.json."""
import json

import numpy as np

import inseason as I
import league as L
import model as M
import sim as S
from draft import LIVE_CONFIG

R, K, SEED = 1000, 64, 7
STAT_LABEL = {"pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TO", "tpm": "3PM"}


BASIS = {"stack_belief": "our model's projections (ESPN blended with player history, adjusted for market information)",
         "espn": "ESPN's own preseason projections, unadjusted"}


def main(my_team_id=13, proj=None, out_name="dashboard.json"):
    proj = proj or LIVE_CONFIG["proj"]
    M.set_kappa(LIVE_CONFIG["kappa"])
    season = I.inseason_season(M.get_season(M.LIVE_SEASON, proj), I.gamelogs())
    df = season.df
    lg = I.load_league()
    B = I.build(season, lg, I.gamelogs())                     # availability: participation, injuries, IR
    rost, tids, names, unmodeled = B["rost"], B["tids"], B["names"], B["unmodeled"]
    ev = M.Evaluator(season, proj_unc=True)
    truth = ev.draw_truth(R, np.random.default_rng(SEED))
    RR = np.broadcast_to(rost[None], (R, 12, M.ROSTER)).copy()
    mS, vS = ev.truth_team_stats(truth, RR)                                    # [R, T, 4, S] typical week
    pc = M.Evaluator.cat_probs(mS[:, :, None, 0], vS[:, :, None, 0], mS[:, None, :, 0], vS[:, None, :, 0])
    pc = pc.mean(0)                                                            # [T, T, 9]
    off = ~np.eye(12, dtype=bool)
    cat_vs_league = np.stack([pc[t][off[t]].mean(0) for t in range(12)])      # [T, 9]
    pw = M.Evaluator.matchup_win(pc)                                           # [T, T]
    np.fill_diagonal(pw, np.nan)
    pcp = M.Evaluator.cat_probs(mS[:, :, None, 1:], vS[:, :, None, 1:], mS[:, None, :, 1:], vS[:, None, :, 1:])
    pwp = M.Evaluator.matchup_win(pcp.mean(0))                                 # [T, T, 3] playoff weeks
    po_win = np.array([pwp[t][off[t]].mean() for t in range(12)])
    g = season.games_matrix                                                    # [P, 4]
    po_games = np.array([g[rost[t][rost[t] >= 0], 1:].sum() for t in range(12)])
    slot = {tid: s for s, tid in enumerate(tids)}
    pairs, tot0, _, sched = I.state(lg, tids)
    done = set(lg.get("completed_weeks", []))
    weeks_left = B["weeks_left"]
    FR = np.broadcast_to(I.full(rost, B["ir"])[None], (R, 12, M.ROSTER + I.N_IR)).copy()
    mW, vW = ev.truth_team_stats(truth, FR, g=B["Geff"])      # each matchup's own games x availability
    import live as LV
    week0 = LV.week0_for(pairs, weeks_left) if proj == LIVE_CONFIG["proj"] else None
    out = L.season_outcomes(mW, vW, np.random.default_rng(SEED + 1), K=K, pairs=pairs, tot0=tot0, per_week=True,
                            week0_pc=week0)
    week_p = {}                                          # (period, team slot) -> P(win that matchup, its own games)
    for k, (wkid, pr) in enumerate(zip(weeks_left, pairs)):
        for h, a_ in pr:
            pcw = M.Evaluator.cat_probs(mW[:, h, k], vW[:, h, k], mW[:, a_, k], vW[:, a_, k]).mean(0)
            ph = float(M.Evaluator.matchup_win(pcw))
            week_p[(wkid, h)], week_p[(wkid, a_)] = ph, 1 - ph
    if week0 is not None:                                # the week in progress: live odds (live.py, model L3)
        for (h, a_), pcw in zip(pairs[0], week0):
            ph = LV.win_week(np.asarray(pcw))
            week_p[(weeks_left[0], h)], week_p[(weeks_left[0], a_)] = ph, 1 - ph
    Gp = B["Geff"]
    team_games = {(wkid, t): int(round(Gp[rost[t][rost[t] >= 0], k].sum())) for k, wkid in enumerate(weeks_left) for t in range(12)}
    wk = mS[:, :, 0].mean(0)                                                   # [T, S] avg regular week
    totals = {lab: wk[:, M.SK[k]] for k, lab in STAT_LABEL.items()}
    totals["FG%"] = wk[:, M.SK["fgx"]] / wk[:, M.SK["fga"]] + M.LG_FG
    totals["FT%"] = wk[:, M.SK["ftx"]] / wk[:, M.SK["fta"]] + M.LG_FT
    val, _ = M.static_value(df)
    z = M.static_value.last_by_cat
    teams = []
    for s in range(12):
        players = []
        for p in rost[s]:
            if p < 0:
                continue
            players.append({"name": df.name[p], "nba": df.team[p], "value": round(float(val[p]), 2),
                            "gp": round(float(df.gp[p]), 0),
                            "line": {lab: round(float(df[k][p]), 1) for k, lab in STAT_LABEL.items()} |
                                    {"FG%": round(float(df.fgm[p] / max(df.fga[p], .1)), 3),
                                     "FT%": round(float(df.ftm[p] / max(df.fta[p], .1)), 3)},
                            "z": {c: round(float(z[c][p]), 2) for c in M.CATS}})
        players.sort(key=lambda x: -x["value"])
        teams.append({"id": tids[s], "name": names[s], "slot": s + 1, "me": tids[s] == my_team_id,
                      "cat_win": {c: round(float(cat_vs_league[s, i]), 3) for i, c in enumerate(M.CATS)},
                      "cats_per_week": round(float(cat_vs_league[s].sum()), 2),
                      "p_win_week_avg": round(float(np.nanmean(pw[s])), 3),
                      "p_win_playoff_week": round(float(po_win[s]), 3), "playoff_games": int(po_games[s]),
                      "weekly_totals": {k: round(float(v[s]), 3 if "%" in k else 1) for k, v in totals.items()},
                      **{k: round(float(v[s]), 4) for k, v in out.items()},
                      "schedule": [{"week": w, "opp": slot[a if h == tids[s] else h],
                                    "home": h == tids[s],
                                    "p_win": round(week_p.get((w, s), float(pw[s, slot[a if h == tids[s] else h]])), 3),
                                    "games": team_games.get((w, s)),
                                    "opp_games": team_games.get((w, slot[a if h == tids[s] else h]))}
                                   for w, h, a in sched if tids[s] in (h, a)],
                      "players": players})
    data = {"generated": str(np.datetime64("today")), "league": "The Cool Guys", "cats": M.CATS,
            "teams": teams, "matchup": [[None if np.isnan(x) else round(float(x), 3) for x in row] for row in pw],
            "completed_weeks": sorted(done), "unmodeled": unmodeled,
            "notes": {"rollouts": R, "seasons_per_rollout": K, "kappa": LIVE_CONFIG["kappa"], "proj": proj,
                      "basis": BASIS.get(proj, proj)}}
    (M.DATA / out_name).write_text(json.dumps(data))
    for t in sorted(teams, key=lambda t: -t["exp_dollars"]):
        print(f"{t['name'][:26]:26} E$ {t['exp_dollars']:6.2f}  title {t['p_1st']:.3f}  playoffs {t['p_playoffs']:.3f}  "
              f"cats/wk {t['cats_per_week']:.2f}  P(win wk) {t['p_win_week_avg']:.3f}  playoff wk {t['p_win_playoff_week']:.3f}  po games {t['playoff_games']}")


if __name__ == "__main__":
    import sys
    if "--espn" in sys.argv:
        main(proj="espn", out_name="dashboard_espn.json")
    else:
        main()
