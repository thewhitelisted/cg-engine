"""Playoff picture data for the Playoff Picture artifact: seeds, round-by-round odds, likely opponents,
playoff-week games and strength, from the in-season state on the real remaining schedule.
Writes data/playoffs.json; build with build_playoffs.py."""
import json

import numpy as np

import inseason as I
import league as L
import model as M
from draft import LIVE_CONFIG

R, K, SEED = 1000, 64, 7          # same draws as dashboard_data: identical odds on every page
PLAYOFF_WEEKS = ["Mar 8–14", "Mar 15–21", "Mar 22–28"]


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main(my_team_id=13):
    M.set_kappa(LIVE_CONFIG["kappa"])
    season = I.inseason_season(M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"]), I.gamelogs())
    lg = I.load_league()
    B = I.build(season, lg, I.gamelogs())                     # availability: participation, injuries, IR
    rost, tids, names, unmodeled = B["rost"], B["tids"], B["names"], B["unmodeled"]
    pairs, tot0, weeks_left = B["pairs"], B["tot0"], B["weeks_left"]
    G = B["Geff"]
    ev = M.Evaluator(season, proj_unc=True)
    truth = ev.draw_truth(R, np.random.default_rng(SEED))
    RR = np.broadcast_to(I.full(rost, B["ir"])[None], (R, 12, M.ROSTER + I.N_IR)).copy()
    mW, vW = ev.truth_team_stats(truth, RR, g=G)
    import live as LV
    out = L.playoff_detail(mW, vW, np.random.default_rng(SEED + 1), K=K, pairs=pairs, tot0=tot0, per_week=True,
                           week0_pc=LV.week0_for(pairs, weeks_left), h2h0=B["h2h0"])
    fr = I.full(rost, B["ir"])
    po_games = np.array([G[fr[t][fr[t] >= 0], -3:].sum(0) for t in range(12)])               # [T, 3]
    pw = out["pw_playoff"]                                                                      # [T, T, 3]
    off = ~np.eye(12, dtype=bool)
    po_week_win = np.array([[pw[t, off[t], k].mean() for k in range(3)] for t in range(12)])
    po_cats = M.Evaluator.cat_probs(mW[:, :, None, -3:], vW[:, :, None, -3:], mW[:, None, :, -3:], vW[:, None, :, -3:]).mean(0)
    po_cat_win = np.array([po_cats[t][off[t]].mean((0, 1)) for t in range(12)])                # [T, 9] over weeks
    teams = []
    for t in range(12):
        def opp_list(rnd):
            f, b = out[f"face_{rnd}"][t], out[f"beat_{rnd}"][t]
            return [{"opp": int(o), "p_face": r(f[o]), "p_beat": r(b[o])} for o in np.argsort(-f) if f[o] > 0.002]
        teams.append({"id": tids[t], "name": names[t], "me": tids[t] == my_team_id,
                      "p_seed": [r(x) for x in out["p_seed"][t]], "p_playoffs": r(out["p_playoffs"][t]),
                      "p_semis": r(out["p_semis"][t]), "p_final": r(out["p_final"][t]), "p_1st": r(out["p_1st"][t]),
                      "p_2nd": r(out["p_2nd"][t]), "p_3rd": r(out["p_3rd"][t]), "exp_dollars": r(out["exp_dollars"][t], 2),
                      "exp_wins": r(out["exp_wins"][t], 1), "exp_cats": r(out["exp_cats"][t], 1),
                      "title_by_seed": [r(x) for x in out["title_by_seed"][t]],
                      "po_games": [int(round(x)) for x in po_games[t]], "po_week_win": [r(x, 3) for x in po_week_win[t]],
                      "po_cat_win": {c: r(po_cat_win[t, i], 3) for i, c in enumerate(M.CATS)},
                      "qf": opp_list("qf"), "sf": opp_list("sf"), "final": opp_list("final")})
    data = {"generated": str(np.datetime64("today")), "cats": M.CATS, "playoff_weeks": PLAYOFF_WEEKS,
            "completed_weeks": sorted(lg.get("completed_weeks", [])), "teams": teams,
            "pw_playoff": [[[r(pw[i, j, k], 3) if i != j else None for k in range(3)] for j in range(12)] for i in range(12)],
            "notes": {"rollouts": R, "seasons_per_rollout": K, "proj": LIVE_CONFIG["proj"], "unmodeled": unmodeled}}
    (M.DATA / "playoffs.json").write_text(json.dumps(data))
    for t in sorted(teams, key=lambda x: -x["p_1st"]):
        print(f"{t['name'][:24]:24} playoffs {t['p_playoffs']:.3f} semis {t['p_semis']:.3f} final {t['p_final']:.3f} "
              f"title {t['p_1st']:.3f}  E$ {t['exp_dollars']:6.2f}  po games {t['po_games']}  po wk win {t['po_week_win']}")
    return data


if __name__ == "__main__":
    main()
