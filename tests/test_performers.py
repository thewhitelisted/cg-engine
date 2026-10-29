"""Best and worst performers on the league page: a player's games this week, as far as they counted for his fantasy
team (a starting slot at the end of the day in lineup_log.csv), against the same games played to the model's line,
measured by how much they moved his team's chance to win the week.

    python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import league_page as LP  # noqa: E402
import live as LV  # noqa: E402
import model as M  # noqa: E402

D26, D27 = pd.Timestamp("2026-10-26"), pd.Timestamp("2026-10-27")
DAY_OF = {7: D26, 8: D27}
LG = {"teams": [{"id": 1, "name": "One", "roster": [[10, "A", 0, "ACTIVE"], [11, "B", 12, "ACTIVE"], [12, "C", 11, "ACTIVE"]]},
                {"id": 2, "name": "Two", "roster": [[20, "D", 0, "ACTIVE"]]}]}
STATS = dict(pts=20.0, reb=5.0, ast=4.0, stl=1.0, blk=1.0, tpm=2.0, tov=2.0, fgm=8.0, fga=16.0, ftm=2.0, fta=2.0)
# Oct 27 03:00 EDT = 07:00 UTC: the end of period 7
LOG = pd.DataFrame([("2026-10-26T12:00:00Z", 7, 1, 10, 0), ("2026-10-26T12:00:00Z", 7, 1, 11, 12),
                    ("2026-10-26T12:00:00Z", 7, 1, 12, 11), ("2026-10-26T12:00:00Z", 7, 1, 13, 0),
                    ("2026-10-26T12:00:00Z", 7, 2, 20, 0),
                    ("2026-10-26T23:00:00Z", 7, 1, 12, 12),      # benched before his game: doesn't count
                    ("2026-10-27T08:00:00Z", 8, 1, 11, 0),       # started the next day
                    ("2026-10-27T08:00:00Z", 8, 1, 10, 12)],
                   columns=["time_utc", "scoring_period", "team_id", "player_id", "slot"])


def logs(rows):
    return pd.DataFrame([dict(STATS, id=i, name=f"P{i}", scoring_period=p, min=mins) for i, p, mins in rows])


class CountedGamesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "lineup_log.csv"
        LOG.to_csv(self.log, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_starts_for_the_current_team(self):
        g = logs([(10, 7, 30), (11, 7, 30), (12, 7, 30), (13, 7, 30), (20, 7, 30), (10, 8, 30), (11, 8, 30), (20, 8, 0)])
        cg = LP.counted_games(LG, g, {7, 8}, DAY_OF, self.log)
        got = sorted(zip(cg["id"], cg["scoring_period"], cg["team_id"]))
        # 10 started p7 (bench p8); 11 bench p7, started p8; 12 benched before the cutoff; 13 is on no roster now;
        # 20 started p7 and had no minutes on p8
        self.assertEqual(got, [(10, 7, 1), (11, 8, 1), (20, 7, 2)])

    def test_no_log(self):
        g = logs([(10, 7, 30)])
        self.assertEqual(len(LP.counted_games(LG, g, {7}, DAY_OF, Path(self.tmp.name) / "none.csv")), 0)


class PerformersTest(unittest.TestCase):
    """Two matchups: 1 vs 2 is close in every category, 3 vs 4 is decided. Player 10 (team 1, home) and 20 (team 2,
    away) beat their lines; 30 (team 3) beats his by more, but in a decided matchup."""

    def lv(self):
        ids = [10, 20, 30]
        df = pd.DataFrame({"id": ids, "pro_team_id": [1, 2, 3]})
        rate = np.zeros(len(M.STATK))
        for s in ("pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fga", "fta"):
            rate[M.SK[s]] = STATS[s] / 2                   # his line was twice his rates
        rate[M.SK["fgx"]] = (0.5 - M.LG_FG) * STATS["fga"] / 2      # a .500 shooter, as he shot
        rate[M.SK["ftx"]] = (1.0 - M.LG_FT) * STATS["fta"] / 2
        for k, s in enumerate(M.STATK):
            df[s] = rate[k]
        lv = mock.Mock()
        lv.season.df = df
        lv.lg = {"teams": [{"id": t, "name": f"T{t}", "roster": [[p, f"P{p}", 0, "ACTIVE"]]} for t, p in ((1, 10), (2, 20), (3, 30), (4, 40))]}
        lv.live = {"matchups": [{"home": 1, "away": 2}, {"home": 3, "away": 4}]}
        sd = np.full(len(M.STATK), 30.0)
        even = np.full(len(M.STATK), 100.0)
        lead = even + 1000                                 # team 3 far ahead in every category
        for s, x in (("fga", 100.0), ("fta", 100.0), ("fgx", 500.0), ("ftx", 500.0), ("tov", 0.0)):
            lead[M.SK[s]] = x
        trail = even.copy()
        for s, x in (("fgx", -500.0), ("ftx", -500.0)):
            trail[M.SK[s]] = x
        decided = (lead, sd ** 2, trail, sd ** 2)
        lv.dists = lambda h, a: ((even, sd ** 2, even.copy(), sd ** 2) if h == 1 else decided) + (None,)
        return lv

    def test_effects(self):
        g = logs([(10, 7, 30), (20, 7, 30), (30, 7, 30)])
        log = pd.DataFrame([("2026-10-26T12:00:00Z", 7, t, p, 0) for t, p in ((1, 10), (2, 20), (3, 30))],
                           columns=LOG.columns)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lineup_log.csv"
            log.to_csv(path, index=False)
            sched = pd.DataFrame({"pro_team_id": [1], "scoring_period": [7], "date": [D26], "game_id": [1]})
            real = pd.read_csv
            with mock.patch.object(pd, "read_csv", lambda f, *a, **k: sched.copy() if "schedule" in str(f) else real(f, *a, **k)):
                best, worst = LP.performers(self.lv(), g, D26, D26, n=6, log=path)
        eff = {r["player"]: r["effect"] for r in best + worst}
        base = LV.win_week(LV.cat_probs(*self.lv().dists(1, 2)[:4]))
        self.assertGreater(eff["P10"], 0)                   # home side, above his line: helped team 1
        self.assertGreater(eff["P20"], 0)                   # away side, same line: helped team 2 by the same
        self.assertAlmostEqual(eff["P10"], eff["P20"], places=4)
        self.assertNotIn("P30", eff)                        # decided matchup: no effect
        self.assertEqual(worst, [])
        self.assertEqual(best[0]["team"], "T1")
        self.assertEqual(best[0]["g"], 1)
        self.assertLess(eff["P10"], 1 - base)


class DistsTest(unittest.TestCase):
    def test_matchup_is_built_from_dists(self):
        lv = LV.Live.__new__(LV.Live)
        mA, vA = np.full(len(M.STATK), 50.0), np.full(len(M.STATK), 100.0)
        mB, vB = np.full(len(M.STATK), 45.0), np.full(len(M.STATK), 100.0)
        with mock.patch.object(LV.Live, "dists", lambda self, h, a, rosters=None: (mA, vA, mB, vB, {"x": 1})):
            pc, pw, (a, b), m = lv.matchup(1, 2)
        np.testing.assert_allclose(pc, LV.cat_probs(mA, vA, mB, vB))
        self.assertAlmostEqual(pw, LV.win_week(pc))
        self.assertEqual(m, {"x": 1})


if __name__ == "__main__":
    unittest.main()
