"""Rest night on a back-to-back (DECISIONS.md S3): a player listed OUT with espn_min >= 20 whose team played yesterday and
plays today, and who played yesterday, sits today's game (0) and is healthy after it (missed run 0, not absent, A = 1);
other OUT players are unchanged.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inseason as I  # noqa: E402
import live as LV  # noqa: E402
import model as M  # noqa: E402

TODAY = pd.Timestamp("2026-10-28")
# teams 1 and 2 play yesterday (period 8) and today (9); team 3 plays only today; team 4 only yesterday
SCHED = pd.DataFrame({"pro_team_id": [1, 2, 4, 1, 2, 3, 1], "scoring_period": [8, 8, 8, 9, 9, 9, 11],
                      "date": pd.to_datetime(["2026-10-27"] * 3 + ["2026-10-28"] * 3 + ["2026-10-30"]), "game_id": range(7)})
#        rest night, DNP yesterday, no game yesterday, bench, DTD, no game today, also rest (team 2)
TEAM = [1, 1, 3, 1, 1, 4, 2]
MIN = [35.0, 30.0, 30.0, 15.0, 30.0, 30.0, 25.0]
STATUS = {0: "OUT", 1: "OUT", 2: "OUT", 3: "OUT", 4: "DAY_TO_DAY", 5: "OUT", 6: "OUT"}
G_LOG = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5, 6], "scoring_period": [8, 8, 6, 8, 8, 8, 8],
                      "min": [37.0, 0.0, 30.0, 20.0, 30.0, 30.0, 28.0]})


def season():
    s = object.__new__(M.Season)
    s.df = pd.DataFrame({"id": range(7), "pro_team_id": TEAM, "espn_min": MIN, "avail": 0.8})
    return s


def rest():
    with mock.patch.object(pd, "read_csv", lambda *a, **k: SCHED.copy()):
        return I.rest_nights(season(), G_LOG, STATUS, today=TODAY)


class RestNightTest(unittest.TestCase):
    def test_who_is_on_a_rest_night(self):
        np.testing.assert_array_equal(rest(), [True, False, False, False, False, False, True])

    def test_no_game_logs(self):
        with mock.patch.object(pd, "read_csv", lambda *a, **k: SCHED.copy()):
            self.assertFalse(I.rest_nights(season(), None, STATUS, today=TODAY).any())

    def test_rest_night_has_no_run_and_is_not_absent(self):
        av = {"K_gp": 10.0, "survival": [1.0, 1.0, 0.5, 0.4, 0.3, 0.25, 0.2], "absent_m": 2}
        c, n, m = np.full(7, 8.0), np.full(7, 8.0), np.zeros(7, int)
        r = rest()
        with mock.patch.object(I, "AVAIL", av), mock.patch.object(I, "participation", lambda *a, **k: (c, n, m)), \
                mock.patch.object(I, "rest_nights", lambda *a, **k: r):
            _, A, info = I.availability(season(), G_LOG, STATUS, [2], np.full((7, 1), 3.0))
        np.testing.assert_array_equal(info["rest"], r)
        np.testing.assert_array_equal(info["m"], [0, I.START_M, I.START_M, I.START_M, 0, I.START_M, 0])
        np.testing.assert_array_equal(info["absent"], [False, True, True, True, False, True, False])
        np.testing.assert_array_equal(A[[0, 6]], 1.0)


class LiveRestTest(unittest.TestCase):
    def test_sits_today_then_healthy(self):
        lv = LV.Live.__new__(LV.Live)
        lv.season = mock.Mock(df=pd.DataFrame({"id": [0, 1], "pro_team_id": [1, 1]}))
        lv.elig, lv.val = np.ones((2, 10)), np.array([2.0, 1.0])
        lv.share, lv.m_now, lv.rest = np.array([0.8, 0.8]), np.zeros(2, int), np.array([True, False])
        lv.p11, lv.S = LV.RHO, np.array([1.0, 1.0, 0.5, 0.4, 0.3])
        lv.days = [(TODAY, np.ones(2)), (TODAY + pd.Timedelta(days=2), np.ones(2))]
        lv.locks, lv._cache, lv._daily, lv.lock_day = False, {}, {}, None
        lv.lock_started = np.zeros(2, bool)
        E, _ = lv.expected_starts([0, 1])
        today = {d: dict(zip(pl, e)) for d, pl, e, _ in lv._daily[((0, 1), ())]}
        self.assertNotIn(0, today[TODAY])                              # sits tonight
        healthy1 = 0.8 + 0.2 * (1 - LV.RHO)
        self.assertAlmostEqual(today[TODAY + pd.Timedelta(days=2)][0], healthy1)     # next game: j = 1
        self.assertAlmostEqual(today[TODAY + pd.Timedelta(days=2)][1], 0.8 + 0.2 * (1 - LV.RHO) ** 2)
        self.assertAlmostEqual(E[0], healthy1)
