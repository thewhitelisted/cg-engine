"""The engine's day is ESPN's scoring day, which rolls over at 03:00 ET: a game still being played after midnight stays on
its own day (its clock, its rest nights, its lineup locks), and last night's box scores are fetched once the day
changes.

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
import inseason as I  # noqa: E402
import live as LV  # noqa: E402
import model as M  # noqa: E402

D28, D29 = pd.Timestamp("2026-10-28"), pd.Timestamp("2026-10-29")


class GameDayTest(unittest.TestCase):
    def test_rolls_over_at_three_am_eastern(self):
        self.assertEqual(I.game_day(pd.Timestamp("2026-10-28 23:30")), D28)
        self.assertEqual(I.game_day(pd.Timestamp("2026-10-29 00:30")), D28)          # late game still running
        self.assertEqual(I.game_day(pd.Timestamp("2026-10-29 02:59:59")), D28)
        self.assertEqual(I.game_day(pd.Timestamp("2026-10-29 03:00")), D29)

    def test_aware_times_are_converted_to_eastern(self):
        self.assertEqual(I.game_day(pd.Timestamp("2026-10-29 04:30", tz="UTC")), D28)  # 00:30 EDT
        self.assertEqual(I.game_day(pd.Timestamp("2027-01-15 07:30", tz="UTC")), pd.Timestamp("2027-01-14"))  # 02:30 EST

    def test_returns_a_naive_date(self):
        d = I.game_day()
        self.assertIsNone(d.tzinfo)
        self.assertEqual(d, d.normalize())


class DefaultsFollowTheGameDay(unittest.TestCase):
    def test_day_schedule_starts_on_the_game_day(self):
        sched = pd.DataFrame({"pro_team_id": [1, 1, 2], "scoring_period": [9, 10, 9],
                              "date": pd.to_datetime(["2026-10-28", "2026-10-29", "2026-10-28"]), "game_id": [1, 2, 3]})
        season = mock.Mock(df=pd.DataFrame({"id": [7, 8], "pro_team_id": [1, 2]}))
        with mock.patch.object(I, "game_day", lambda now=None: D28), \
                mock.patch.object(LV, "period_dates", lambda s, p: (D28, D28 + pd.Timedelta(days=4))), \
                mock.patch.object(pd, "read_csv", lambda *a, **k: sched.copy()):
            days = LV.day_schedule(season, 2, today_frac={1: 0.4, 2: 0.0})
        self.assertEqual(days[0][0], D28)                                  # last night's game, 40% left
        np.testing.assert_allclose(days[0][1], [0.4, 0.0])
        self.assertEqual(days[1][0], D29)

    def test_rest_nights_use_the_game_day(self):
        sched = pd.DataFrame({"pro_team_id": [1, 1], "scoring_period": [8, 9],
                              "date": pd.to_datetime(["2026-10-27", "2026-10-28"]), "game_id": [1, 2]})
        s = object.__new__(M.Season)
        s.df = pd.DataFrame({"id": [7], "pro_team_id": [1], "espn_min": [35.0]})
        g = pd.DataFrame({"id": [7], "scoring_period": [8], "min": [37.0]})
        with mock.patch.object(pd, "read_csv", lambda *a, **k: sched.copy()):
            with mock.patch.object(I, "game_day", lambda now=None: D28):      # 00:30 on the 29th
                self.assertTrue(I.rest_nights(s, g, {0: "OUT"})[0])
            with mock.patch.object(I, "game_day", lambda now=None: D29):      # after 03:00
                self.assertFalse(I.rest_nights(s, g, {0: "OUT"})[0])


class GamelogCacheTest(unittest.TestCase):
    def test_refetched_when_the_game_day_changes(self):
        calls = []

        def fetch():
            calls.append(1)
            return pd.DataFrame({"id": [len(calls)]})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "DATA", Path(tmp)), \
                mock.patch.object(I, "_fetch_gamelogs", fetch):
            for day, expect in ((D28, 1), (D28, 1), (D29, 2)):
                I._GAMELOGS = None
                with mock.patch.object(I, "game_day", lambda now=None, d=day: d):
                    self.assertEqual(int(I.gamelogs()["id"][0]), expect)
            I._GAMELOGS = None
        self.assertEqual(len(calls), 2)
