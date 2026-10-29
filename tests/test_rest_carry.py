"""The days after a rest night (DECISIONS.md S3b): a player whose team's last game day was a rest night for him keeps
S3's treatment (missed run 0, not absent, A = 1) on his team's gap days, while ESPN lists him OUT or DAY_TO_DAY, until
his team's next game day. His status on the rest night is the last one logged before that game's scheduled tip.

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
import model as M  # noqa: E402

D27, D28, D29, D30 = (pd.Timestamp(f"2026-10-{d}") for d in (27, 28, 29, 30))
# team 1: back-to-back Oct 27-28, next game Oct 30 (Oct 29 is a gap day); team 2 also plays Oct 29
SCHED = pd.DataFrame({"pro_team_id": [1, 2, 1, 2, 2, 1], "scoring_period": [8, 8, 9, 9, 10, 11],
                      "date": [D27, D27, D28, D28, D29, D30], "game_id": range(6)})
#   0 rested, now DTD; 1 rested, now ACTIVE; 2 DTD at the end of the rest night; 3 DNP on Oct 27; 4 bench;
#   5 team plays today; 6 played on Oct 28; 7 OUT logged after the tip; 8 rested, still OUT
TEAM = [1, 1, 1, 1, 1, 2, 1, 1, 1]
MIN = [35.0, 35.0, 35.0, 35.0, 15.0, 35.0, 35.0, 35.0, 30.0]
NOW = {0: "DAY_TO_DAY", 1: "ACTIVE", 2: "DAY_TO_DAY", 3: "OUT", 4: "OUT", 5: "OUT", 6: "OUT", 7: "OUT", 8: "OUT"}
G_LOG = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5, 6, 6, 7, 8], "scoring_period": [8, 8, 8, 8, 8, 8, 8, 9, 8, 8],
                      "min": [36.0, 36.0, 30.0, 0.0, 20.0, 30.0, 30.0, 25.0, 30.0, 33.0]})
TIP = pd.Timestamp("2026-10-28T23:30:00Z")                            # team 1's Oct 28 game, 19:30 EDT
LOG = pd.DataFrame([("2026-10-22T22:00:00Z", p, "ACTIVE") for p in range(9)]
                   + [("2026-10-28T19:00:00Z", p, "OUT") for p in (0, 1, 3, 4, 5, 6, 8)]
                   + [("2026-10-28T19:00:00Z", 2, "DAY_TO_DAY"),
                      ("2026-10-28T23:30:00Z", 8, "OUT"),            # re-logged at the tip: still OUT
                      ("2026-10-28T23:35:00Z", 7, "OUT"),            # after it
                      ("2026-10-29T05:06:00Z", 0, "DAY_TO_DAY"),     # cleared to DTD after the game (Doncic, Oct 28)
                      ("2026-10-29T14:00:00Z", 1, "ACTIVE")],
                   columns=["time_utc", "player_id", "status"])


def season():
    s = object.__new__(M.Season)
    s.df = pd.DataFrame({"id": range(9), "pro_team_id": TEAM, "espn_min": MIN, "avail": 0.8})
    return s


class TipTimesTest(unittest.TestCase):
    def test_teams_of_games_dated_that_day(self):
        js = {"events": [
            {"date": "2026-10-28T23:30Z", "competitions": [{"competitors": [{"team": {"id": "13"}}, {"team": {"id": "9"}}]}]},
            {"date": "2026-10-29T02:00Z", "competitions": [{"competitors": [{"team": {"id": "1"}}, {"team": {"id": "2"}}]}]},
            {"date": "2026-10-29T23:00Z", "competitions": [{"competitors": [{"team": {"id": "5"}}, {"team": {"id": "6"}}]}]}]}
        resp = mock.Mock(json=lambda: js, raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "DATA", Path(tmp)), \
                mock.patch.object(I.requests, "get", lambda *a, **k: resp):
            tips = I.tip_times(D28)
        self.assertEqual(set(tips), {13, 9, 1, 2})                      # 02:00 UTC Oct 29 is 22:00 EDT Oct 28
        self.assertEqual(tips[13], pd.Timestamp("2026-10-28T23:30Z"))

    def test_feed_down_is_empty_and_not_cached(self):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise OSError("down")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "DATA", Path(tmp)), \
                mock.patch.object(I.requests, "get", boom):
            self.assertEqual(I.tip_times(D28), {})
            self.assertEqual(I.tip_times(D28), {})
        self.assertEqual(len(calls), 2)


class RestCarryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        SCHED.to_csv(d / "schedule_2027.csv", index=False)
        LOG.to_csv(d / "status_log.csv", index=False)
        self.tips = {D28: {1: TIP, 2: TIP}}
        self.patches = [mock.patch.object(M, "DATA", d),
                        mock.patch.object(I, "tip_times", lambda day: self.tips.get(day, {}))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def carry(self, today, status=NOW, g=G_LOG):
        return I.rest_carry(season(), g, status, today=today)

    def test_who_carries_on_the_gap_day(self):
        np.testing.assert_array_equal(self.carry(D29), [True, False, False, False, False, False, False, False, True])

    def test_ends_on_the_next_game_day(self):
        self.assertFalse(self.carry(D30).any())

    def test_rest_night_itself_is_S3s(self):
        self.assertFalse(self.carry(D28).any())                       # team 1 plays today: rest_nights covers it

    def test_no_tip_time_no_carry(self):
        self.tips = {}
        self.assertFalse(self.carry(D29).any())

    def test_no_log_or_no_game_logs(self):
        self.assertFalse(self.carry(D29, g=None).any())
        (M.DATA / "status_log.csv").unlink()
        self.assertFalse(self.carry(D29).any())

    def test_gap_day_has_no_run_and_is_not_absent(self):
        av = {"K_gp": 10.0, "survival": [1.0, 1.0, 0.5, 0.4, 0.3, 0.25, 0.2], "absent_m": 2}
        c, n, m = np.full(9, 8.0), np.full(9, 7.0), np.ones(9, int)
        with mock.patch.object(I, "AVAIL", av), mock.patch.object(I, "participation", lambda *a, **k: (c, n, m)), \
                mock.patch.object(I, "game_day", lambda now=None: D29):
            _, A, info = I.availability(season(), G_LOG, NOW, [2], np.full((9, 1), 2.0))
        np.testing.assert_array_equal(info["rest_carry"][[0, 8]], True)
        self.assertFalse(info["rest"].any())                           # nothing is taken out of a game today
        np.testing.assert_array_equal(info["m"][[0, 8]], 0)
        self.assertFalse(info["absent"][[0, 8]].any())
        np.testing.assert_array_equal(A[[0, 8]], 1.0)
        self.assertEqual(info["m"][3], I.START_M)                      # OUT without a rest night: absence path
        self.assertEqual(info["m"][2], 1)                              # DTD, not rested: production's DTD


if __name__ == "__main__":
    unittest.main()
