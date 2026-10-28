"""Players' current NBA team (DECISIONS.md T1): inseason_season puts each player on his ESPN team, so a traded player gets
his new team's schedule and a waived one (team 0) no games; unlisted players keep the preseason file's team, and the
cached preseason season is not modified.

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


def season():
    """Three players on IND (11), LAC (12), LAL (13); team x week games for weeks 0-21 (regular 0-18, playoffs 19-21)."""
    s = object.__new__(M.Season)
    s.df = pd.DataFrame({"id": [101, 102, 103], "pro_team_id": [11, 12, 13], "team": ["IND", "LAC", "LAL"],
                         "po_mult": [1.0, 1.0, 1.0]})
    s.wg = pd.DataFrame({w: [3, 4, 2] for w in range(22)}, index=pd.Index([11, 12, 13], name="pro_team_id"))
    s.reg_weeks, s.playoff_weeks = list(range(19)), (19, 20, 21)
    return s


class WithTeamsTest(unittest.TestCase):
    def test_moved_player_takes_the_new_teams_schedule(self):
        s0 = season()
        s = I.with_teams(s0, {101: 12})                            # IND -> LAC
        self.assertEqual((s.df["pro_team_id"][0], s.df["team"][0]), (12, "LAC"))
        np.testing.assert_array_equal(s.games_matrix[0], s.games_matrix[1])
        np.testing.assert_array_equal(I.games_by_period(s, [2, 3])[0], [4, 4, 4, 4, 4])
        self.assertEqual(s.team_idx[0], 12)

    def test_no_team_no_games(self):
        s = I.with_teams(season(), {102: 0})
        self.assertEqual((s.df["pro_team_id"][1], s.df["team"][1]), (0, "FA"))
        np.testing.assert_array_equal(s.games_matrix[1], 0)
        np.testing.assert_array_equal(I.games_by_period(s, [2, 3])[1], 0)

    def test_unlisted_and_unchanged_players_keep_their_team(self):
        s0 = season()
        s = I.with_teams(s0, {103: 13, 999: 5})                    # same team; a player the model doesn't have
        pd.testing.assert_frame_equal(s.df, s0.df)
        np.testing.assert_array_equal(s.games_matrix, s0.games_matrix)

    def test_cached_season_not_modified(self):
        s0 = season()
        before = s0.games_matrix.copy()                            # cached on the preseason season first
        s = I.with_teams(s0, {101: 0})
        self.assertEqual(s0.df["pro_team_id"][0], 11)
        np.testing.assert_array_equal(s0.games_matrix, before)
        self.assertEqual(s.games_matrix[0].sum(), 0)                # recomputed for the copy, not inherited

    def test_live_remaining_games_follow_the_team(self):
        s = I.with_teams(season(), {101: 0, 102: 11})
        s.season_start = pd.Timestamp("2026-10-19")
        sched = pd.DataFrame({"pro_team_id": [11, 12, 11], "scoring_period": [9, 9, 10],
                              "date": ["2026-10-28", "2026-10-28", "2026-10-29"], "game_id": [1, 2, 3]})
        with mock.patch.object(pd, "read_csv", lambda *a, **k: sched.assign(date=pd.to_datetime(sched.date))):
            n = LV.remaining_games(s, pd.DataFrame(columns=["id", "scoring_period"]), 2, today=pd.Timestamp("2026-10-28"))
        np.testing.assert_array_equal(n, [0, 2, 0])                # waived: none; moved to IND: IND's two


class CurrentTeamsTest(unittest.TestCase):
    def test_league_rostered_players_override_the_feed(self):
        lg = {"pro_team": {"101": 13, "104": 20, "105": None}}
        with mock.patch.object(I, "all_teams", lambda: {101: 11, 102: 0}), mock.patch.object(I, "load_league", lambda: lg):
            self.assertEqual(I.current_teams(), {101: 13, 102: 0, 104: 20})

    def test_no_league_file(self):
        def missing():
            raise FileNotFoundError
        with mock.patch.object(I, "all_teams", lambda: {101: 11}), mock.patch.object(I, "load_league", missing):
            self.assertEqual(I.current_teams(), {101: 11})


class PlayerFeedTest(unittest.TestCase):
    def test_statuses_and_teams_from_one_feed(self):
        pl = pd.DataFrame({"id": [101, 102, 103], "injury": ["OUT", None, "ACTIVE"], "pro_team_id": [11, 0, np.nan]})
        calls, disk = [], {}
        with mock.patch.object(I.fetch, "fetch_players", lambda *a, **k: calls.append(1) or []), \
                mock.patch.object(I.fetch, "flatten", lambda *a, **k: pl), \
                mock.patch.object(I, "disk_cached", lambda name, ttl, fn: disk.setdefault(name, fn()) if name not in disk
                                  else disk[name]), \
                mock.patch.object(I, "_STATUSES", None), mock.patch.object(I, "_TEAMS", None):
            self.assertEqual(I.all_statuses(), {101: "OUT", 102: "", 103: "ACTIVE"})
            self.assertEqual(I.all_teams(), {101: 11, 102: 0})
        self.assertEqual((len(calls), list(disk)), (1, ["player_feed"]))      # one download serves both

    def test_feed_failure(self):
        def down(*a, **k):
            raise ConnectionError
        with mock.patch.object(I.fetch, "fetch_players", down), mock.patch.object(I, "disk_cached", lambda n, t, fn: fn()), \
                mock.patch.object(I, "_STATUSES", None), mock.patch.object(I, "_TEAMS", None):
            self.assertEqual((I.all_statuses(), I.all_teams()), ({}, {}))


if __name__ == "__main__":
    unittest.main()
