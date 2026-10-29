"""K1 lineup locks (DECISIONS.md, 2026-10-25): Live.expected_starts on small synthetic rosters, and the scoreboard's
"started" flag. Every player is fully available (share ~1), so expected starts are ~1 for a starter and ~0 otherwise,
times the share of his game still to play.

    python -m unittest discover -s tests
"""
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live as LV  # noqa: E402
import live_server as S  # noqa: E402
from research_lineup import SLOTS, eligible  # noqa: E402

TODAY = pd.Timestamp(date.today())
PG, SG, SF, PF, C, G, F, UTIL, BENCH = 0, 1, 2, 3, 4, 5, 6, 11, 12


def live_obj(players, frac, started=(), later_days=1, locks=True):
    """A Live with just what expected_starts reads. players: [(nba_team, eligibility, value, ESPN slot)];
    frac: {nba_team: share of today's game left} (teams without a game today are absent)."""
    lv = LV.Live.__new__(LV.Live)
    team = np.array([p[0] for p in players])
    lv.season = mock.Mock(df=pd.DataFrame({"id": range(len(players)), "pro_team_id": team}))
    lv.elig = np.array([[eligible(p[1], s) for s in SLOTS] for p in players], float)
    lv.val = np.array([p[2] for p in players], float)
    lv.share, lv.m_now = np.full(len(players), 0.999), np.zeros(len(players), int)
    lv.rest = np.zeros(len(players), bool)
    lv.p11, lv.S = LV.RHO, np.array([1.0, 0.5, 0.25, 0.1, 0.05])
    today_w = np.array([frac.get(t, 0.0) for t in team])
    has_today = np.array([t in frac for t in team])
    lv.days = [(TODAY, today_w * has_today)] + [(TODAY + pd.Timedelta(days=k), np.ones(len(players)))
                                               for k in range(1, later_days + 1)]
    lv.slot_now = np.array([p[3] for p in players])
    lv.locks, lv._cache, lv._daily = locks, {}, {}
    lv.lock_day, lv.lock_started = TODAY, np.isin(team, list(started))
    return lv


def starts(lv, n):
    E, _ = lv.expected_starts(np.arange(n))
    return E


class LockTest(unittest.TestCase):
    def test_benched_player_in_started_game_counts_zero(self):
        players = [(1, "PG", 5, PG), (1, "SG", 4, BENCH)]                 # both on team 1, game half played
        on = live_obj(players, {1: 0.5}, started={1}, later_days=0)
        off = live_obj(players, {1: 0.5}, started={1}, later_days=0, locks=False)
        self.assertAlmostEqual(starts(on, 2)[1], 0.0)
        self.assertAlmostEqual(starts(off, 2)[1], 0.5, places=2)          # L4 credits the unplayed half
        self.assertAlmostEqual(starts(on, 2)[0], 0.5, places=2)

    def test_benched_player_before_tip_stays_eligible(self):
        players = [(1, "PG", 5, PG), (2, "SG", 4, BENCH)]                 # team 2 hasn't tipped
        lv = live_obj(players, {1: 0.5, 2: 1.0}, started={1}, later_days=0)
        self.assertAlmostEqual(starts(lv, 2)[1], 1.0, places=2)

    def test_active_player_in_started_game_holds_his_slot(self):
        # 10 later-game players fill all 10 slots by value; a low-value centre already playing is in the C slot.
        later = [(2, "PG", 9, BENCH), (2, "SG", 9, BENCH), (2, "SF", 9, BENCH), (2, "PF", 9, BENCH), (2, "C", 9, BENCH),
                 (2, "PG", 8, BENCH), (2, "SF", 8, BENCH), (2, "SG", 7, BENCH), (2, "PF", 7, BENCH), (2, "C", 1, BENCH)]
        players = later + [(1, "C", 0.5, C)]
        on = live_obj(players, {1: 0.4, 2: 1.0}, started={1}, later_days=0)
        off = live_obj(players, {1: 0.4, 2: 1.0}, started={1}, later_days=0, locks=False)
        e_on, e_off = starts(on, 11), starts(off, 11)
        self.assertAlmostEqual(e_off[10], 0.0, places=2)                   # L4 swaps him out for a later player
        self.assertAlmostEqual(e_on[10], 0.4, places=2)                    # K1: he's locked in, 40% of his game left
        self.assertAlmostEqual(e_on[9], 0.0, places=2)                     # the C slot is gone: the weaker C sits
        self.assertAlmostEqual(e_on[4], 1.0, places=2)                     # the better C still fits (UTIL)
        self.assertAlmostEqual(sum(e_on[:10]), 9.0, places=1)

    def test_finished_game_still_holds_the_slot(self):
        later = [(2, pos, 9, BENCH) for pos in ("PG", "SG", "SF", "PF", "C", "PG", "SG", "SF", "PF", "C")]
        players = later + [(1, "PG", 1, UTIL)]                             # played and finished in a UTIL slot
        on = live_obj(players, {1: 0.0, 2: 1.0}, started={1}, later_days=0)
        off = live_obj(players, {1: 0.0, 2: 1.0}, started={1}, later_days=0, locks=False)
        self.assertAlmostEqual(sum(starts(off, 11)[:10]), 10.0, places=1)  # L4: all ten tonight
        self.assertAlmostEqual(sum(starts(on, 11)[:10]), 9.0, places=1)    # K1: his UTIL is taken for the day

    def test_no_started_game_is_identical(self):
        players = [(1 + i % 3, ["PG", "SG", "SF", "PF", "C"][i % 5], 10 - i * 0.5, [PG, BENCH, UTIL][i % 3])
                   for i in range(14)]                                     # crowded: the Monte Carlo path runs
        for started in (set(), {7}):                                       # nobody's game started / another team's
            on = live_obj(players, {1: 1.0, 2: 1.0, 3: 1.0}, started=started, later_days=2)
            off = live_obj(players, {1: 1.0, 2: 1.0, 3: 1.0}, started=started, later_days=2, locks=False)
            np.testing.assert_array_equal(on.expected_starts(np.arange(14))[0], off.expected_starts(np.arange(14))[0])

    def test_future_days_unchanged(self):
        players = [(1 + i % 3, ["PG", "SG", "SF", "PF", "C"][i % 5], 10 - i * 0.5, [PG, BENCH, UTIL][i % 3])
                   for i in range(14)]
        on = live_obj(players, {1: 0.3, 2: 1.0, 3: 1.0}, started={1}, later_days=3)
        off = live_obj(players, {1: 0.3, 2: 1.0, 3: 1.0}, started={1}, later_days=3, locks=False)
        on.expected_starts(np.arange(14))
        off.expected_starts(np.arange(14))
        d_on, d_off = on._daily[(tuple(range(14)), ())], off._daily[(tuple(range(14)), ())]
        self.assertFalse(np.allclose(d_on[0][2], d_off[0][2]))             # today differs
        for a, b in zip(d_on[1:], d_off[1:]):                              # later days: identical numbers
            np.testing.assert_array_equal(a[2], b[2])

    def test_slot_change_before_tip_reaches_the_model(self):
        players = [(1, "PG", 5, BENCH)]
        lv = live_obj(players, {1: 0.5}, started={1}, later_days=0)
        self.assertAlmostEqual(starts(lv, 1)[0], 0.0)
        lv.slot_now = np.array([PG])                                       # what update(slots=...) installs
        lv._cache, lv._daily = {}, {}
        self.assertAlmostEqual(starts(lv, 1)[0], 0.5, places=2)

    def test_update_installs_slots_and_started(self):
        lv = LV.Live.__new__(LV.Live)
        lv.season = mock.Mock(df=pd.DataFrame({"id": [101, 102], "pro_team_id": [1, 2]}))
        lv.g, lv.period, lv.lg = pd.DataFrame(), 1, {"teams": []}
        with mock.patch.object(LV, "remaining_games", return_value=np.zeros(2)), \
                mock.patch.object(LV, "MODEL", "L1b"):
            lv.update({"period": 1}, {1: 0.5}, started={1}, slots={101: PG, 102: BENCH})
        self.assertEqual(list(lv.slot_now), [PG, BENCH])
        self.assertEqual(lv.started, {1})


class ScoreboardTest(unittest.TestCase):
    def scoreboard(self, events):
        resp = mock.Mock(json=lambda: {"events": events}, raise_for_status=lambda: None)
        with mock.patch.object(S.requests, "get", return_value=resp):
            return S.scoreboard(datetime(2026, 10, 25, 20, 0), now=pd.Timestamp("2026-10-25T21:30Z"))

    def event(self, tid, state, name, tip):
        return {"date": tip, "competitions": [{"status": {"type": {"state": state, "name": name, "shortDetail": ""},
                                                          "period": 2, "clock": 300},
                                               "competitors": [{"team": {"id": str(tid), "abbreviation": "X"}},
                                                               {"team": {"id": str(tid + 100), "abbreviation": "Y"}}]}]}

    def test_started_flag(self):
        past, future = "2026-10-25T21:00Z", "2026-10-25T23:30Z"           # 5 PM ET (passed) / 7:30 PM ET
        sb = self.scoreboard([self.event(1, "in", "STATUS_IN_PROGRESS", past),
                              self.event(2, "post", "STATUS_FINAL", past),
                              self.event(3, "post", "STATUS_POSTPONED", past),
                              self.event(4, "pre", "STATUS_SCHEDULED", past),       # tip time passed, not tipped
                              self.event(5, "pre", "STATUS_SCHEDULED", future)])
        self.assertEqual({t: v[4] for t, v in sb.items() if t < 100}, {1: True, 2: True, 3: False, 4: True, 5: False})
        self.assertEqual(LV.started_teams(sb) & {1, 2, 3, 4, 5}, {1, 2, 4})
        self.assertIsNone(LV.started_teams(None))


if __name__ == "__main__":
    unittest.main()
