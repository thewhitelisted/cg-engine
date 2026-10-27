"""Scoring a move in weekly.py: inseason.displaced_games works out a returning IR player's displacement for the roster being
scored, so a move that drops or trades the weakest player moves it with him (weekly.py used to keep it on the original
weakest players: dropping one looked like adding a 14th player); and the week in progress is re-priced for that roster.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inseason as I  # noqa: E402
import weekly as W  # noqa: E402

G = np.array([[3.0, 4.0]] * 7)                        # players 0-6, two periods
A = np.ones_like(G)
A[5] = [0.25, 1.0]                                    # IR player 5: out most of period 1, back for period 2
A[6] = [0.0, 0.5]                                     # IR player 6: out, then back for half of period 2
VAL = np.array([3.0, 1.0, 2.0, 0.0, 0.5, 5.0, 4.0])   # player 3 is a free agent, the weakest of all
IRS = np.array([[5, -2], [-2, -2]])


class DisplacementTest(unittest.TestCase):
    def test_weakest_active_gives_up_the_returners_games(self):
        g = I.displaced_games(G, A, np.array([[0, 1, 2], [4, -1, -1]]), IRS, VAL)
        np.testing.assert_allclose(g[1], [3 * 0.75, 0])                  # player 1, weakest on team 0
        np.testing.assert_allclose(g[[0, 2, 4]], G[[0, 2, 4]])             # nobody else, nor the other team

    def test_moves_with_the_roster(self):
        """Drop player 1 for free agent 3: the displacement moves to 3; player 1, now unrostered, keeps his games."""
        g = I.displaced_games(G, A, np.array([[0, 3, 2], [4, -1, -1]]), IRS, VAL)
        np.testing.assert_allclose(g[3], [3 * 0.75, 0])
        np.testing.assert_allclose(g[1], G[1])
        before = I.displaced_games(G, A, np.array([[0, 1, 2], [4, -1, -1]]), IRS, VAL)
        self.assertAlmostEqual(before[[0, 1, 2]].sum(), g[[0, 3, 2]].sum())  # the roster's games are unchanged

    def test_two_returners_two_weakest(self):
        irs = np.array([[5, 6], [-2, -2]])
        g = I.displaced_games(G, A, np.array([[0, 1, 2], [4, -1, -1]]), irs, VAL)
        np.testing.assert_allclose(g[1], [3 * 0.75, 0])                  # weakest: player 5's return
        np.testing.assert_allclose(g[2], [3, 4 * 0.5])                   # next weakest: player 6's
        np.testing.assert_allclose(g[0], G[0])

    def test_no_ir(self):
        np.testing.assert_allclose(I.displaced_games(G, A, np.array([[0, 1, 2], [4, -1, -1]]), np.full((2, 2), -2), VAL),
                                   G * A)


class WeekInProgressTest(unittest.TestCase):
    """weekly.Ctx.week0_for: a move's effect on the week in progress is priced by the live engine with that roster; a
    player who changes team sits on the bench (ESPN places adds and incoming trades there), and slots are restored."""

    def ctx(self):
        seen = []
        lv = SimpleNamespace(slot_now=np.array([0, 11, 12, -1, 4, 13, 13]))
        lv.week0_pc = lambda pairs, rost: seen.append((lv.slot_now.copy(), rost.copy())) or "priced"
        cx = SimpleNamespace(lv=lv, rost=np.array([[0, 1, 2], [4, -1, -1]]), df=np.zeros(7), week0="base", pairs=[[[0, 1]]])
        return cx, lv, seen

    def test_unchanged_rosters_reuse_the_base(self):
        cx, _, seen = self.ctx()
        self.assertEqual(W.Ctx.week0_for(cx, cx.rost.copy()), "base")
        self.assertEqual(seen, [])

    def test_moved_players_start_on_the_bench(self):
        cx, lv, seen = self.ctx()
        r = np.array([[0, 3, 2], [4, 1, -1]])                    # add free agent 3, trade player 1 to team 1
        self.assertEqual(W.Ctx.week0_for(cx, r), "priced")
        np.testing.assert_array_equal(seen[0][0], [0, 12, 12, 12, 4, 13, 13])
        np.testing.assert_array_equal(lv.slot_now, [0, 11, 12, -1, 4, 13, 13])

    def test_no_live_week(self):
        cx, _, _ = self.ctx()
        cx.lv = None
        self.assertEqual(W.Ctx.week0_for(cx, np.array([[0, 3, 2], [4, -1, -1]])), "base")


if __name__ == "__main__":
    unittest.main()
