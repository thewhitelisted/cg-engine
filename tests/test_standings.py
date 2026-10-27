"""league.py standings (DECISIONS.md F1). The league is ESPN H2H Most Categories: the regular season is ranked by
matchup record (ties half), seeding ties by head-to-head among the tied teams (only if every pair of them met equally
often), then a coin flip, restarting after each seed. STANDINGS = "categories" keeps the earlier rule.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inseason as I  # noqa: E402
import league as L  # noqa: E402

T = 12


def run(weeks, tot0=None, h2h0=None, R=1, K=1, seed=0, standings="matchups", pc=None):
    """_places_model on a schedule of remaining weeks [(a, b, categories a wins), ...] x 6, each week decided exactly
    (category probabilities 0 or 1), or on given category probabilities pc [R, n, 6, 9]. Playoff games are coin flips."""
    pairs = np.array([[(a, b) for a, b, _ in wk] for wk in weeks])
    if pc is None:
        pc = np.array([[[[1.0] * c + [0.0] * (9 - c) for _, _, c in wk] for wk in weeks]] * R)
    W = len(weeks) + 3

    def cat_probs(mA, vA, mB, vB):
        if mA.ndim == 4:                                   # the regular-season weeks [R, n, 6, stat]
            return pc
        return np.full(np.broadcast_shapes(mA.shape, mB.shape)[:-1] + (9,), 0.5)

    mS = np.zeros((R, T, W, 11))
    with mock.patch.object(L.M.Evaluator, "cat_probs", cat_probs), mock.patch.object(L, "STANDINGS", standings):
        return L._places_model(mS, mS, np.random.default_rng(seed), K, pairs, tot0, per_week=True, detail=True,
                               h2h0=h2h0)


WEEK1 = [(0, 1, 5), (2, 3, 9), (4, 5, 5), (6, 7, 5), (8, 9, 5), (10, 11, 5)]
WEEK2 = [(0, 3, 5), (2, 1, 4), (4, 7, 5), (6, 5, 5), (8, 11, 5), (10, 9, 5)]


class StandingsTest(unittest.TestCase):
    def test_weeks_won_beat_categories_won(self):
        """Team 0 wins 5-4 twice (2 wins, 10 categories); team 2 wins 9-0 and loses 4-5 (1 win, 13 categories)."""
        d = run([WEEK1, WEEK2])
        self.assertEqual((d["tot"][0, 0, 0], d["tot"][0, 0, 2]), (2, 1))
        seeds = list(d["seeds"][0, 0])
        self.assertLess(seeds.index(0), seeds.index(2))
        old = run([WEEK1, WEEK2], standings="categories")
        self.assertEqual(old["seeds"][0, 0, 0], 2)                          # the earlier rule: most categories first
        np.testing.assert_array_equal(d["cats"], old["cats"])               # same draws of categories won

    def test_head_start_is_one_win(self):
        """Team 0 won its finished week and loses this one; team 2 had lost and wins this one: level on 1."""
        tot0 = np.zeros(T)
        tot0[0] = 1
        d = run([[(0, 1, 4), (2, 3, 5), (4, 5, 5), (6, 7, 5), (8, 9, 5), (10, 11, 5)]], tot0=tot0)
        self.assertEqual(d["tot"][0, 0, 0], 1)
        self.assertEqual(d["tot"][0, 0, 2], 1)

    def test_points_conserved(self):
        rng = np.random.default_rng(3)
        pc = rng.random((20, 3, 6, 9))
        tot0 = np.arange(T) * 0.5
        d = run([WEEK1, WEEK2, WEEK1], tot0=tot0, R=20, K=4, pc=pc)
        np.testing.assert_allclose(d["tot"].sum(-1), 18 + tot0.sum())
        self.assertTrue(np.isin(d["tot"] - tot0, [0, 1, 2, 3]).all())

    def test_paired_with_the_earlier_rule(self):
        """Both rules use the same draws of categories won and the same playoff draws (F1 check 2)."""
        pc = np.random.default_rng(4).random((5, 2, 6, 9))
        us = []
        for rule in ("categories", "matchups"):
            with mock.patch.object(L, "_bracket_vec", wraps=L._bracket_vec) as bv:
                d = run([WEEK1, WEEK2], R=5, K=3, pc=pc, standings=rule)
            us.append(bv.call_args.args[2])
            if rule == "categories":
                cats = d["cats"]
        np.testing.assert_array_equal(us[0], us[1])
        np.testing.assert_array_equal(cats, d["cats"])

    def test_simulated_weeks_enter_head_to_head(self):
        """Two simulated weeks of 0-1 and 2-3 (a, b, win: [R=1, K=1, week, pair]) on top of a finished 0-1 tie."""
        a = np.array([[0, 2], [2, 1]]).reshape(1, 1, 2, 2)
        b = np.array([[1, 3], [3, 0]]).reshape(1, 1, 2, 2)
        win = np.array([[True, False], [True, False]]).reshape(1, 1, 2, 2)   # 0 beats 1 twice; 3, then 2, win
        h2h0 = np.zeros((2, 4, 4))
        h2h0[0, 0, 1] = h2h0[0, 1, 0] = 0.5
        h2h0[1, 0, 1] = h2h0[1, 1, 0] = 1
        hh, mt = L._head_to_head(a, b, win, 4, h2h0)
        self.assertEqual((hh[0, 0, 1], hh[0, 1, 0], mt[0, 0, 1], mt[0, 1, 0]), (5, 1, 3, 3))   # twice the points
        self.assertEqual((hh[0, 2, 3], hh[0, 3, 2], mt[0, 2, 3]), (2, 2, 2))


class SeedTest(unittest.TestCase):
    N = 3000

    def seed(self, pts, results, meetings=None):
        """results: {(i, j): times i beat j}; meetings: {(i, j): count}, default the results' totals."""
        hh, mt = np.zeros((T, T), int), np.zeros((T, T), int)
        for (i, j), n in results.items():
            hh[i, j] += 2 * n
            mt[i, j] += n
            mt[j, i] += n
        for (i, j), n in (meetings or {}).items():
            mt[i, j] = mt[j, i] = n
        p = np.broadcast_to(np.asarray(pts, float), (self.N, T))
        flip = np.random.default_rng(5).random((self.N, 8, T))
        return L._seed(p, np.broadcast_to(hh, (self.N, T, T)), np.broadcast_to(mt, (self.N, T, T)), flip)

    def test_most_points_first(self):
        s = self.seed(np.arange(T), {})
        self.assertTrue((s == np.arange(11, 3, -1)).all())

    def test_head_to_head_breaks_a_two_way_tie(self):
        pts = np.zeros(T)
        pts[[0, 2]] = 5
        s = self.seed(pts, {(2, 0): 1})
        self.assertTrue((s[:, 0] == 2).all() and (s[:, 1] == 0).all())

    def test_unequal_meetings_skip_head_to_head(self):
        """0 beat 1 twice, 1 beat 2 once, 0 and 2 never met: not all pairs met equally often -> coin flip."""
        pts = np.zeros(T)
        pts[[0, 1, 2]] = 5
        s = self.seed(pts, {(0, 1): 2, (1, 2): 1})
        share = np.bincount(s[:, 0], minlength=3)[:3] / self.N
        self.assertTrue(np.all(np.abs(share - 1 / 3) < 0.04), share)
        rest = s[s[:, 0] == 2]                                              # restart: 0 and 1 met twice; 0 won both
        self.assertTrue((rest[:, 1] == 0).all())

    def test_restart_after_each_seed(self):
        """A three-way cycle (0 beat 1, 1 beat 2, 2 beat 0): a coin flip seeds one, then the other two's game decides."""
        pts = np.zeros(T)
        pts[[0, 1, 2]] = 5
        s = self.seed(pts, {(0, 1): 1, (1, 2): 1, (2, 0): 1})
        share = np.bincount(s[:, 0], minlength=3)[:3] / self.N
        self.assertTrue(np.all(np.abs(share - 1 / 3) < 0.04), share)
        nxt = {0: 1, 1: 2, 2: 0}                                            # the winner of the remaining pair
        self.assertTrue(all((s[s[:, 0] == f, 1] == n).all() for f, n in nxt.items()))

    def test_half_points(self):
        pts = np.zeros(T)
        pts[0], pts[1] = 1.5, 1.0                                           # a tie counts half
        s = self.seed(pts, {(1, 0): 1})
        self.assertTrue((s[:, 0] == 0).all())


class EspnTest(unittest.TestCase):
    LG = {"teams": [{"id": t, "record": r} for t, r in ((1, [0, 0, 1]), (2, [1, 0, 0]), (3, [0, 1, 0]), (4, [0, 0, 1]))],
          "schedule": [[1, 1, 4], [1, 2, 3], [2, 1, 2], [2, 3, 4]], "completed_weeks": [1],
          "results": [[1, 1, 4, "TIE", 4, 4, 1, {}], [1, 2, 3, "HOME", 5, 4, 0, {}]]}

    def test_record_and_head_to_head(self):
        tids = [1, 2, 3, 4]
        pairs, tot0, weeks, _ = I.state(self.LG, tids)
        np.testing.assert_array_equal(tot0, [0.5, 1, 0, 0.5])               # ties count half
        self.assertEqual(weeks, [2])
        h = I.head_to_head(self.LG, tids)
        self.assertEqual((h[0, 0, 3], h[0, 3, 0], h[1, 0, 3]), (0.5, 0.5, 1))
        self.assertEqual((h[0, 1, 2], h[0, 2, 1], h[1, 2, 1]), (1, 0, 1))
        self.assertEqual(h[1].sum(), 4)


if __name__ == "__main__":
    unittest.main()
