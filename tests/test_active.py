"""Off the injury report (DECISIONS.md S2): a player ESPN lists ACTIVE with espn_min >= 20 has no missed run, so he gets
the healthy near-term chain and is not absent (A = 1, his misses count in his share); bench players, other statuses and
OUT / IR are unchanged.

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

AV = {"K_gp": 10.0, "survival": [1.0, 1.0, 0.5, 0.4, 0.3, 0.25, 0.2], "absent_m": 2}
#            starter ACTIVE, bench ACTIVE, starter DTD, starter OUT, starter no status, starter ACTIVE m=1
STATUS = {0: "ACTIVE", 1: "ACTIVE", 2: "DAY_TO_DAY", 3: "OUT", 5: "ACTIVE"}
MIN = [30.0, 15.0, 30.0, 30.0, 30.0, 25.0]
M_RUN = [3, 3, 3, 0, 1, 1]


def run():
    s = object.__new__(M.Season)
    s.df = pd.DataFrame({"id": range(6), "avail": 0.8, "espn_min": MIN})
    c, n = np.full(6, 8.0), np.array([8.0 - m for m in M_RUN])
    G = np.full((6, 2), 3.0)
    with mock.patch.object(I, "AVAIL", AV), mock.patch.object(I, "participation", lambda *a, **k: (c, n, np.array(M_RUN))):
        return I.availability(s, None, STATUS, [2, 3], G)


class ActiveTest(unittest.TestCase):
    def test_active_starter_has_no_run(self):
        share, A, info = run()
        self.assertEqual(info["m"][0], 0)
        self.assertFalse(info["absent"][0])
        np.testing.assert_array_equal(A[0], 1.0)
        self.assertAlmostEqual(share[0], (10 * 0.8 + 5) / (10 + 8))      # not absent: the misses count in his share

    def test_active_starter_gets_the_healthy_chain(self):
        share, _, info = run()
        S = np.asarray(AV["survival"])
        p = LV.near_term_p(share[[0, 5]], info["m"][[0, 5]], 1, LV.RHO, S)
        np.testing.assert_allclose(p, share[[0, 5]] + (1 - share[[0, 5]]) * (1 - LV.RHO))

    def test_others_unchanged(self):
        share, A, info = run()
        np.testing.assert_array_equal(info["m"][1:5], [3, 3, I.START_M, 1])  # bench ACTIVE, DTD, OUT, no status
        np.testing.assert_array_equal(info["absent"][1:5], [True, True, True, False])
        self.assertTrue((A[1] < 1).all() and (A[2] < 1).all())
        self.assertAlmostEqual(share[1], (10 * 0.8 + 5) / (10 + 5))      # absent: the current run is left out
