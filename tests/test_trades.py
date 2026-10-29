"""weekly.apply_trade: a roster spot freed by a 2-for-1 is filled with the best free agent, meaning a player on no
league roster (a player in another team's IR slot is not a free agent); with more incoming than outgoing, the
lowest-value players are dropped.

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M  # noqa: E402
import weekly as W  # noqa: E402

N = 2 * M.ROSTER + 4      # two full active rosters, one IR player each, and two free agents


def ctx():
    """Team A: players 0..12; team B: 13..25; 26 on A's IR, 27 on B's IR (both very valuable); 28, 29 free agents."""
    val = np.arange(N, dtype=float)
    val[26], val[27], val[28], val[29] = 100.0, 99.0, 50.0, 40.0
    ids = np.arange(1000, 1000 + N)
    rost = np.stack([np.arange(M.ROSTER), np.arange(M.ROSTER, 2 * M.ROSTER)])
    lg = {"teams": [{"id": 1, "roster": [[int(ids[p]), "", 0, "ACTIVE"] for p in rost[0]] + [[int(ids[26]), "", 13, "OUT"]]},
                    {"id": 2, "roster": [[int(ids[p]), "", 0, "ACTIVE"] for p in rost[1]] + [[int(ids[27]), "", 13, "OUT"]]}]}
    return mock.Mock(val=val, rost=rost, lg=lg, df=pd.DataFrame({"id": ids}))


class ApplyTradeTest(unittest.TestCase):
    def test_freed_spot_takes_the_best_real_free_agent(self):
        cx = ctx()
        r = W.apply_trade(cx, cx.rost[0], out_players=(0, 1), in_players=(20,))
        self.assertEqual(len(r), M.ROSTER)
        self.assertIn(28, r)                                   # the best free agent, value 50
        self.assertNotIn(26, r)                                # own IR player is not a free agent
        self.assertNotIn(27, r)                                # nor another team's IR player
        self.assertIn(20, r)
        self.assertFalse({0, 1} & set(r))

    def test_freed_spot_skips_a_free_agent_who_breaks_the_position_limits(self):
        cx = ctx()
        cx.roster_ok = lambda roster, team_slot: 28 not in roster          # 28 would be one center too many
        r = W.apply_trade(cx, cx.rost[0], out_players=(0, 1), in_players=(20,), team_slot=0)
        self.assertNotIn(28, r)
        self.assertIn(29, r)                                   # the next best legal free agent

    def test_more_incoming_drops_the_lowest_value(self):
        cx = ctx()
        r = W.apply_trade(cx, cx.rost[1], out_players=(20,), in_players=(0, 1))
        self.assertEqual(len(r), M.ROSTER)
        self.assertNotIn(0, r)                                 # value 0: dropped
        self.assertIn(1, r)
        self.assertNotIn(20, r)
