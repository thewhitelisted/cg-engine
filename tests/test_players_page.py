"""The players page's data (league_page.production_table): every player who has played this week and has a model
line and is on a fantasy roster, ranked by 9-category production against the model's line from the same games; bench
games count; free agents are left out.

    python -m unittest discover -s tests
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import league_page as LP  # noqa: E402
import model as M  # noqa: E402

LINE = dict(pts=20.0, reb=6.0, ast=4.0, stl=1.0, blk=1.0, tpm=2.0, tov=2.0, fgm=8.0, fga=16.0, ftm=2.0, fta=2.0)


def logs(rows):
    """rows: (id, scoring period, minutes, multiple of LINE)"""
    return pd.DataFrame([{**{k: v * x for k, v in LINE.items()}, "id": i, "name": f"P{i}", "scoring_period": p, "min": m}
                         for i, p, m, x in rows])


def lv(ids=(1, 2, 3, 4)):
    df = pd.DataFrame({"id": list(ids), "pro_team_id": [10, 10, 0, 20][:len(ids)]})
    for s in M.STATK:
        df[s] = 0.0
    for s in ("pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fga", "fta"):
        df[s] = LINE[s]                                   # expected: one LINE per game
    df["fgx"] = LINE["fgm"] - M.LG_FG * LINE["fga"]
    df["ftx"] = LINE["ftm"] - M.LG_FT * LINE["fta"]
    x = mock.Mock()
    x.season.df = df
    x.lg = {"teams": [{"id": 7, "name": "Seven Up", "abbrev": "SVN",
                       "roster": [[1, "P1", 0, "ACTIVE"], [2, "P2", 0, "ACTIVE"], [3, "P3", 12, "ACTIVE"]]}]}
    return x


class ProductionTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        pd.DataFrame({"pro_team_id": [10, 20, 10], "scoring_period": [7, 7, 9],
                      "date": ["2026-10-26", "2026-10-26", "2026-10-28"], "game_id": [1, 2, 3]}).to_csv(d / "schedule_2027.csv", index=False)
        pd.DataFrame({"pro_team_id": [10, 20], "abbrev": ["AAA", "BBB"], "team": ["A", "B"]}).to_csv(d / "teams.csv", index=False)
        scale = {s: (M.SK[s], None, 1.0, -1 if s == "tov" else 1)
                 for s in ("fgx", "ftx", "tpm", "reb", "ast", "stl", "blk", "tov", "pts")}
        self.patches = [mock.patch.object(M, "DATA", d), mock.patch.object(M, "static_value", lambda df: (None, scale))]
        for p in self.patches:
            p.start()
        LP._PLAYERS.clear()
        self.week = (pd.Timestamp("2026-10-26"), pd.Timestamp("2026-11-01"))

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_ranked_by_the_gap(self):
        # 1: two games at 1.5x; 2: one at 1.2x; 3 (benched, still counts; no NBA team now): one at 0.5x; 4: free
        # agent, left out; 5: no model line; 2 also had a DNP and a game last week (period 3)
        g = logs([(1, 7, 30, 1.5), (1, 9, 30, 1.5), (2, 7, 30, 1.2), (2, 9, 0, 0.0), (2, 3, 30, 3.0), (3, 7, 30, 0.5),
                  (4, 7, 30, 1.0), (5, 7, 30, 2.0)])
        rows = LP.production_table(lv(), g, *self.week)
        self.assertEqual([r["player"] for r in rows], ["P1", "P2", "P3"])
        by = {r["player"]: r for r in rows}
        self.assertEqual((by["P1"]["g"], by["P2"]["g"], by["P3"]["g"]), (2, 1, 1))
        self.assertAlmostEqual(by["P2"]["pct"], 0.2, places=6)              # 1.2x his line on every stat
        self.assertGreater(by["P1"]["pct"], 0)
        self.assertLess(by["P3"]["pct"], 0)
        self.assertEqual({r["team"] for r in rows}, {"SVN"})
        self.assertEqual((by["P1"]["nba"], by["P3"]["nba"]), ("AAA", "—"))
        self.assertEqual(by["P1"]["s"], [60, 18, 12, 3, 3, 6, 6, 24, 48, 6, 6])
        json.dumps(rows)                                   # plain JSON

    def test_kept_until_the_game_logs_change(self):
        g = logs([(1, 7, 30, 1.5)])
        x = lv()
        first = LP.production_table(x, g, *self.week)
        self.assertIs(LP.production_table(x, g, *self.week), first)
        more = logs([(1, 7, 30, 1.5), (2, 9, 30, 1.0)])
        self.assertEqual(len(LP.production_table(x, more, *self.week)), 2)

    def test_nobody_played(self):
        self.assertEqual(LP.production_table(lv(), logs([(1, 3, 30, 1.0)]), *self.week), [])


if __name__ == "__main__":
    unittest.main()
