"""league_page.py: the public API state (only what the pages show; versioned by it), the page shells, and the site copy.
Set CG_INTEGRATION=1 (needs data/espn_auth.json and the network) to also build today's real state and render it in Node.

    python -m unittest discover -s tests
"""
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import league_page as LP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get("CG_NODE") or str(Path.home() / ".local/opt/node-v22.11.0-linux-x64/bin/node")


def data(**over):
    """A minimal page-data dict as build() returns it, with the fields the page doesn't show."""
    d = {"period": 1, "dates": "Oct 20 – Oct 25", "start": "2026-10-20", "end": "2026-10-25", "updated": "Sun 5:00 PM",
         "v": "x", "ts": 5, "standings": [], "grid": None, "h2h": None, "allplay": [], "records": [], "top": [], "worst": [],
         "last": None, "scorecard": None, "awards": None, "ranks": None, "activity": [],
         "players": [{"player": "Q", "nba": "AAA", "team": "TT", "g": 2, "effect": 0.025, "s": [1] * 11}],
         "games": [{"key": "1-2", "home": "A", "away": "B", "home_abbr": "AA", "away_abbr": "BB", "p_home": np.float64(0.25),
                    "score": [np.int64(3), 6, 0], "games": [(4, 5), (6, 7)], "totals": [], "lead": [], "swing": None,
                    "in_play": [], "trend": [["2026-10-21T12:00", 0.123456]],
                    "lean": [("FG%", np.float64(0.4)), ("FT%", 0.123456)]}],
         "today": [{"team": "A", "n": 1, "lineup": 1, "players": [{"name": "P", "when": "7:00p", "order": 3,
                                                                     "bench": False, "out": False}]}],
         "moves": [{"ts": 123, "when": "Sun", "kind": "free agent", "team": "A", "add": "X", "drop": ""}]}
    d.update(over)
    return d


class ContentKeyTest(unittest.TestCase):
    def test_ignores_build_stamps(self):
        a = {"games": [1], "updated": "Sun Oct 25, 5:00 PM", "v": "x", "ts": 1}
        b = {"games": [1], "updated": "Sun Oct 25, 5:01 PM", "v": "y", "ts": 2}
        self.assertEqual(LP.content_key(a), LP.content_key(b))
        self.assertNotEqual(LP.content_key(a), LP.content_key({**a, "games": [2]}))


class ApiStateTest(unittest.TestCase):
    def test_only_what_the_page_shows(self):
        s = LP.api_state(data())
        g = s["games"][0]
        self.assertNotIn("lean", g)
        self.assertEqual(g["p_cats"], [0.4, 0.1235])                      # the category page's chances, in CATS order
        self.assertNotIn("order", s["today"][0]["players"][0])
        self.assertNotIn("ts", s["moves"][0])
        self.assertEqual(g["trend"], [["2026-10-21T12:00", 0.1235]])
        self.assertEqual(json.loads(json.dumps(s)), s)                    # plain JSON (numpy values converted)
        self.assertEqual((g["p_home"], g["score"][0], g["games"]), (0.25, 3, [[4, 5], [6, 7]]))
        self.assertEqual(s["players"], data()["players"])                 # the players page's list, as built

    def test_version_follows_what_is_shown(self):
        base = LP.api_state(data())["v"]
        hidden = data()
        hidden["moves"][0]["ts"] = 456
        self.assertEqual(LP.api_state(hidden)["v"], base)                 # a hidden change: same version, no update
        tiny = data()
        tiny["games"][0]["lean"][1] = ("FT%", 0.12346)
        self.assertEqual(LP.api_state(tiny)["v"], base)                   # below what's published
        cats = data()
        cats["games"][0]["lean"][0] = ("FG%", 0.41)
        self.assertNotEqual(LP.api_state(cats)["v"], base)                # a category chance is shown (category page)
        players = data()
        players["players"][0]["g"] = 3
        self.assertNotEqual(LP.api_state(players)["v"], base)             # the players page is shown too
        self.assertEqual(LP.api_state(data(ts=99))["v"], base)
        shown = data()
        shown["games"][0]["p_home"] = 0.3
        self.assertNotEqual(LP.api_state(shown)["v"], base)
        self.assertNotEqual(LP.api_state(data(updated="Sun 5:01 PM"))["v"], LP.api_state(data())["v"] + "z")


class LastWeekTest(unittest.TestCase):
    def test_tie(self):
        """ESPN writes a tied week as winner "TIE" (matchup 1: Joey's Toes 4-4-1 Thinny Thongs)."""
        lg = {"completed_weeks": [1], "teams": [{"id": 11, "name": "Joey"}, {"id": 1, "name": "Thinny"},
                                                {"id": 5, "name": "Nathan"}, {"id": 13, "name": "Chrissy"}],
              "results": [[1, 11, 1, "TIE", 4, 4, 1, {}], [1, 5, 13, "HOME", 5, 4, 0, {}]]}
        rows = LP.last_week(lg, {"1": {"11-1": 0.253, "5-13": 0.513}})["rows"]
        self.assertEqual([(r["tie"], r["home_won"]) for r in rows], [(True, False), (False, True)])


class ShellTest(unittest.TestCase):
    def test_shell(self):
        s = LP.shell()
        self.assertNotIn('http-equiv="refresh"', s)                       # no timed full reloads
        self.assertIn('<script src="app.js"></script>', s)
        self.assertIn(f'window.CG_API = window.CG_API ?? "{LP.LIVE_URL}"', s)
        self.assertIn("function initNav()", s)
        self.assertIn("CG.initCharts = ", s)
        self.assertIn('<div class="wrap">', s)
        self.assertNotIn("%(", s)
        self.assertIsNone(re.search(r"playoff odds|title odds|champion(ship)? odds", s + LP.APP_JS.read_text(), re.I))
        self.assertIn(LP.STYLE, s)

    def test_cats_page(self):
        s = LP.cats_page()
        self.assertNotIn('http-equiv="refresh"', s)
        self.assertIn('<script src="../app.js"></script>', s)             # one folder down: /cats/
        self.assertIn('<body data-page="cats" data-root="../">', s)
        self.assertIn(f'window.CG_API = window.CG_API ?? "{LP.LIVE_URL}"', s)
        self.assertIn(LP.STYLE, s)                                        # the league page's look
        self.assertIn('<div class="wrap">', s)
        self.assertNotIn("%(", s)
        self.assertIsNone(re.search(r"playoff|title odds|champion", s, re.I))

    def test_players_page(self):
        s = LP.players_page()
        self.assertIn('<script src="../app.js"></script>', s)             # one folder down: /players/
        self.assertIn('<body data-page="players" data-root="../">', s)
        self.assertIn(f'window.CG_API = window.CG_API ?? "{LP.LIVE_URL}"', s)
        self.assertIn(LP.STYLE, s)
        self.assertIn('<meta name="robots" content="noindex, nofollow">', s)
        self.assertNotIn("%(", s)
        self.assertIsNone(re.search(r"playoff|title odds|champion", s, re.I))

    def test_site_files(self):
        self.assertEqual(sorted(LP.site_files()), ["app.js", "cats/index.html", "index.html", "insights.html",
                                                   "players/index.html"])
        files = LP.site_files({"v": "x"})
        self.assertEqual(json.loads(files["api/state.json"]), {"v": "x"})


@unittest.skipUnless(os.environ.get("CG_INTEGRATION") == "1", "set CG_INTEGRATION=1 to build the real state")
class RealStateTest(unittest.TestCase):
    def test_build_state_and_render(self):
        import inseason as I
        import live as LV
        import live_server as S
        import serve
        import sync
        lg = sync.normalize(S.espn(serve.FULL_VIEWS))
        live, _, _ = S.live_from(S.espn(serve.LIVE_VIEWS))
        sb = S.scoreboard(I.game_day())                                   # the day serve.py reads
        lv = LV.Live(live=live, today_frac={k: v[1] for k, v in sb.items()}, lg=lg, started=LV.started_teams(sb))
        history = {}
        state = LP.api_state(LP.build(lv, history, sb=sb, tx=None))
        again = LP.api_state(LP.build(lv, json.loads(json.dumps(history)), sb=sb, tx=None))
        self.assertEqual(again["v"], state["v"])                          # same content, same version
        for g in state["games"]:                                          # each matchup's category chances, as computed
            pc, pw, _, _ = lv.matchup(*map(int, g["key"].split("-")))
            self.assertEqual(g["p_cats"], [round(float(p), 4) for p in pc])
            self.assertAlmostEqual(g["p_home"], pw)
        if Path(NODE).exists():
            for fn in ("render", "renderCats"):
                script = f"const app = require(process.argv[1]); process.stdout.write(app.{fn}(JSON.parse(process.argv[2])));"
                html = subprocess.run([NODE, "-e", script, str(ROOT / "web" / "app.js"), json.dumps(state)],
                                      capture_output=True, text=True, check=True).stdout
                self.assertNotIn("%(", html)
                for t in lg["teams"]:
                    self.assertIn(t["name"].replace("'", "&#x27;"), html)

    def test_serve_league_snapshot_matches_publish(self):
        """serve.py's full refresh must see the same league as publish_league.py (week_totals needs mScoreboard)."""
        import live_server as S
        import serve
        import sync
        a = sync.normalize(S.espn(serve.FULL_VIEWS))
        b = sync.normalize(S.espn(["mTeam", "mRoster", "mMatchupScore", "mScoreboard", "mSettings", "mStatus"]))
        a.pop("generated"), b.pop("generated")
        self.assertTrue(a["week_totals"])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
