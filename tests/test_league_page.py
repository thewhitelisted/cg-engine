"""league_page.py: the content key and version stamps that drive the page's live refresh, and the refresh script's
placeholders. Set CG_INTEGRATION=1 (needs data/espn_auth.json and the network) to also build and render the real page.

    python -m unittest discover -s tests
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import league_page as LP  # noqa: E402


class ContentKeyTest(unittest.TestCase):
    def test_ignores_build_stamps(self):
        a = {"games": [1], "updated": "Sun 5:00 PM", "v": "x", "ts": 1}
        b = {"games": [1], "updated": "Sun 5:01 PM", "v": "y", "ts": 2}
        self.assertEqual(LP.content_key(a), LP.content_key(b))
        self.assertNotEqual(LP.content_key(a), LP.content_key({**a, "games": [2]}))

    def test_public_json_carries_the_stamps(self):
        data = {"period": 1, "dates": "Oct 20 – Oct 25", "updated": "u", "v": "abc", "ts": 5,
                "games": [{"home": "A", "away": "B", "p_home": 0.5, "score": [1, 2, 0], "lean": []}]}
        pj = LP.public_json(data)
        self.assertEqual((pj["v"], pj["ts"]), ("abc", 5))
        self.assertEqual(set(pj["games"][0]), {"home", "away", "p_home", "score"})   # nothing beyond what was public


class PageScriptTest(unittest.TestCase):
    def test_refresh_script(self):
        script = LP.PAGE[LP.PAGE.index("// live refresh"):]
        for ph in ("%(live_url)s", "%(poll_ms)s"):
            self.assertIn(ph, script)
        self.assertIn('data-v="%(v)s" data-ts="%(ts)s"', LP.PAGE)
        self.assertIn("(d.ts || 0) < cur.ts", script)                     # an older copy never replaces newer numbers
        self.assertIn("d.v === cur.v", script)                             # refresh on any new content, not per minute
        self.assertTrue(LP.LIVE_URL.startswith("https://") and LP.LIVE_URL.endswith("/"))

    def test_no_playoff_or_title_odds(self):
        self.assertIsNone(re.search(r"playoff odds|title odds|champion(ship)? odds", LP.PAGE, re.I))


@unittest.skipUnless(os.environ.get("CG_INTEGRATION") == "1", "set CG_INTEGRATION=1 to build the real page")
class RealPageTest(unittest.TestCase):
    def test_build_and_render(self):
        from datetime import datetime
        import live as LV
        import live_server as S
        import serve
        import sync
        lg = sync.normalize(S.espn(serve.FULL_VIEWS))
        live, _, _ = S.live_from(S.espn(serve.LIVE_VIEWS))
        sb = S.scoreboard(datetime.now(LP.ET))
        lv = LV.Live(live=live, today_frac={k: v[1] for k, v in sb.items()}, lg=lg)
        history = {}
        data = LP.build(lv, history, sb=sb, tx=None)
        page = LP.render(data)
        self.assertNotIn("%(", page)                                      # every placeholder filled
        self.assertIn(f'data-v="{data["v"]}"', page)
        self.assertIn(f'const LIVE = "{LP.LIVE_URL}"', page)
        self.assertIn(f"setInterval(check, {LP.POLL_MS})", page)
        self.assertEqual(json.loads(json.dumps(LP.public_json(data)))["v"], data["v"])
        again = LP.build(lv, json.loads(json.dumps(history)), sb=sb, tx=None)
        self.assertEqual(again["v"], data["v"])                           # same content, same version

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
