"""The page's live refresh in a real DOM: renders two pages from today's real data (the second with the cards re-sorted
and one matchup's odds, totals and chart changed) and runs tests/page_refresh.test.js on them with Node + jsdom.

    CG_INTEGRATION=1 CG_NODE=~/.local/opt/node-v22.11.0-linux-x64/bin/node \
        NODE_PATH=~/.local/opt/jsdom/node_modules python -m unittest tests.test_page_refresh
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import league_page as LP  # noqa: E402


@unittest.skipUnless(os.environ.get("CG_INTEGRATION") == "1" and os.environ.get("CG_NODE"),
                     "set CG_INTEGRATION=1 and CG_NODE (with jsdom on NODE_PATH)")
class PageRefreshTest(unittest.TestCase):
    def test_in_place_refresh(self):
        import live as LV
        import live_server as S
        import serve
        import sync
        lg = sync.normalize(S.espn(serve.FULL_VIEWS))
        live, _, _ = S.live_from(S.espn(serve.LIVE_VIEWS))
        sb = S.scoreboard(datetime.now(LP.ET))
        lv = LV.Live(live=live, today_frac={k: v[1] for k, v in sb.items()}, lg=lg, started=LV.started_teams(sb))
        a = LP.build(lv, {}, sb=sb, tx=None)
        a.update(v="aaaa", ts=1000)
        b = copy.deepcopy(a)
        b["games"][0], b["games"][1] = b["games"][1], b["games"][0]          # re-sorted
        g = next(x for x in b["games"] if x["key"] == a["games"][0]["key"])  # the card whose chart the reader opens
        g["p_home"] = min(0.99, g["p_home"] + 0.07)
        g["totals"][2][1] = str(int(float(g["totals"][2][1])) + 3)           # 3PM
        g["trend"] = g["trend"] + [["2099-01-01T00:00", g["p_home"]]]
        b.update(v="bbbb", ts=2000, updated="Sun Oct 25, 11:59 PM ET")
        with tempfile.TemporaryDirectory() as d:
            files = {"A.html": LP.render(a), "B.html": LP.render(b), "B.json": json.dumps(LP.public_json(b)),
                     "OLD.json": json.dumps({**LP.public_json(a), "v": "old", "ts": 500})}
            for name, text in files.items():
                Path(d, name).write_text(text)
            r = subprocess.run([os.path.expanduser(os.environ["CG_NODE"]), str(Path(__file__).with_name("page_refresh.test.js")),
                                *(str(Path(d, n)) for n in files)], capture_output=True, text=True, timeout=120)
        print(r.stdout, r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
