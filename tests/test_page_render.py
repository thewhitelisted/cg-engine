"""The page in the browser: web/app.js renders exactly the page the retired Python renderer produced
(tests/page_render.test.js), and the real page shell + app follow the live API in a DOM (tests/page_app.test.js, jsdom).
Runs when Node is available (CG_NODE, or ~/.local/opt); the DOM test also needs jsdom (NODE_PATH, or ~/.local/opt/jsdom).
"""
import copy
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api  # noqa: E402
import league_page as LP  # noqa: E402

HERE = Path(__file__).resolve().parent
NODE = os.environ.get("CG_NODE") or str(Path.home() / ".local/opt/node-v22.11.0-linux-x64/bin/node")
NODE_PATH = os.environ.get("NODE_PATH") or str(Path.home() / ".local/opt/jsdom/node_modules")
CASES = json.loads(gzip.open(HERE / "fixtures" / "page_cases.json.gz", "rt").read())


def node(*args):
    return subprocess.run([NODE, *map(str, args)], capture_output=True, text=True, timeout=180,
                          env={**os.environ, "NODE_PATH": NODE_PATH})


@unittest.skipUnless(Path(NODE).exists(), "Node not available")
class PageRenderTest(unittest.TestCase):
    def test_identical_to_the_python_page(self):
        r = node(HERE / "page_render.test.js")
        print(r.stdout)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


@unittest.skipUnless(Path(NODE).exists() and Path(NODE_PATH, "jsdom").exists(), "Node with jsdom not available")
class PageAppTest(unittest.TestCase):
    def test_follows_the_api_in_place(self):
        a = CASES[0]["state"]
        b = copy.deepcopy(a)
        k = b["games"][0]["key"]                                         # the card whose chart the reader opens
        b["games"][0], b["games"][1] = b["games"][1], b["games"][0]      # re-sorted
        g = next(x for x in b["games"] if x["key"] == k)
        g["p_home"] = min(0.99, g["p_home"] + 0.07)
        g["totals"][2][1] = str(int(float(g["totals"][2][1])) + 3)
        g["trend"] = g["trend"] + [["2099-01-01T00:00", g["p_home"]]]
        b.update(updated="Sun Oct 25, 11:59 PM ET", v="bbbb", ts=a["ts"] + 10)
        older = {**copy.deepcopy(a), "v": "old", "ts": a["ts"] - 10}
        with tempfile.TemporaryDirectory() as d:
            files = {"shell.html": LP.shell(), "app.js": LP.APP_JS.read_text(),
                     "cases.json": json.dumps({"a": a, "b": b, "older": older, "key": k,
                                               "ops": api.diff(api.strip(a), api.strip(b))})}
            for name, text in files.items():
                Path(d, name).write_text(text)
            r = node(HERE / "page_app.test.js", *(Path(d, n) for n in files))
        print(r.stdout, r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
