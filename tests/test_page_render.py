"""The page in the browser: web/app.js renders exactly the page the retired Python renderer produced
(tests/page_render.test.js), and the real page shell + app follow the live API in a DOM (tests/page_app.test.js, jsdom);
the category page draws every matchup's category chances and follows the API the same way (tests/cats_page.test.js),
and so does the players page with every player's week against the model's line (tests/players_page.test.js).
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


def with_cat_chances(state):
    """The state with each matchup's category chances (p_cats, as api_state publishes them): fixed made-up values."""
    s = copy.deepcopy(state)
    for j, g in enumerate(s["games"]):
        g["p_cats"] = [round((0.5 + 0.45 * ld if ld else 0.5) * 0.9 + 0.05 * ((i + j) % 3), 4)
                       for i, ld in enumerate(g["lead"])]
    return s


@unittest.skipUnless(Path(NODE).exists() and Path(NODE_PATH, "jsdom").exists(), "Node with jsdom not available")
class CatsPageTest(unittest.TestCase):
    def test_every_matchups_category_chances(self):
        a = with_cat_chances(CASES[0]["state"])
        b = copy.deepcopy(a)
        b["games"] = b["games"][::-1]                                    # the league page's order changed
        g = b["games"][0]
        k = g["key"]
        g["p_cats"][3] = round(1 - g["p_cats"][3], 4)                    # REB swings to the other side
        g["p_cats"][8] = 0.5
        g["totals"][3][1] = str(int(g["totals"][3][1]) + 12)
        g["lead"][3] = 1
        g["p_home"] = min(0.99, g["p_home"] + 0.07)
        b.update(updated="Sun Oct 25, 11:59 PM ET", v="bbbb", ts=a["ts"] + 10)
        with tempfile.TemporaryDirectory() as d:
            files = {"cats.html": LP.cats_page(), "app.js": LP.APP_JS.read_text(),
                     "cases.json": json.dumps({"a": a, "b": b, "old": CASES[1]["state"], "key": k,
                                               "ops": api.diff(api.strip(a), api.strip(b))})}
            for name, text in files.items():
                Path(d, name).write_text(text)
            r = node(HERE / "cats_page.test.js", *(Path(d, n) for n in files))
        print(r.stdout, r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()


def with_players(state):
    """The state with a players list (as api_state publishes it): made-up rows covering a free agent, no free throws
    and an unknown vs expected."""
    s = copy.deepcopy(state)
    s["players"] = [{"player": f"Player {i}", "nba": ["LAL", "BOS", "UTAH"][i % 3], "team": None if i % 4 == 0 else f"T{i % 5}",
                     "g": 1 + i % 3, "pct": None if i == 7 else round(0.9 - 0.13 * i, 4),
                     "s": [20 + i, 5, 4, 1, i % 2, 2, 3, 8, 15 + i, 0 if i % 5 == 0 else 3, 0 if i % 5 == 0 else 4]}
                    for i in range(12)]
    return s


@unittest.skipUnless(Path(NODE).exists() and Path(NODE_PATH, "jsdom").exists(), "Node with jsdom not available")
class PlayersPageTest(unittest.TestCase):
    def test_every_players_week(self):
        a = with_players(CASES[0]["state"])
        b = copy.deepcopy(a)
        b["players"][0], b["players"][1] = b["players"][1], b["players"][0]     # re-ranked
        b["players"][1]["s"][0] += 10
        b["players"] = b["players"] + [{**b["players"][2], "player": "New Guy"}]
        b.update(updated="Sun Oct 25, 11:59 PM ET", v="pppp", ts=a["ts"] + 10)
        with tempfile.TemporaryDirectory() as d:
            files = {"players.html": LP.players_page(), "app.js": LP.APP_JS.read_text(),
                     "cases.json": json.dumps({"a": a, "b": b, "ops": api.diff(api.strip(a), api.strip(b))})}
            for name, text in files.items():
                Path(d, name).write_text(text)
            r = node(HERE / "players_page.test.js", *(Path(d, n) for n in files))
        print(r.stdout, r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
