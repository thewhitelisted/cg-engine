"""api.py: patches between page states (Python and web/app.js must agree) and the push stream.

    python -m unittest discover -s tests        # the browser half runs when Node is available (CG_NODE or ~/.local/opt)
"""
import copy
import gzip
import http.client
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(gzip.open(ROOT / "tests" / "fixtures" / "page_cases.json.gz", "rt").read())
NODE = os.environ.get("CG_NODE") or str(Path.home() / ".local/opt/node-v22.11.0-linux-x64/bin/node")


def mutate(state, rng):
    """A plausible next state: odds and totals move, cards re-sort, a chart point is added, sections come and go."""
    s = copy.deepcopy(state)
    for g in s["games"]:
        if rng.random() < 0.6:
            g["p_home"] = round(min(1, max(0, g["p_home"] + rng.uniform(-0.1, 0.1))), 6)
            g["totals"][rng.randrange(9)][1] = str(rng.randrange(300))
        if rng.random() < 0.3:
            g["trend"].append([f"2026-10-25T{rng.randrange(24):02d}:{rng.randrange(60):02d}", round(rng.random(), 4)])
        if rng.random() < 0.2:
            g["in_play"] = None if g["in_play"] else [["STL", "close"]]
    if rng.random() < 0.5:
        rng.shuffle(s["games"])
    if rng.random() < 0.3:
        s["moves"] = s["moves"][1:] + [{"when": "now", "kind": "free agent", "team": "X", "add": "A", "drop": ""}]
    if rng.random() < 0.2:
        s["scorecard"] = None if s.get("scorecard") else {"fav": [1, 0], "big": [0, 0], "brier": 0.2, "conf": 0.6, "n": 1}
    if rng.random() < 0.2 and s.get("today"):
        s["today"] = s["today"][:-1]
    s["updated"] = "Sun Oct 25, 11:59 PM ET"
    s["v"], s["ts"] = f"v{rng.random()}", s["ts"] + 10
    return s


class PatchTest(unittest.TestCase):
    def pairs(self, n=60):
        rng = random.Random(4)
        out = []
        for c in CASES:
            a = c["state"]
            for _ in range(n // len(CASES)):
                b = mutate(a, rng)
                out.append((a, b))
                a = b
        return out

    def test_round_trip(self):
        for a, b in self.pairs():
            ops = api.diff(api.strip(a), api.strip(b))
            self.assertEqual(api.apply(api.strip(a), ops), api.strip(b))

    def test_small_change_small_patch(self):
        a = CASES[0]["state"]
        b = copy.deepcopy(a)
        b["games"][2]["p_home"] += 0.013
        b["games"][2]["totals"][8][1] = "999"
        b["games"][0], b["games"][1] = b["games"][1], b["games"][0]
        ops = api.diff(api.strip(a), api.strip(b))
        self.assertEqual(sorted(op[0] for op in ops), ["order", "set", "set"])
        self.assertLess(len(json.dumps(ops)), 300)
        self.assertEqual(api.diff(api.strip(a), api.strip(a)), [])

    def test_type_changes_replace(self):
        self.assertEqual(api.diff({"x": 1}, {"x": 1.0}), [["set", ["x"], 1.0]])
        self.assertEqual(api.diff({"x": None}, {"x": [1]}), [["set", ["x"], [1]]])
        self.assertEqual(api.diff({"x": [1, 2, 3]}, {"x": [1, 2]}), [["set", ["x"], [1, 2]]])
        self.assertEqual(api.apply({"a": 1}, [["set", [], {"b": 2}]]), {"b": 2})

    @unittest.skipUnless(Path(NODE).exists(), "Node not available")
    def test_browser_applies_the_same(self):
        pairs = self.pairs(30)
        jobs = [{"a": api.strip(a), "ops": api.diff(api.strip(a), api.strip(b)), "b": api.strip(b)} for a, b in pairs]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(jobs, f)
        script = ("const app = require(process.argv[1]); const jobs = require(process.argv[2]);"
                  "const bad = jobs.filter(j => JSON.stringify(app.applyPatch(j.a, j.ops)) !== JSON.stringify(j.b));"
                  "console.log(bad.length); process.exit(bad.length ? 1 : 0);")
        r = subprocess.run([NODE, "-e", script, str(ROOT / "web" / "app.js"), f.name], capture_output=True, text=True)
        os.unlink(f.name)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "0"), r.stderr)


class Stream:
    """A test client for /api/events: yields (event, id, data) as they arrive."""

    def __init__(self, port, v=None, last=None):
        self.c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.c.request("GET", "/api/events" + (f"?v={v}" if v else ""), headers={"Last-Event-ID": last} if last else {})
        self.r = self.c.getresponse()

    def next(self):
        ev = {}
        while True:
            line = self.r.fp.readline().decode().rstrip("\n")
            if not line:
                if "event" in ev:
                    return ev.get("event"), ev.get("id"), ev.get("data")
                continue
            k, _, v = line.partition(": ")
            ev[k] = v

    def close(self):
        self.r.close()                      # the response holds the socket open, not the connection
        self.c.close()


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.hub = api.Hub()
        self.srv = api.start(self.hub, port=0)
        self.port = self.srv.server_address[1]
        self.addCleanup(self.srv.shutdown)
        self.s1 = {**api.strip(CASES[0]["state"]), "v": "v1", "ts": 1}
        self.s2 = {**copy.deepcopy(self.s1), "v": "v2", "ts": 2}
        self.s2["games"][0]["p_home"] = 0.42

    def wait_clients(self, n):
        for _ in range(100):
            if self.hub.count() == n:
                return
            time.sleep(0.02)
        self.fail(f"{self.hub.count()} clients, expected {n}")

    def test_state_then_patches(self):
        c = Stream(self.port)
        self.assertEqual(c.r.getheader("Content-Type"), "text/event-stream")
        self.wait_clients(1)
        self.hub.publish(self.s1)
        ev, eid, data = c.next()
        self.assertEqual((ev, eid, json.loads(data)["v"]), ("state", "v1", "v1"))
        self.hub.publish(self.s2)
        ev, eid, data = c.next()
        m = json.loads(data)
        self.assertEqual((ev, eid, m["from"], m["v"]), ("patch", "v2", "v1", "v2"))
        self.assertEqual(api.apply(api.strip(self.s1), m["ops"]), api.strip(self.s2))
        c.close()

    def test_late_joiner_gets_whole_state_only_if_behind(self):
        self.hub.publish(self.s1)
        self.hub.publish(self.s2)
        behind = Stream(self.port, v="v1")
        self.assertEqual(behind.next()[:2], ("state", "v2"))
        current = Stream(self.port, last="v2")                           # a reconnect that missed nothing
        self.wait_clients(2)
        s3 = {**copy.deepcopy(self.s2), "v": "v3", "ts": 3}
        s3["updated"] = "later"
        self.hub.publish(s3)
        self.assertEqual(current.next()[:2], ("patch", "v3"))
        behind.close()
        current.close()

    def test_ping_and_leave(self):
        with mock.patch.object(api, "PING", 0.1):
            c = Stream(self.port)
            self.assertEqual(c.next()[0], "ping")
            c.close()
            self.wait_clients(0)                                         # noticed on the next ping it can't send

    def test_slow_listener_is_dropped(self):
        q, _ = self.hub.join()
        for i in range(api.QUEUE + 1):
            self.hub.publish({**self.s1, "v": f"x{i}", "ts": i})
        self.assertEqual(self.hub.count(), 0)
        self.assertIsNone(q.get_nowait())                                # the handler sees None and closes

    def test_unknown_path(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/nope")
        self.assertEqual(c.getresponse().status, 404)

    def test_local_preview_serves_folder_pages(self):
        d = Path(tempfile.mkdtemp())
        (d / "cats").mkdir()
        (d / "index.html").write_text("league")
        (d / "cats" / "index.html").write_text("cats")
        srv = api.start(self.hub, port=0, static=d)
        self.addCleanup(srv.shutdown)
        for path, want in (("/", (200, "league")), ("/cats/", (200, "cats")), ("/cats", (200, "cats")),
                           ("/../cats/index.html", (404, None))):              # nothing outside the folder
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            c.request("GET", path)
            r = c.getresponse()
            body = r.read().decode()
            self.assertEqual((path, r.status, body if r.status == 200 else None), (path, *want))
            c.close()


if __name__ == "__main__":
    unittest.main()
