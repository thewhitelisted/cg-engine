"""serve.py: conditional ESPN requests, lineup-change detection, when a tick recomputes and writes, alerts, and code
updates. ESPN, the model and the page builder are stubbed; nothing here touches the network.

    python -m unittest discover -s tests
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve  # noqa: E402


def live_json(lineups, period=1):
    """A minimal ESPN live feed (the shape of mMatchupScore + mScoreboard + mStatus): status plus this period's
    matchups with rosterForCurrentScoringPeriod."""
    teams = list(lineups)
    sched = []
    for h, a in zip(teams[::2], teams[1::2]):
        side = lambda t: {"teamId": t, "rosterForCurrentScoringPeriod": {"entries": [
            {"playerId": p, "lineupSlotId": s, "playerPoolEntry": {"player": {"id": p, "injuryStatus": st}}}
            for p, s, st in lineups[t]]}}
        sched.append({"matchupPeriodId": period, "home": side(h), "away": side(a)})
    sched.append({"matchupPeriodId": period + 1, "home": {"teamId": 99}, "away": {"teamId": 98}})   # another week: ignored
    return {"status": {"currentMatchupPeriod": period, "latestScoringPeriod": 6}, "schedule": sched}


def league(lineups):
    return {"teams": [{"id": t, "name": f"T{t}", "roster": [[p, f"P{p}", s, st] for p, s, st in lu]} for t, lu in lineups.items()]}


LINEUPS = {1: [(10, 0, "ACTIVE"), (11, 12, "ACTIVE")], 2: [(20, 4, "DAY_TO_DAY"), (21, 13, "OUT")]}


class Response:
    def __init__(self, status, body=None, etag=None):
        self.status_code, self.body, self.headers = status, body, {"ETag": etag} if etag else {}

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FeedTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(serve.S, "cookies", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_conditional_requests(self):
        session = mock.Mock()
        session.get.side_effect = [Response(200, {"a": 1}, "e1"), Response(304), Response(200, {"a": 2}, "e2")]
        feed = serve.Feed(session, ["mStatus"])
        self.assertEqual(feed.get(), ({"a": 1}, True))
        self.assertEqual(session.get.call_args_list[0].kwargs["headers"], {})
        self.assertEqual(feed.get(), ({"a": 1}, False))                      # 304: same body, nothing new
        self.assertEqual(session.get.call_args_list[1].kwargs["headers"], {"If-None-Match": "e1"})
        self.assertEqual(feed.get(), ({"a": 2}, True))

    def test_expired_cookies(self):
        session = mock.Mock()
        session.get.return_value = Response(401)
        with self.assertRaisesRegex(RuntimeError, "401"):
            serve.Feed(session, ["mStatus"]).get()


class LineupTest(unittest.TestCase):
    def test_same_lineups(self):
        self.assertEqual(serve.lineups_live(live_json(LINEUPS)), {1: sorted(LINEUPS[1]), 2: sorted(LINEUPS[2])})
        self.assertFalse(serve.lineups_differ(serve.lineups_live(live_json(LINEUPS)), league(LINEUPS)))

    def test_slot_move_add_drop_and_status(self):
        for change in ({1: [(10, 12, "ACTIVE"), (11, 0, "ACTIVE")]}, {1: [*LINEUPS[1], (12, 12, "ACTIVE")]},
                       {2: [(20, 4, "DAY_TO_DAY")]}, {2: [(20, 4, "OUT"), (21, 13, "OUT")]}):
            with self.subTest(change=change):
                self.assertTrue(serve.lineups_differ(serve.lineups_live(live_json({**LINEUPS, **change})), league(LINEUPS)))

    def test_write_atomic(self):
        d = Path(tempfile.mkdtemp())
        serve.write_atomic(d / "data.json", "x")
        serve.write_atomic(d / "data.json", "y")
        self.assertEqual((d / "data.json").read_text(), "y")
        self.assertEqual([p.name for p in d.iterdir()], ["data.json"])     # no temp file left behind


class TickTest(unittest.TestCase):
    """Publisher.tick with ESPN, the model and the page builder stubbed."""

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.lineups = dict(LINEUPS)
        self.score = 0
        self.content = "a"
        self.built = []
        stubs = {
            (serve.S, "live_from"): lambda j: ({"period": 1, "latest_scoring_period": 6, "matchups": [self.score]}, [], []),
            (serve.S, "scoreboard"): lambda day: {},
            (serve.S, "transactions"): lambda sp, back: [],
            (serve.sync, "normalize"): lambda j: league(self.lineups),
            (serve.LV, "Live"): mock.Mock(side_effect=lambda **kw: mock.Mock(period=1)),
            (serve.LP, "build"): self.build,
            (serve.LP, "render"): lambda data: f"<html>{data['v']}</html>",
            (serve.LP, "public_json"): lambda data: {"v": data["v"]},
        }
        for (mod, name), fn in stubs.items():
            p = mock.patch.object(mod, name, fn)
            p.start()
            self.addCleanup(p.stop)
        self.live_cls = serve.LV.Live
        self.pub = serve.Publisher(self.out, dry_run=True)
        self.pub.live_feed = mock.Mock(get=lambda: (live_json(self.lineups), True))
        self.pub.full_feed = mock.Mock(get=lambda: ({}, True))

    def build(self, lv, history, sb, tx):
        self.built.append(lv)
        if getattr(self, "new_point", False):                          # LP.build adds a chart point when odds move
            history.setdefault("trend", {}).setdefault("1", []).append(["now", {}])
        return {"games": [], "content": self.content, "updated": "now", "v": self.content, "ts": 0}

    def test_first_tick_builds_and_writes(self):
        what = self.pub.tick()
        self.assertTrue(what.startswith("full refresh"))
        self.assertEqual((self.out / "data.json").read_text(), '{"v": "a"}')
        self.assertEqual((self.out / "index.html").read_text(), "<html>a</html>")

    def test_nothing_changed_skips_the_model(self):
        self.pub.tick()
        self.assertIsNone(self.pub.tick())
        self.assertEqual(len(self.built), 1)

    def test_new_scores_use_the_cheap_refresh(self):
        self.pub.tick()
        lv = self.pub.lv
        self.score = 1
        self.content = "b"
        self.assertTrue(self.pub.tick().startswith("live refresh"))
        self.assertIs(self.pub.lv, lv)                                     # same model, updated in place
        lv.update.assert_called_once()
        self.assertEqual((self.out / "data.json").read_text(), '{"v": "b"}')

    def test_same_page_is_not_rewritten(self):
        self.pub.tick()
        before = (self.out / "data.json").stat().st_mtime_ns
        self.score = 1                                               # inputs moved, page didn't
        self.assertIsNone(self.pub.tick())
        self.assertEqual(len(self.built), 2)
        self.assertEqual((self.out / "data.json").stat().st_mtime_ns, before)

    def test_lineup_change_forces_a_full_refresh_after_the_gap(self):
        self.pub.tick()
        self.lineups = {**LINEUPS, 1: [(10, 12, "ACTIVE"), (11, 0, "ACTIVE")]}   # live feed shows a slot move
        self.score = 1
        self.pub.tick()
        self.assertEqual(self.live_cls.call_count, 1)                      # within MIN_FULL_GAP: no rebuild yet
        self.pub.full_at -= serve.MIN_FULL_GAP
        self.score = 2
        self.content = "b"
        self.assertTrue(self.pub.tick().startswith("full refresh"))
        self.assertEqual(self.live_cls.call_count, 2)

    def test_archives_only_new_chart_points(self):
        self.pub.archive = mock.Mock()
        self.pub.tick()
        self.pub.archive.assert_not_called()
        self.new_point, self.score, self.content = True, 1, "b"
        self.pub.tick()
        self.pub.archive.assert_called_once()

    def test_full_refresh_on_schedule(self):
        self.pub.tick()
        self.pub.full_at -= serve.FULL_EVERY
        self.content = "b"
        self.assertTrue(self.pub.tick().startswith("full refresh"))


class FakeLive:
    """matchup() gives L4 0.4 / K1 0.7 for the week, 0.5 / 0.6 per category."""
    locks = False

    def set_locks(self, on):
        self.locks = on

    def matchup(self, home, away):
        return [0.6 if self.locks else 0.5] * 9, 0.7 if self.locks else 0.4, None, None


class ArchiveTest(unittest.TestCase):
    def test_both_rules_recorded_and_published_rule_restored(self):
        d = Path(tempfile.mkdtemp())
        with mock.patch.object(serve, "REPLAY_DIR", d):
            pub = serve.Publisher(d / "out", dry_run=True)
            pub.dry, pub.lv, pub.lg = False, FakeLive(), league(LINEUPS)
            live = {"period": 1, "latest_scoring_period": 6,
                    "matchups": [{"home": 1, "away": 2, "home_tot": {"pts": 1}, "away_tot": {"pts": 2}}]}
            pub.archive(live, {1: 0.5}, {1}, {10: 0, 11: 12})
            pub.archive(live, {1: 0.25}, {1}, {10: 0, 11: 12})
        files = list(d.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        recs = [json.loads(x) for x in files[0].read_text().splitlines()]
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["odds"]["L4"]["1-2"], [0.4] + [0.5] * 9)
        self.assertEqual(recs[0]["odds"]["K1"]["1-2"], [0.7] + [0.6] * 9)
        self.assertEqual((recs[1]["frac"], recs[0]["started"], recs[0]["slots"]), ({"1": 0.25}, [1], {"10": 0, "11": 12}))
        self.assertEqual(recs[0]["published"], "K1" if serve.LV.LINEUP_LOCKS else "L4")
        self.assertEqual(pub.lv.locks, serve.LV.LINEUP_LOCKS)             # the published rule is back on


class AlertTest(unittest.TestCase):
    def setUp(self):
        self.state, self.saves = {}, 0
        self.calls = []
        self.now = 1000.0
        p = mock.patch.object(serve.time, "time", lambda: self.now)
        p.start()
        self.addCleanup(p.stop)

    def alert(self):
        def save():
            self.saves += 1
        a = serve.Alert(self.state, save)

        def gh(*args):
            self.calls.append(args)
            return subprocess.CompletedProcess(args, 0, "https://github.com/x/y/issues/7\n", "")
        a.gh = gh
        return a

    def test_opens_once_after_threshold_and_closes_on_recovery(self):
        a = self.alert()
        a.update("tick", False, "Traceback\nRuntimeError: ESPN down")
        self.now += serve.ALERT_AFTER["tick"] - 1
        a.update("tick", False, "RuntimeError: ESPN down")
        self.assertEqual(self.calls, [])                                   # not yet
        self.now += 1
        a.update("tick", False, "RuntimeError: ESPN down")
        a.update("tick", False, "RuntimeError: ESPN down")
        self.assertEqual([c[:2] for c in self.calls], [("issue", "create")])
        self.assertIn("ESPN down", self.calls[0][3])
        self.assertEqual(self.state["issues"]["tick"], "7")
        self.alert().update("tick", False, "x")                           # a restart doesn't open a second one
        self.assertEqual(len(self.calls), 1)
        a.update("tick", True)
        self.assertEqual(self.calls[-1][:3], ("issue", "close", "7"))
        self.assertIsNone(self.state["issues"]["tick"])

    def test_brief_failures_never_alert(self):
        a = self.alert()
        for _ in range(3):
            a.update("tick", False, "x")
            self.now += serve.ALERT_AFTER["tick"] / 2
            a.update("tick", True)
        self.assertEqual(self.calls, [])

    def test_disabled_in_dry_runs(self):
        a = serve.Alert(self.state, lambda: None, enabled=False)
        a.gh = mock.Mock()
        a.update("tick", False, "x")
        self.now += 10 * serve.ALERT_AFTER["tick"]
        a.update("tick", False, "x")
        a.gh.assert_not_called()


class CodeUpdateTest(unittest.TestCase):
    """code_changed() against a throwaway upstream: log commits don't restart, code commits do."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                              capture_output=True, text=True)

    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.up, self.work, self.other = root / "up.git", root / "work", root / "other"
        self.git(root, "init", "-q", "--bare", "-b", "main", str(self.up))
        self.git(root, "clone", "-q", str(self.up), str(self.other))
        (self.other / "data").mkdir()
        (self.other / "a.py").write_text("x = 1\n")
        (self.other / "data" / "status_log.csv").write_text("t\n")
        self.git(self.other, "add", ".")
        self.git(self.other, "commit", "-q", "-m", "init")
        self.git(self.other, "push", "-q", "origin", "HEAD:main")
        self.git(root, "clone", "-q", str(self.up), str(self.work))
        self.cwd = Path.cwd()
        import os
        os.chdir(self.work)
        self.addCleanup(os.chdir, self.cwd)

    def push(self, path, text):
        f = self.other / path
        f.write_text(text)
        self.git(self.other, "commit", "-qam", path)
        self.git(self.other, "push", "-q", "origin", "HEAD:main")

    def test_logs_only(self):
        self.push("data/status_log.csv", "t\n1\n")
        self.assertFalse(serve.code_changed())
        (self.other / "data" / "replay").mkdir()
        (self.other / "data" / "replay" / "2026-10-25.jsonl").write_text("{}\n")
        self.git(self.other, "add", ".")
        self.git(self.other, "commit", "-qm", "replay log")
        self.git(self.other, "push", "-q", "origin", "HEAD:main")
        self.assertFalse(serve.code_changed())                             # replay logs don't restart either
        self.assertEqual((self.work / "data" / "status_log.csv").read_text(), "t\n1\n")

    def test_code(self):
        self.push("a.py", "x = 2\n")
        self.assertTrue(serve.code_changed())
        self.assertFalse(serve.code_changed())                             # nothing newer the second time


if __name__ == "__main__":
    unittest.main()
