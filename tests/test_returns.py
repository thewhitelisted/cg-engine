"""The expected-return log (DECISIONS.md E1): ESPN's NBA injury list parsed into {player: (status, return date)}, logged
as changes only, with a player leaving the list logged once as ACTIVE; and serve.py keeps refreshing when the feed fails.

    python -m unittest discover -s tests
"""
import csv
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_server as S  # noqa: E402
import serve  # noqa: E402
import status_log  # noqa: E402


def entry(pid, status, date, name="X"):
    return {"athlete": {"displayName": name, "links": [{"href": f"https://www.espn.com/nba/player/_/id/{pid}/x"}]},
            "status": status, "details": {"returnDate": date} if date else {}}


FEED = {"injuries": [{"displayName": "76ers", "injuries": [entry(3059318, "Out", "2026-10-28", "Joel Embiid"),
                                                            entry(5124612, "Day-To-Day", "2026-10-27")]},
                     {"displayName": "Suns", "injuries": [entry(4601023, "Out", None),
                                                          {"athlete": {"displayName": "no link"}, "status": "Out"}]}]}


class FeedTest(unittest.TestCase):
    def test_parse(self):
        resp = mock.Mock(json=lambda: FEED, raise_for_status=lambda: None)
        with mock.patch.object(S.requests, "get", return_value=resp):
            got = S.injury_returns()
        self.assertEqual(got, {3059318: ("OUT", "2026-10-28"), 5124612: ("DAY_TO_DAY", "2026-10-27"), 4601023: ("OUT", "")})


class LogTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "return_log.csv"
        self.t = 0

    def record(self, entries):
        self.t += 1
        return status_log.record_returns(entries, self.path, now=datetime(2026, 10, 26, 13, self.t, tzinfo=timezone.utc))

    def rows(self):
        with self.path.open(newline="") as f:
            return [(r["player_id"], r["status"], r["return_date"]) for r in csv.DictReader(f)]

    def test_changes_only(self):
        self.assertEqual(self.record({1: ("OUT", "2026-10-28"), 2: ("DAY_TO_DAY", "2026-10-27")}), 2)
        self.assertEqual(self.record({1: ("OUT", "2026-10-28"), 2: ("DAY_TO_DAY", "2026-10-27")}), 0)
        self.assertEqual(self.record({1: ("OUT", "2026-10-30"), 2: ("DAY_TO_DAY", "2026-10-27")}), 1)   # date moved
        self.assertEqual(self.rows()[-1], ("1", "OUT", "2026-10-30"))

    def test_leaving_the_list_is_logged_once(self):
        self.record({1: ("OUT", "2026-10-28"), 2: ("OUT", "")})
        self.assertEqual(self.record({2: ("OUT", "")}), 1)
        self.assertEqual(self.rows()[-1], ("1", "ACTIVE", ""))
        self.assertEqual(self.record({2: ("OUT", "")}), 0)
        self.assertEqual(self.record({1: ("OUT", "2026-11-02"), 2: ("OUT", "")}), 1)                 # back on it

    def test_no_feed_logs_nothing(self):
        self.record({1: ("OUT", "2026-10-28")})
        self.assertEqual(self.record({}), 0)
        self.assertEqual(len(self.rows()), 1)

    def test_header(self):
        self.record({1: ("OUT", "2026-10-28")})
        self.assertEqual(self.path.read_text().splitlines()[0], "time_utc,player_id,status,return_date")


class ServeTest(unittest.TestCase):
    """full_refresh logs returns when publishing for real, and a failing feed doesn't stop it."""

    def setUp(self):
        stubs = {(serve.sync, "normalize"): lambda j: {"teams": []},
                 (serve.S, "transactions"): lambda sp, back: [],
                 (serve.I, "all_statuses"): lambda: {},
                 (serve.status_log, "record"): lambda st: 0,
                 (serve.status_log, "record_lineups"): lambda lg, sp: 0}
        for (mod, name), fn in stubs.items():
            p = mock.patch.object(mod, name, fn)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(serve, "M", mock.Mock(DATA=Path(tempfile.mkdtemp())))
        p.start()
        self.addCleanup(p.stop)
        self.pub = serve.Publisher(Path(tempfile.mkdtemp()), dry_run=True)
        self.pub.dry = False
        self.pub.full_feed = mock.Mock(get=lambda: ({}, True))

    def test_logged(self):
        rec = mock.Mock(return_value=1)
        with mock.patch.object(serve.S, "injury_returns", return_value={1: ("OUT", "2026-10-28")}), \
                mock.patch.object(serve.status_log, "record_returns", rec):
            self.pub.full_refresh({"latest_scoring_period": 7})
        rec.assert_called_once_with({1: ("OUT", "2026-10-28")})

    def test_feed_failure_does_not_stop_the_refresh(self):
        with mock.patch.object(serve.S, "injury_returns", side_effect=OSError("down")):
            self.pub.full_refresh({"latest_scoring_period": 7})
        self.assertIsNotNone(self.pub.lg)


if __name__ == "__main__":
    unittest.main()
