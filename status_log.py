"""Log ESPN injury-status changes (DECISIONS.md S1): one row each time a player's status differs from the last
logged one, so the status going into any tip-off can be rebuilt later and P(play | status) measured.

    data/status_log.csv    time_utc, player_id, status        (an empty ESPN status is logged as ACTIVE)

Written by publish_league.py on every run; in cg-engine the cloud loop commits the file when it changes.
"""
import csv
from datetime import datetime, timezone

import model as M

LOG = M.DATA / "status_log.csv"


def last_logged(path=LOG):
    """{player_id: last logged status}."""
    cur = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                cur[row["player_id"]] = row["status"]
    return cur


def record(statuses, path=LOG, now=None):
    """Append a row for every player whose status changed since the last logged one; returns rows written."""
    if not statuses:                                    # feed unavailable: log nothing rather than a false change
        return 0
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = last_logged(path)
    rows = [(stamp, str(pid), st or "ACTIVE") for pid, st in sorted(statuses.items())
            if prev.get(str(pid)) != (st or "ACTIVE")]
    if not rows:
        return 0
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "player_id", "status"])
        w.writerows(rows)
    return len(rows)
