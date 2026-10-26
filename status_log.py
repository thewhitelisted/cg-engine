"""Log ESPN injury-status changes (DECISIONS.md S1), expected return dates (E1) and fantasy lineup changes: one row each
time a value differs from the last logged one, so the status going into any tip-off, the return ESPN expected at any
moment, and each team's lineup on any day can be rebuilt later.

    data/status_log.csv    time_utc, player_id, status        (an empty ESPN status is logged as ACTIVE)
    data/return_log.csv    time_utc, player_id, status, return_date   (ESPN's NBA injury list; ACTIVE = off the list)
    data/lineup_log.csv    time_utc, scoring_period, team_id, player_id, slot   (slot -1 = no longer on the team)

Written by publish_league.py on every run; in cg-engine the cloud loop commits the file when it changes.
"""
import csv
from datetime import datetime, timezone

import model as M

LOG = M.DATA / "status_log.csv"
RETURN_LOG = M.DATA / "return_log.csv"
LINEUP_LOG = M.DATA / "lineup_log.csv"


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


def record_lineups(lg, scoring_period, path=LINEUP_LOG, now=None):
    """Append a row for every (team, player) whose lineup slot or scoring period changed; returns rows written."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                prev[(row["team_id"], row["player_id"])] = (row["scoring_period"], row["slot"])
    cur = {(str(t["id"]), str(e[0])): (str(scoring_period), str(e[2])) for t in lg["teams"] for e in t["roster"]}
    rows = [(stamp, sp, tid, pid, slot) for (tid, pid), (sp, slot) in sorted(cur.items()) if prev.get((tid, pid)) != (sp, slot)]
    rows += [(stamp, str(scoring_period), tid, pid, "-1") for (tid, pid), (sp, slot) in sorted(prev.items())
             if (tid, pid) not in cur and slot != "-1"]
    if not rows:
        return 0
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "scoring_period", "team_id", "player_id", "slot"])
        w.writerows(rows)
    return len(rows)


def record_returns(entries, path=RETURN_LOG, now=None):
    """Append a row for every player whose listed status or expected return changed ({player_id: (status, date)} from
    live_server.injury_returns); a player who leaves the injury list is logged once as ACTIVE with no date. Returns rows
    written."""
    if not entries:                                     # list unavailable: log nothing rather than everyone recovering
        return 0
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                prev[row["player_id"]] = (row["status"], row["return_date"])
    rows = [(stamp, str(pid), st, rd) for pid, (st, rd) in sorted(entries.items()) if prev.get(str(pid)) != (st, rd)]
    listed = {str(pid) for pid in entries}
    rows += [(stamp, pid, "ACTIVE", "") for pid, (st, _) in sorted(prev.items()) if pid not in listed and st != "ACTIVE"]
    if not rows:
        return 0
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "player_id", "status", "return_date"])
        w.writerows(rows)
    return len(rows)
