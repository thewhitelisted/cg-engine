"""Decide whether this run should publish (stdlib only, runs before installing anything).

Runs: from 30 min before today's first NBA tip-off until 3 h after the last one (late games carry past
midnight), plus one morning refresh (9:00-9:29 ET), plus every manual run. If the scoreboard can't be read,
it runs anyway. Skips when an earlier run of this workflow is still going (loop.py keeps one job publishing
through the game window). Writes run=true/false to $GITHUB_OUTPUT.
"""
import json
import os
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
BEFORE, AFTER = timedelta(minutes=30), timedelta(hours=3)


def tipoffs(day):
    url = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=" + day.strftime("%Y%m%d"))
    with urllib.request.urlopen(url, timeout=20) as r:
        events = json.load(r).get("events", [])
    return [datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(ET) for e in events]


def game_window(now):
    try:
        for day in (now.date(), now.date() - timedelta(days=1)):     # yesterday: late games past midnight
            t = tipoffs(datetime.combine(day, datetime.min.time()))
            if t and min(t) - BEFORE <= now <= max(t) + AFTER:
                return True, f"game window {min(t):%-I:%M %p}-{(max(t) + AFTER):%-I:%M %p} ET"
        return False, "outside game hours"
    except Exception as e:
        return True, f"scoreboard unavailable ({type(e).__name__}); running anyway"


def earlier_run_active():
    """True if an older run of this workflow is still in progress (needs GITHUB_TOKEN with actions: read)."""
    token, repo, me = (os.environ.get(k) for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    if not (token and repo and me):
        return False
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/workflows/publish.yml/runs?status=in_progress",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            runs = json.load(r).get("workflow_runs", [])
    except Exception:
        return False
    return any(r["id"] < int(me) for r in runs)


def decide(now):
    if earlier_run_active():
        return False, "an earlier run is still publishing"
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return True, "manual run"
    if now.hour == 9 and now.minute < 30:
        return True, "morning refresh"
    return game_window(now)


if __name__ == "__main__":
    run, why = decide(datetime.now(ET))
    print(f"{datetime.now(ET):%a %H:%M} ET: {'run' if run else 'skip'} ({why})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
