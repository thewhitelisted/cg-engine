"""Decide whether this scheduled run should publish (stdlib only, runs before installing anything).

Runs: from 30 min before today's first NBA tip-off until 3 h after the last one (late games carry past
midnight), plus one morning refresh (9:00-9:29 ET), plus every manual run. If the scoreboard can't be read,
it runs anyway. Writes run=true/false to $GITHUB_OUTPUT.
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


def decide(now):
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return True, "manual run"
    if now.hour == 9 and now.minute < 30:
        return True, "morning refresh"
    try:
        for day in (now.date(), now.date() - timedelta(days=1)):     # yesterday: late games past midnight
            t = tipoffs(datetime.combine(day, datetime.min.time()))
            if t and min(t) - BEFORE <= now <= max(t) + AFTER:
                return True, f"game window {min(t):%-I:%M %p}-{(max(t) + AFTER):%-I:%M %p} ET"
        return False, "outside game hours"
    except Exception as e:
        return True, f"scoreboard unavailable ({type(e).__name__}); running anyway"


if __name__ == "__main__":
    run, why = decide(datetime.now(ET))
    print(f"{datetime.now(ET):%a %H:%M} ET: {'run' if run else 'skip'} ({why})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
