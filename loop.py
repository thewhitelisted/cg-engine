"""Publish every 10 minutes while NBA games are on, inside one GitHub Actions job.

GitHub's scheduler starts runs late or drops them, so one run that lands in a game window stays alive and
publishes on its own clock until the window closes or MAX_RUN is reached. GitHub stops jobs at 6 h, so at the
limit, if games are still on, it starts a fresh run of this workflow itself (workflow_dispatch, handoff=true)
instead of waiting for the unreliable schedule. Outside a game window (morning refresh, manual runs) it
publishes once.
Exits 1 if every publish failed, so a broken login shows up as a failed run.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import gate

INTERVAL = timedelta(minutes=10)
MAX_RUN = timedelta(hours=5, minutes=30)


def hand_off():
    """Start a fresh run of this workflow on main, flagged as a handoff so its gate doesn't wait for this one."""
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    wf = os.environ.get("GITHUB_WORKFLOW_REF", "").split("@")[0].rsplit("/", 1)[-1]
    if not (token and repo and wf):
        print("handoff skipped: not running in GitHub Actions")
        return
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches", method="POST",
                                 data=json.dumps({"ref": "main", "inputs": {"handoff": "true"}}).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"handed off to a fresh run (HTTP {r.status})")
    except Exception as e:
        print(f"handoff failed ({type(e).__name__}: {e}); the schedule will have to pick it up")


def main():
    start = datetime.now(gate.ET)
    ok = fails = 0
    while True:
        t0 = datetime.now(gate.ET)
        if subprocess.run([sys.executable, "publish_league.py"]).returncode == 0:
            ok += 1
        else:
            fails += 1
        live, why = gate.game_window(datetime.now(gate.ET))
        nxt = t0 + INTERVAL
        if not live or nxt + INTERVAL - start > MAX_RUN:
            print(f"stopping after {ok + fails} publishes ({fails} failed): {why if not live else 'time limit'}")
            if live:
                hand_off()
            break
        time.sleep(max(0.0, (nxt - datetime.now(gate.ET)).total_seconds()))
    sys.exit(1 if ok == 0 else 0)


if __name__ == "__main__":
    main()
