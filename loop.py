"""Publish every 2 minutes while NBA games are on, inside one GitHub Actions job. The site deploys through GitHub
Actions (no hourly build cap) and the page refreshes itself in place.

GitHub's scheduler starts runs late or drops them, so one run that lands in a game window stays alive and
publishes on its own clock until the window closes or MAX_RUN is reached. GitHub stops jobs at 6 h, so at the
limit, if games are still on, it starts a fresh run of this workflow itself (workflow_dispatch, handoff=true)
instead of waiting for the unreliable schedule. Outside a game window (morning refresh, manual runs) it
publishes once.
After each publish it commits data/status_log.csv (injury-status changes) and data/lineup_log.csv (fantasy lineup
changes) back to this repo if they grew.
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

INTERVAL = timedelta(minutes=2)
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


LOGS = ["data/status_log.csv", "data/lineup_log.csv", "data/replay",     # replay: serve.py's chart-point inputs
        "data/odds_history.json",                                         # the odds-history chart
        "data/return_log.csv"]                                            # ESPN's expected returns (DECISIONS.md E1)


def save_status_log():
    """Commit and push the status and lineup logs if publish_league.py appended to them (or an earlier push failed)."""
    git = ["git", "-c", "user.name=cg-engine", "-c", "user.email=cg-engine@users.noreply.github.com"]
    changed = [p for p in LOGS
               if subprocess.run(["git", "status", "--porcelain", "--", p], capture_output=True, text=True).stdout.strip()]
    if changed:
        subprocess.run(git + ["add", *changed])
        subprocess.run(git + ["commit", "-q", "-m", "Status and lineup logs"])
    ahead = subprocess.run(["git", "rev-list", "--count", "@{u}..HEAD"], capture_output=True, text=True).stdout.strip()
    if ahead in ("", "0"):
        return
    subprocess.run(git + ["pull", "-q", "--rebase", "--autostash"])
    if subprocess.run(["git", "push", "-q"]).returncode != 0:
        print("status log push failed; retrying after the next publish")


def update_code():
    """Pull the latest code before each publish so changes pushed mid-window go live within one interval
    (publish_league.py runs as a fresh process; this loop itself keeps the version it started with)."""
    r = subprocess.run(["git", "pull", "-q", "--rebase", "--autostash"], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], capture_output=True)
        print(f"code update skipped: {r.stderr.strip()[:200]}")


def main():
    start = datetime.now(gate.ET)
    ok = fails = 0
    while True:
        t0 = datetime.now(gate.ET)
        update_code()
        if subprocess.run([sys.executable, "publish_league.py"]).returncode == 0:
            ok += 1
        else:
            fails += 1
        save_status_log()
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
