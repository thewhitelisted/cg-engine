"""Publish every 10 minutes while NBA games are on, inside one GitHub Actions job.

GitHub's scheduler starts runs late or drops them, so one run that lands in a game window stays alive and
publishes on its own clock until the window closes or MAX_RUN is reached (GitHub stops jobs at 6 h; a later
scheduled run then takes over). Outside a game window (morning refresh, manual runs) it publishes once.
Exits 1 if every publish failed, so a broken login shows up as a failed run.
"""
import subprocess
import sys
import time
from datetime import datetime, timedelta

import gate

INTERVAL = timedelta(minutes=10)
MAX_RUN = timedelta(hours=5, minutes=30)


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
            break
        time.sleep(max(0.0, (nxt - datetime.now(gate.ET)).total_seconds()))
    sys.exit(1 if ok == 0 else 0)


if __name__ == "__main__":
    main()
