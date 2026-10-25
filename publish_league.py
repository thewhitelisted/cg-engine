"""Fallback publisher: writes a copy of the site (page, app, api/state.json) to the GitHub Pages repo and pushes it.

The server (serve.py) normally publishes; this runs from the GitHub Actions workflow (manual) while the server is down.
The page then finds the live API unreachable and reads this copy's api/state.json instead.

    python3 publish_league.py              # sync ESPN, compute the odds, commit + push if the page changed

Config: data/publish.json  {"repo_dir": "/path/to/local/clone/of/the/pages/repo"}
Needs data/espn_auth.json (ESPN cookies) and working git credentials for the push.
"""
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import inseason as I
import league_page as LP
import live as LV
import live_server as S
import model as M
import serve
import status_log
import sync

LOG = M.DATA / "publish.log"
HISTORY = serve.HISTORY


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with LOG.open("a") as f:
        f.write(line + "\n")


def main():
    cfg = json.loads((M.DATA / "publish.json").read_text())
    repo = cfg["repo_dir"]
    lg = sync.normalize(S.espn(["mTeam", "mRoster", "mMatchupScore", "mScoreboard", "mSettings", "mStatus"]))
    (M.DATA / "league_2027.json").write_text(json.dumps(lg))
    statuses = dict(I.all_statuses())                   # whole player pool, plus rostered players outside it
    if statuses:
        statuses.update({e[0]: e[3] for t in lg["teams"] for e in t["roster"]})
    n = status_log.record(statuses)
    if n:
        log(f"status log: {n} change{'s' if n != 1 else ''}")
    live, _, _ = S.live_from(S.espn(["mMatchupScore", "mScoreboard", "mStatus"]))
    (M.DATA / "live.json").write_text(json.dumps(live))
    n = status_log.record_lineups(lg, live["latest_scoring_period"])
    if n:
        log(f"lineup log: {n} change{'s' if n != 1 else ''}")
    try:
        sb = S.scoreboard(datetime.now())
        frac = {k: v[1] for k, v in sb.items()}
    except Exception:
        sb = frac = None
    try:
        tx = S.transactions(live["latest_scoring_period"], back=3)   # earlier moves are already counted in the history
    except Exception as e:
        log(f"transactions unavailable: {type(e).__name__}: {e}")
        tx = None
    lv = LV.Live(live=live, today_frac=frac, lg=I.load_league(), started=LV.started_teams(sb))
    history = {}
    for path in (HISTORY, Path(repo) / "history.json"):               # the server's odds history; the old page copy's
        try:
            history = json.loads(path.read_text())
            break
        except Exception:
            pass
    state = LP.api_state(LP.build(lv, history, sb=sb, tx=tx))
    try:
        old = json.loads((Path(repo) / "api" / "state.json").read_text()).get("v")
    except Exception:
        old = None
    HISTORY.write_text(json.dumps(history))                             # committed with the logs (loop.py)
    page = Path(repo) / "index.html"
    if state["v"] == old and page.exists() and page.read_text() == LP.shell():
        log("page content unchanged; nothing to push")
        return
    if serve.publish_site(Path(repo), LP.site_files(state), f"Update odds {state['updated']}"):
        log(f"published matchup {state['period']}: " + "; ".join(f"{g['home'][:12]} {g['p_home']:.0%}" for g in state["games"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
