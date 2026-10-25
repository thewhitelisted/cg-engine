"""Refresh the public league page and push it to its GitHub Pages repo (cg-engine runs it every 10 min during games).

    python3 publish_league.py              # sync ESPN, rebuild the page, commit + push if the odds changed

Config: data/publish.json  {"repo_dir": "/path/to/local/clone/of/the/pages/repo"}
Needs data/espn_auth.json (ESPN cookies) and working git credentials for the push.
"""
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import inseason as I
import league_page as LP
import live as LV
import live_server as S
import model as M
import status_log
import sync

LOG = M.DATA / "publish.log"
INSIGHTS = Path(__file__).parent / "web" / "insights.html"     # static explainer page, linked from the footer


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with LOG.open("a") as f:
        f.write(line + "\n")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)


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
        tx = S.transactions(live["latest_scoring_period"], back=3)   # earlier moves are already counted in history.json
    except Exception as e:
        log(f"transactions unavailable: {type(e).__name__}: {e}")
        tx = None
    lv = LV.Live(live=live, today_frac=frac, lg=I.load_league(), started=LV.started_teams(sb))
    try:
        history = json.loads(open(f"{repo}/history.json").read())
    except Exception:
        history = {}
    data = LP.build(lv, history, sb=sb, tx=tx)
    key = LP.content_key(data)
    try:
        old = open(f"{repo}/.content_key").read()
    except Exception:
        old = ""
    insights = INSIGHTS.read_text() if INSIGHTS.exists() else None
    try:
        old_insights = open(f"{repo}/insights.html").read()
    except Exception:
        old_insights = None
    if key == old and insights == old_insights:
        log("page content unchanged; nothing to push")
        return
    if insights is not None:
        open(f"{repo}/insights.html", "w").write(insights)
    open(f"{repo}/index.html", "w").write(LP.render(data))
    open(f"{repo}/data.json", "w").write(json.dumps(LP.public_json(data)))
    open(f"{repo}/history.json", "w").write(json.dumps(history))
    open(f"{repo}/.content_key", "w").write(key)
    git(repo, "pull", "--rebase", "--quiet")
    git(repo, "add", "index.html", "data.json", "history.json", ".content_key", "insights.html")
    c = git(repo, "commit", "-m", f"Update odds {data['updated']}")
    p = git(repo, "push", "--quiet")
    if p.returncode != 0:
        log(f"push failed: {p.stderr.strip()[:300]}")
        sys.exit(1)
    log(f"published matchup {data['period']}: " + "; ".join(f"{g['home'][:12]} {g['p_home']:.0%}" for g in data["games"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
