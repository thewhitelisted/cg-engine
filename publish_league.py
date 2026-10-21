"""Refresh the public league page and push it to its GitHub Pages repo (run every 30 min by launchd).

    python3 publish_league.py              # sync ESPN, rebuild the page, commit + push if the odds changed

Config: data/publish.json  {"repo_dir": "/path/to/local/clone/of/the/pages/repo"}
Needs data/espn_auth.json (ESPN cookies) and working git credentials for the push.
"""
import json
import subprocess
import sys
import traceback
from datetime import datetime

import inseason as I
import league_page as LP
import live as LV
import live_server as S
import model as M
import sync

LOG = M.DATA / "publish.log"


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
    live, _, _ = S.live_from(S.espn(["mMatchupScore", "mScoreboard", "mStatus"]))
    (M.DATA / "live.json").write_text(json.dumps(live))
    try:
        frac = {k: v[1] for k, v in S.scoreboard(datetime.now()).items()}
    except Exception:
        frac = None
    lv = LV.Live(live=live, today_frac=frac, lg=I.load_league())
    try:
        history = json.loads(open(f"{repo}/history.json").read())
    except Exception:
        history = {}
    data = LP.build(lv, history)
    key = json.dumps({k: v for k, v in data.items() if k != "updated"}, sort_keys=True, default=str)
    try:
        old = open(f"{repo}/.content_key").read()
    except Exception:
        old = ""
    if key == old:
        log("page content unchanged; nothing to push")
        return
    open(f"{repo}/index.html", "w").write(LP.render(data))
    open(f"{repo}/data.json", "w").write(json.dumps(LP.public_json(data)))
    open(f"{repo}/history.json", "w").write(json.dumps(history))
    open(f"{repo}/.content_key", "w").write(key)
    git(repo, "pull", "--rebase", "--quiet")
    git(repo, "add", "index.html", "data.json", "history.json", ".content_key")
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
