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
    lg = sync.normalize(S.espn(["mTeam", "mRoster", "mMatchupScore", "mSettings", "mStatus"]))
    (M.DATA / "league_2027.json").write_text(json.dumps(lg))
    live, _, _ = S.live_from(S.espn(["mMatchupScore", "mScoreboard", "mStatus"]))
    (M.DATA / "live.json").write_text(json.dumps(live))
    try:
        frac = {k: v[1] for k, v in S.scoreboard(datetime.now()).items()}
    except Exception:
        frac = None
    lv = LV.Live(live=live, today_frac=frac, lg=I.load_league())
    data = LP.build(lv)
    key = json.dumps([[g["home"], g["away"], round(g["p_home"], 3), g["score"]] for g in data["games"]])
    old = ""
    try:
        old = json.dumps([[g["home"], g["away"], round(g["p_home"], 3), g["score"]]
                          for g in json.loads(open(f"{repo}/data.json").read())["games"]])
    except Exception:
        pass
    if key == old:
        log("odds unchanged; nothing to push")
        return
    open(f"{repo}/index.html", "w").write(LP.render(data))
    open(f"{repo}/data.json", "w").write(json.dumps({k: data[k] for k in ("period", "dates", "updated")} |
                                                    {"games": [{k: g[k] for k in ("home", "away", "p_home", "score")}
                                                               for g in data["games"]]}))
    git(repo, "pull", "--rebase", "--quiet")
    git(repo, "add", "index.html", "data.json")
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
