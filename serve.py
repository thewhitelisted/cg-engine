"""Live publisher for this server: computes the league page's odds around the clock and serves them as a JSON API.

Every TICK seconds it asks ESPN whether the league's live scores changed (If-None-Match: an unchanged answer costs
nothing) and reads the NBA scoreboard. When either changed it recomputes the odds; when what the page shows changed it
writes api/state.json into OUT and pushes a patch to every open page over api/events (api.py). Caddy serves OUT
(the page, web/app.js, the state) and proxies api/events to API_PORT; see server/install.sh.

    live refresh   new scores and game clocks on the model already built (about 0.2 s)
    full refresh   rosters, lineups, records, transactions and injury statuses, then the model rebuilt (about 3 s):
                   every FULL_EVERY, and within MIN_FULL_GAP of the live feed showing a lineup or a rostered
                   player's injury status that differs from ours

At every odds-history chart point it appends the exact model inputs and both lineup rules' odds (L4 and K1,
DECISIONS.md 2026-10-25) to data/replay/<date>.jsonl. Every LOGS_EVERY it commits the status, lineup and replay logs
and the odds history to this repo. On start it puts the current page code into OUT and into the GitHub Pages copy (a
commit only when the code changed; data is never committed there). Every CODE_EVERY it pulls main and restarts if code
changed, so pushes to main go live. When ticks keep failing for ALERT_AFTER it opens an issue in this repo, which
GitHub emails, and closes it once things recover.

    python serve.py                                         # under systemd: server/cg-live.service
    python serve.py --dry-run --once --out DIR              # one tick into DIR: no pushes, logs, state or issues
    python serve.py --dry-run --out DIR --serve-static      # local preview: http://127.0.0.1:8787/ serves DIR too
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests

import api
import inseason as I
import league_page as LP
import live as LV
import live_server as S
import loop
import model as M
import status_log
import sync

TICK = 10
FULL_EVERY, MIN_FULL_GAP = 300, 60
LOGS_EVERY, LOGS_RETRY = 3600, 300
CODE_EVERY = 120
ALERT_AFTER = {"tick": 300, "logs": 900, "site": 900}
OUT = Path("/srv/cg-live")
API_PORT = 8787
STATE = M.DATA / "serve_state.json"
HISTORY = M.DATA / "odds_history.json"         # the odds-history chart, committed with the logs
REPLAY_DIR = M.DATA / "replay"
LEGACY = ("data.json", "history.json", ".content_key")   # the server-rendered page's files, retired 2026-10-25
ISSUES_REPO = "thewhitelisted/cg-engine"
UV = Path.home() / ".local" / "bin" / "uv"
LIVE_VIEWS = ["mMatchupScore", "mScoreboard", "mStatus"]   # live totals, lineups, statuses: about 450 KB when changed
FULL_VIEWS = ["mTeam", "mRoster", "mMatchupScore", "mScoreboard", "mSettings", "mStatus"]   # publish_league.py's: about 2 MB


def log(msg):
    print(f"{datetime.now(LP.ET):%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


class Feed:
    """One ESPN league request, repeated with If-None-Match. get() returns (json, changed)."""

    def __init__(self, session, views):
        self.session, self.params = session, [("view", v) for v in views]
        self.etag = self.json = None

    def get(self):
        headers = {"If-None-Match": self.etag} if self.etag and self.json is not None else {}
        r = self.session.get(S.BASE, params=self.params, cookies=S.cookies(), headers=headers, timeout=30)
        if r.status_code == 304:
            return self.json, False
        if r.status_code == 401:
            raise RuntimeError("ESPN refused the login cookies (401). They may have expired: copy fresh ones.")
        r.raise_for_status()
        self.etag, self.json = r.headers.get("ETag"), r.json()
        return self.json, True


def lineups_live(j):
    """{team_id: sorted [(player_id, slot, injury status)]} from the live feed's rosters for today's scoring period."""
    cur, out = j["status"]["currentMatchupPeriod"], {}
    for m in j.get("schedule", []):
        if m.get("matchupPeriodId") != cur:
            continue
        for side in ("home", "away"):
            s = m.get(side) or {}
            entries = (s.get("rosterForCurrentScoringPeriod") or {}).get("entries")
            if entries is not None:
                out[s["teamId"]] = sorted((e["playerId"], e["lineupSlotId"],
                                           e["playerPoolEntry"]["player"].get("injuryStatus") or "") for e in entries)
    return out


def lineups_differ(live_lineups, lg):
    """True if any team's lineup in the live feed isn't the one in our league snapshot (an add, drop, slot move or
    injury status change)."""
    held = {t["id"]: sorted((e[0], e[2], e[3]) for e in t["roster"]) for t in lg["teams"]}
    return any(held.get(t) != lu for t, lu in live_lineups.items())


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def page_repo():
    return Path(json.loads((M.DATA / "publish.json").read_text())["repo_dir"])


class Alert:
    """One GitHub issue per failing thing (key), opened once it has failed for ALERT_AFTER[key] seconds and closed
    with a comment when it next succeeds. The open issue numbers live in the state file, so restarts don't repeat them."""

    def __init__(self, state, save, enabled=True):
        self.state, self.save, self.enabled = state, save, enabled
        self.since = {}

    def gh(self, *args):
        return subprocess.run(["gh", *args, "-R", ISSUES_REPO], capture_output=True, text=True, timeout=60)

    def update(self, key, ok, err=""):
        issues = self.state.setdefault("issues", {})
        if ok:
            self.since.pop(key, None)
            if issues.get(key) and self.enabled:
                self.gh("issue", "close", str(issues[key]), "-c", f"Recovered at {datetime.now(LP.ET):%a %b %-d, %-I:%M %p} ET.")
                log(f"alert closed: {key} (issue #{issues[key]})")
                issues[key] = None
                self.save()
            return
        first = self.since.setdefault(key, time.time())
        if issues.get(key) or not self.enabled or time.time() - first < ALERT_AFTER[key]:
            return
        body = (f"The live odds publisher on the server has been failing ({key}) since "
                f"{datetime.fromtimestamp(first, LP.ET):%a %b %-d, %-I:%M %p} ET.\n\nLatest error:\n\n```\n{err[-1500:]}\n```\n\n"
                f"Logs: `journalctl -u cg-live -n 100`. This issue closes itself when publishing recovers.")
        r = self.gh("issue", "create", "-t", f"Live odds publisher failing ({key}): {err.splitlines()[-1][:80] if err else '?'}",
                    "-b", body)
        if r.returncode == 0:
            issues[key] = r.stdout.strip().rsplit("/", 1)[-1]
            log(f"alert opened: {key} (issue #{issues[key]})")
            self.save()
        else:
            log(f"alert failed to open: {r.stderr.strip()[:200]}")


def git_in(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=cg-engine",
                           "-c", "user.email=cg-engine@users.noreply.github.com", *args],
                          capture_output=True, text=True, timeout=120)


def publish_site(repo, files, message):
    """Write files into the GitHub Pages copy, remove the retired ones, and push if anything changed; returns whether
    it pushed."""
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    gone = [n for n in LEGACY if (repo / n).exists()]
    if gone:
        git_in(repo, "rm", "-q", "--ignore-unmatch", *gone)
    git_in(repo, "add", *files)
    if git_in(repo, "diff", "--cached", "--quiet").returncode == 0:
        return False
    git_in(repo, "commit", "-q", "-m", message)
    if git_in(repo, "pull", "-q", "--rebase", "-X", "theirs").returncode != 0:      # conflicts: this copy wins
        git_in(repo, "rebase", "--abort")
        raise RuntimeError("page repo pull failed")
    p = git_in(repo, "push", "-q")
    if p.returncode != 0:
        raise RuntimeError(f"page push failed: {p.stderr.strip()[:300]}")
    return True


class Publisher:
    def __init__(self, out, dry_run=False, hub=None):
        self.out, self.dry = out, dry_run
        self.hub = hub or api.Hub()
        self.session = requests.Session()
        self.live_feed, self.full_feed = Feed(self.session, LIVE_VIEWS), Feed(self.session, FULL_VIEWS)
        self.lg = self.lv = self.tx = self.data = self.published = None
        self.full_at, self.rebuild, self.inputs = 0.0, True, None
        self.state = {} if dry_run else load_state()
        history = self.state.get("history")
        for path in (HISTORY, None):
            if history is None:
                try:
                    history = json.loads((path or page_repo() / "history.json").read_text())
                except Exception:
                    pass
        self.history = history or {}

    def install_site(self):
        """The page code into OUT (Caddy serves it): page, app and insights; the retired files go."""
        for name in LEGACY:
            (self.out / name).unlink(missing_ok=True)
        for name, text in LP.site_files().items():
            write_atomic(self.out / name, text)

    def save(self):
        if not self.dry:
            write_atomic(STATE, json.dumps({**self.state, "history": self.history}))

    def full_refresh(self, live):
        j, _ = self.full_feed.get()
        lg = sync.normalize(j)
        I._GAMELOGS = I._STATUSES = None              # their disk caches decide whether to refetch (30 / 10 min)
        try:
            self.tx = S.transactions(live["latest_scoring_period"], back=3)
        except Exception as e:
            log(f"transactions unavailable: {type(e).__name__}: {e}")
        if not self.dry:
            (M.DATA / "league_2027.json").write_text(json.dumps(lg))
            statuses = dict(I.all_statuses())
            if statuses:
                statuses.update({e[0]: e[3] for t in lg["teams"] for e in t["roster"]})
            status_log.record(statuses)
            status_log.record_lineups(lg, live["latest_scoring_period"])
            try:
                status_log.record_returns(S.injury_returns())     # DECISIONS.md E1
            except Exception as e:
                log(f"injury list unavailable: {type(e).__name__}: {e}")
        self.lg, self.full_at, self.rebuild = lg, time.time(), True

    def tick(self):
        """One round; returns what it did (for the log)."""
        j, _ = self.live_feed.get()
        live, _, _ = S.live_from(j)
        since_full = time.time() - self.full_at
        if self.lg is None or since_full >= FULL_EVERY or (since_full >= MIN_FULL_GAP and lineups_differ(lineups_live(j), self.lg)):
            self.full_refresh(live)
        try:
            sb = S.scoreboard(datetime.now(LP.ET))
            frac = {k: v[1] for k, v in sb.items()}
        except Exception:                             # no game clocks this round (as publish_league.py does)
            sb = frac = None
        started = LV.started_teams(sb)
        slots = {pid: sl for lu in lineups_live(j).values() for pid, sl, _ in lu}   # newest lineups (K1 locks)
        inputs = (self.full_at, json.dumps(live, sort_keys=True), json.dumps(sb, sort_keys=True, default=str),
                  json.dumps(sorted(slots.items())))
        if inputs == self.inputs:
            return None
        if self.rebuild or self.lv is None or self.lv.period != live["period"]:
            self.lv = LV.Live(live=live, today_frac=frac, lg=self.lg, started=started, slots=slots)
            self.rebuild, what = False, "full"
        else:
            self.lv.update(live, frac, started=started, slots=slots)
            what = "live"
        if not self.dry:
            (M.DATA / "live.json").write_text(json.dumps(live))
        trend = len((self.history.get("trend") or {}).get(str(live["period"]), []))
        data = LP.build(self.lv, self.history, sb=sb, tx=self.tx)
        if len((self.history.get("trend") or {}).get(str(live["period"]), [])) > trend:   # a new chart point
            self.archive(live, frac, started, slots)
        self.inputs = inputs
        state = LP.api_state(data)
        if self.published is not None and state["v"] == self.published["v"]:
            return None                                                  # nothing the page shows has changed
        write_atomic(self.out / "api" / "state.json", json.dumps(state, separators=(",", ":")))
        msg = self.hub.publish(state)
        self.data, self.published = data, state
        self.save()
        return (f"{what} refresh, published {state['v']} ({len(msg[1])} B {msg[0]} to {self.hub.count()} open): "
                + "; ".join(f"{g['home'][:12]} {g['p_home']:.0%}" for g in state["games"]))

    def archive(self, live, frac, started, slots):
        """Append this chart point's inputs and both lineup rules' odds (shadow: only one is published)."""
        odds = {}
        try:
            for arm, on in (("L4", False), ("K1", True)):
                self.lv.set_locks(on)
                odds[arm] = {f"{m['home']}-{m['away']}": [float(x) for x in (lambda r: [r[1], *r[0]])(
                    self.lv.matchup(m["home"], m["away"]))] for m in live["matchups"]}
        finally:
            self.lv.set_locks(LV.LINEUP_LOCKS)
        rosters = {t["id"]: [[e[0], e[2], e[3]] for e in t["roster"]] for t in self.lg["teams"]}
        rec = {"t": datetime.now(LP.ET).isoformat(timespec="seconds"), "published": "K1" if LV.LINEUP_LOCKS else "L4",
               "live": live, "frac": frac, "started": sorted(started) if started is not None else None,
               "slots": slots, "rosters": rosters, "odds": odds}            # odds: [P(home wins week), 9 categories]
        if not self.dry:
            REPLAY_DIR.mkdir(exist_ok=True)
            with (REPLAY_DIR / f"{datetime.now(LP.ET):%Y-%m-%d}.jsonl").open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def commit_logs(self):
        """Commit the status, lineup and replay logs and the odds history to this repo."""
        write_atomic(HISTORY, json.dumps(self.history))
        loop.save_status_log()
        self.state["logs_at"] = time.time()
        self.save()


def code_changed():
    """Pull main. True if anything besides the logs changed (the caller restarts to run it); reinstalls the
    requirements first if they changed."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=120)
    head = git("rev-parse", "HEAD").stdout.strip()
    r = git("pull", "-q", "--rebase", "--autostash")
    if r.returncode != 0:
        git("rebase", "--abort")
        log(f"code update skipped: {r.stderr.strip()[:200]}")
        return False
    changed = [f for f in git("diff", "--name-only", head, "HEAD").stdout.split()
               if not any(f == p or f.startswith(p + "/") for p in loop.LOGS)]
    if not changed:
        return False
    if "requirements.txt" in changed:
        subprocess.run([str(UV), "pip", "install", "-q", "--python", sys.executable, "-r", "requirements.txt"], check=True)
    log(f"new code ({', '.join(changed[:5])}{' ...' if len(changed) > 5 else ''}): restarting")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help=f"folder Caddy serves (default {OUT})")
    ap.add_argument("--dry-run", action="store_true", help="write OUT only: no pushes, logs, state, code updates or issues")
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    ap.add_argument("--api-port", type=int, default=API_PORT, help="port for api/events on 127.0.0.1 (0: no stream)")
    ap.add_argument("--serve-static", action="store_true", help="local preview: the API port also serves OUT")
    args = ap.parse_args()
    os.chdir(Path(__file__).parent)
    pub = Publisher(args.out, args.dry_run)
    alert = Alert(pub.state, pub.save, enabled=not args.dry_run)
    pub.install_site()
    if args.api_port:
        api.start(pub.hub, port=args.api_port, static=args.out if args.serve_static else None)
    if not args.dry_run:
        try:
            if publish_site(page_repo(), LP.site_files(), "Site: page code"):
                log("page code pushed to the GitHub copy")
            alert.update("site", True)
        except Exception as e:
            log(f"site publish failed: {type(e).__name__}: {e}")
            alert.update("site", False, traceback.format_exc())
    code_at = time.time()
    log(f"serving {args.out} every {TICK}s, events on :{args.api_port}{' (dry run)' if args.dry_run else ''}")
    while True:
        t0 = time.time()
        try:
            what = pub.tick()
            if what:
                log(what)
            alert.update("tick", True)
        except Exception as e:
            log(f"tick failed: {type(e).__name__}: {e}")
            alert.update("tick", False, traceback.format_exc())
        if not args.dry_run and time.time() - pub.state.get("logs_at", 0) >= LOGS_EVERY:
            try:
                pub.commit_logs()
                alert.update("logs", True)
            except Exception as e:
                log(f"log commit failed: {type(e).__name__}: {e}")
                pub.state["logs_at"] = time.time() - LOGS_EVERY + LOGS_RETRY
                alert.update("logs", False, traceback.format_exc())
        if not args.dry_run and time.time() - code_at >= CODE_EVERY:
            code_at = time.time()
            try:
                if code_changed():
                    os.execv(sys.executable, [sys.executable, *sys.argv])
            except Exception as e:
                log(f"code update failed: {type(e).__name__}: {e}")
        if args.once:
            break
        time.sleep(max(0.0, TICK - (time.time() - t0)))


if __name__ == "__main__":
    main()
