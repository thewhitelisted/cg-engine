"""Live matchup page on your Mac: http://localhost:8765

    python3 live_server.py            # then open http://localhost:8765 (Ctrl+C to stop)

Every 45 seconds it pulls your league's live category totals from
ESPN (with your cookies in data/espn_auth.json), the NBA scoreboard (game clocks), and re-computes the chance
of winning each category and the week (live.py, validated L1b). In-progress games count only their unplayed
share of a game (share of the 48 minutes left), an approximation that was not part of the validation.
Mac notifications: when your chance to win the week moves by 8+ points, when a category flips to the other
side of 50%, and a summary once the night's games are final.
"""
import json
import subprocess
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import inseason as I
import live as LV
import model as M
import sync

ROOT = Path(__file__).parent
PORT = 8765
FAST, SLOW, ROSTER_EVERY = 45, 45, 900   # refresh every 45 s at all times; rosters every 15 min
STAT_IDS = {"0": "pts", "1": "blk", "2": "stl", "3": "ast", "6": "reb", "11": "tov", "13": "fgm", "14": "fga",
            "15": "ftm", "16": "fta", "17": "tpm", "40": "min"}
BASE = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{sync.SEASON}/segments/0/leagues/"
        f"{sync.LEAGUE}")
NOTIFY_WIN_MOVE, NOTIFY_CAT_MOVE = 0.08, 0.10

STATE = {"status": "starting", "updated": None, "history": [], "error": None}
LOCK = threading.Lock()


def cookies():
    f = M.DATA / "espn_auth.json"
    if not f.exists():
        raise RuntimeError("data/espn_auth.json not found. See the setup steps in the reply or at the top of sync.py.")
    a = json.loads(f.read_text())
    return {"espn_s2": a["espn_s2"], "SWID": a["SWID"]}


def espn(views):
    r = requests.get(BASE, params=[("view", v) for v in views], cookies=cookies(), timeout=30)
    if r.status_code == 401:
        raise RuntimeError("ESPN refused the login cookies (401). They may have expired: copy fresh ones.")
    r.raise_for_status()
    return r.json()


def scoreboard(day: datetime):
    """NBA games today: {pro_team_id: (state, share of game left, short detail, score line)}."""
    r = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                     params={"dates": day.strftime("%Y%m%d")}, timeout=20)
    r.raise_for_status()
    out = {}
    want = day.strftime("%Y-%m-%d")
    for e in r.json().get("events", []):
        # only games actually dated `day` (US Eastern): a stale or wrong reply must not zero out today's games
        try:
            ev_day = pd.Timestamp(e["date"]).tz_convert("US/Eastern").strftime("%Y-%m-%d")
        except Exception:
            continue
        if ev_day != want:
            continue
        c = e["competitions"][0]
        st = c["status"]
        state = st["type"]["state"]
        per, clock = int(st.get("period") or 0), float(st.get("clock") or 0)
        if state == "pre":
            left = 1.0
        elif state == "post":
            left = 0.0
        else:
            left = max(0.0, ((4 - per) * 720 + clock) / 2880) if per <= 4 else 0.0
        teams = [(int(t["team"]["id"]), t["team"]["abbreviation"], t.get("score")) for t in c["competitors"]]
        line = " · ".join(f"{a} {s}" for _, a, s in teams)
        for tid, _, _ in teams:
            out[tid] = (state, left, st["type"].get("shortDetail", ""), line)
    return out


def live_from(j):
    cur = j["status"]["currentMatchupPeriod"]

    def tot(s):
        sbs = ((s.get("cumulativeScoreLive") or s.get("cumulativeScore") or {}).get("scoreByStat") or {})
        return {STAT_IDS[k]: (v.get("score") or 0) for k, v in sbs.items() if k in STAT_IDS}

    def tonight(s):
        rows = []
        for e in ((s.get("rosterForCurrentScoringPeriod") or {}).get("entries") or []):
            pl = e["playerPoolEntry"]["player"]
            st = next((x["stats"] for x in pl.get("stats") or []
                       if x.get("scoringPeriodId") == j["status"]["latestScoringPeriod"] and x.get("statSplitTypeId") == 5
                       and x.get("statSourceId") == 0 and x.get("stats")), {})
            rows.append({"name": pl["fullName"], "slot": e["lineupSlotId"], "pro": pl.get("proTeamId"),
                         "inj": pl.get("injuryStatus") or "", "line": {STAT_IDS[k]: v for k, v in st.items() if k in STAT_IDS}})
        return rows

    ms = [m for m in j["schedule"] if m.get("matchupPeriodId") == cur and m.get("away")]
    live = {"period": cur, "latest_scoring_period": j["status"]["latestScoringPeriod"],
            "matchups": [{"home": m["home"]["teamId"], "away": m["away"]["teamId"],
                          "home_tot": tot(m["home"]), "away_tot": tot(m["away"])} for m in ms]}
    mine = next(m for m in ms if LV.MY_TEAM in (m["home"]["teamId"], m["away"]["teamId"]))
    home = mine["home"]["teamId"] == LV.MY_TEAM
    return live, tonight(mine["home"] if home else mine["away"]), tonight(mine["away"] if home else mine["home"])


def notify(title, text):
    try:
        subprocess.run(["osascript", "-e", f'display notification {json.dumps(text)} with title {json.dumps(title)}'],
                       timeout=10, check=False)
    except Exception:
        pass


def fmt(k, cat):
    if cat in ("FG%", "FT%"):
        x, a = ("fgx", "fga") if cat == "FG%" else ("ftx", "fta")
        lg = M.LG_FG if cat == "FG%" else M.LG_FT
        return round(float(k[M.SK[x]] / k[M.SK[a]] + lg), 3) if k[M.SK[a]] > 0 else None
    s = {"3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "PTS": "pts"}[cat]
    return round(float(k[M.SK[s]]), 1)


def compute(lv, live, me_rows, op_rows, games):
    mine = next(x for x in live["matchups"] if LV.MY_TEAM in (x["home"], x["away"]))
    home = mine["home"] == LV.MY_TEAM
    opp = mine["away"] if home else mine["home"]
    pc, pw, (mA, mB), m = lv.matchup(mine["home"], mine["away"])
    if not home:
        pc, pw, mA, mB = 1 - pc, 1 - pw, mB, mA
    kn_me = LV.known_totals(m["home_tot"] if home else m["away_tot"])
    kn_op = LV.known_totals(m["away_tot"] if home else m["home_tot"])
    names = {t["id"]: t["name"] for t in lv.lg["teams"]}
    left = lv.games_left

    def players(rows):
        out = []
        for r in rows:
            g = games.get(r["pro"])
            out.append({**r, "game": None if g is None else {"state": g[0], "detail": g[2], "score": g[3]}})
        return out

    league = []
    for x in live["matchups"]:
        _, p, _, _ = lv.matchup(x["home"], x["away"])
        league.append({"home": names[x["home"]], "away": names[x["away"]], "p_home": round(p, 3)})
    return {
        "period": live["period"], "opponent": names[opp], "me": names[LV.MY_TEAM], "p_win": round(pw, 4),
        "games_left": {"me": round(left(LV.MY_TEAM), 1), "opp": round(left(opp), 1)},
        "cats": [{"cat": c, "me_now": fmt(kn_me, c), "opp_now": fmt(kn_op, c), "me_final": fmt(mA, c),
                  "opp_final": fmt(mB, c), "p": round(float(pc[i]), 4)} for i, c in enumerate(M.CATS)],
        "tonight": {"me": players(me_rows), "opp": players(op_rows)},
        "games": sorted({v[3]: {"line": v[3], "state": v[0], "detail": v[2]} for v in games.values()}.values(),
                        key=lambda g: g["detail"]),
        "league": league,
    }


def loop():
    lv, last_roster, last_note, notified_final = None, 0.0, None, set()
    while True:
        wait = SLOW
        try:
            now = datetime.now()
            if lv is None or time.time() - last_roster > ROSTER_EVERY:
                lg = sync.normalize(espn(["mTeam", "mRoster", "mMatchupScore", "mSettings", "mStatus"]))
                (M.DATA / "league_2027.json").write_text(json.dumps(lg))
                last_roster = time.time()
                I._GAMELOGS = None                     # fresh season-to-date stats with each roster refresh
                lv = None
            j = espn(["mMatchupScore", "mScoreboard", "mStatus"])
            live, me_rows, op_rows = live_from(j)
            games = scoreboard(now)
            frac = {tid: v[1] for tid, v in games.items()}
            if lv is None or lv.period != live["period"]:
                lv = LV.Live(live=live, today_frac=frac, lg=I.load_league())
            else:
                lv.update(live, frac)
            st = compute(lv, live, me_rows, op_rows, games)
            any_live = any(v[0] == "in" for v in games.values())
            evening_games = any(v[0] == "pre" for v in games.values()) and now.hour >= 17
            wait = FAST if (any_live or evening_games) else SLOW
            with LOCK:
                hist = STATE["history"] + [[now.strftime("%H:%M"), st["p_win"]]]
                STATE.update(st, status="ok", updated=now.strftime("%Y-%m-%d %H:%M:%S"), error=None,
                             history=hist[-400:], refresh_s=wait,
                             in_game_note="In-progress games count only their unplayed share (an approximation)." if any_live else "")
            # notifications
            if last_note is None:
                last_note = {"p": st["p_win"], "cats": {c["cat"]: c["p"] for c in st["cats"]}}
            else:
                if abs(st["p_win"] - last_note["p"]) >= NOTIFY_WIN_MOVE:
                    notify("Matchup update", f"Chance to win the week: {last_note['p']:.0%} → {st['p_win']:.0%}")
                    last_note["p"] = st["p_win"]
                for c in st["cats"]:
                    old = last_note["cats"][c["cat"]]
                    if (old - 0.5) * (c["p"] - 0.5) < 0 and abs(c["p"] - old) >= NOTIFY_CAT_MOVE:
                        notify("Category flipped", f"{c['cat']}: {old:.0%} → {c['p']:.0%} for you")
                        last_note["cats"][c["cat"]] = c["p"]
            day = now.strftime("%Y-%m-%d")
            if games and all(v[0] == "post" for v in games.values()) and day not in notified_final:
                lead = sum(c["p"] > 0.5 for c in st["cats"])
                notify("Tonight's games are final", f"You're favoured in {lead} of 9 categories; {st['p_win']:.0%} to win the week")
                notified_final.add(day)
        except Exception as e:
            with LOCK:
                STATE.update(status="error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            wait = FAST
        time.sleep(wait)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body = (ROOT / "web" / "live_page.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    print(f"Live matchup page: http://localhost:{PORT}  (Ctrl+C to stop)")
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
