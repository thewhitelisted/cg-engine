"""K1 lineup locks (DECISIONS.md, 2026-10-25): replay matchup 1's odds-history chart with L4 and with K1.

Inputs at each chart point t are rebuilt as registered: rosters and slots from lineup_log.csv, injury statuses from
status_log.csv, game logs of completed days, counting stats from the page commit published at t, FG/FT attempts from
exact start-of-day totals (game logs x ESPN's historical lineups) plus the elapsed share of games in progress, game
clocks and "started" from ESPN play-by-play, and on Oct 25 from 16:50 ET the cadence log's exact totals. The model's
clock is frozen at t's date. The same Live object gives both arms (set_locks), so they differ only in K1.

    python research_locks.py replay [PAGE_REPO]     # replay every chart point -> data/.cache_k1_replay.pkl
    python research_locks.py validity               # harness precondition: replayed vs published odds
    python research_locks.py compare                # L4 vs K1 against matchup 1's final results
    python research_locks.py bench                  # do ESPN's live totals leave out benched players? (cadence log)
"""
import contextlib
import csv
import html
import json
import pickle
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import inseason as I
import live as LV
import live_server as S
import model as M

ET = ZoneInfo("America/New_York")
PERIOD = 1
START = pd.Timestamp("2026-10-23 18:26", tz=ET)          # first lineup_log row: 22:25:09Z
CADENCE = Path.home() / "fantasy-local" / "espn_cadence" / "2026-10-25"
CADENCE_FROM = pd.Timestamp("2026-10-25 16:50", tz=ET)
ACTIVE = (0, 1, 2, 3, 4, 5, 6, 11)
CAT_COLS = {"FG%": "fg", "FT%": "ft", "3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov",
            "PTS": "pts"}
REPLAY = M.DATA / ".cache_k1_replay.pkl"
K1_LIVE = None                    # when the published odds switched from L4 to K1 (the server publisher's start)
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------- the chart and the published pages
def chart_points(page):
    tr = json.loads((Path(page) / "history.json").read_text())["trend"][str(PERIOD)]
    pts = [(pd.Timestamp(t, tz=ET), odds) for t, odds in tr]
    return [(t, o) for t, o in pts if t >= START]


def page_totals(page, commit, name_id):
    """{team id: {"tpm": .., "reb": .., ..., "fg": pct or None, "ft": pct or None}} from a published page's cards."""
    doc = git(page, "show", f"{commit}:index.html")
    homes = re.findall(r'class="team ha">([^<]*)<', doc)
    aways = re.findall(r'class="team hb">([^<]*)<', doc)
    tables = re.findall(r'<table class="cats">(.*?)</table>', doc, re.S)
    out = {}
    for h, a, tb in zip(homes, aways, tables):
        rows = re.findall(r"<tr>(.*?)</tr>", tb, re.S)
        cats = re.findall(r"<th>([^<]+)</th>", rows[0])
        for name, row in ((h, rows[1]), (a, rows[2])):
            vals = re.findall(r"<td[^>]*>([^<]*)</td>", row)[-len(cats):]
            tot = {}
            for c, v in zip(cats, vals):
                k = CAT_COLS[c]
                tot[k] = (None if v in ("—", "") else float(v)) if k in ("fg", "ft") else float(v)
            out[name_id[html.unescape(name)]] = tot
    return out


# ---------------------------------------------------------------- logs as of a moment
def last_rows(path, t_utc, key):
    cur = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if pd.Timestamp(row["time_utc"]) > t_utc:
                break
            cur[key(row)] = row
    return cur


def lineups_at(t_utc):
    rows = last_rows(M.DATA / "lineup_log.csv", t_utc, lambda r: (int(r["team_id"]), int(r["player_id"])))
    out = {}
    for (tid, pid), r in rows.items():
        if int(r["slot"]) >= 0:
            out.setdefault(tid, {})[pid] = int(r["slot"])
    return out


def statuses_at(t_utc):
    rows = last_rows(M.DATA / "status_log.csv", t_utc, lambda r: int(r["player_id"]))
    return {pid: ("" if r["status"] == "ACTIVE" else r["status"]) for pid, r in rows.items()}


# ---------------------------------------------------------------- ESPN: historical lineups, games, play-by-play
def espn_lineups(sp):
    j = requests.get(S.BASE, params=[("view", "mRoster"), ("scoringPeriodId", sp)], cookies=S.cookies(), timeout=60).json()
    return {t["id"]: {e["playerId"]: e["lineupSlotId"] for e in t["roster"]["entries"]} for t in j["teams"]}


def clock_s(txt):
    if ":" in txt:
        m, s = txt.split(":")
        return 60 * int(m) + float(s)
    return float(txt or 0)


def games_on(day):
    """[{teams, tip, status, plays: [(wallclock, period, clock seconds, type)]}] for one ET date."""
    ev = requests.get(f"{SB}/scoreboard", params={"dates": day.strftime("%Y%m%d")}, timeout=30).json()["events"]
    out = []
    for e in ev:
        s = requests.get(f"{SB}/summary", params={"event": e["id"]}, timeout=30).json()
        plays = [(pd.Timestamp(p["wallclock"]), p["period"]["number"], clock_s(p["clock"]["displayValue"]),
                  p.get("type", {}).get("text", "")) for p in s.get("plays", []) if p.get("wallclock")]
        out.append({"teams": [int(c["team"]["id"]) for c in e["competitions"][0]["competitors"]],
                    "tip": pd.Timestamp(e["date"]), "status": e["status"]["type"]["name"], "plays": sorted(plays)})
    return out


def clocks_at(games, t_utc):
    """(share of each team's game left, teams whose game has started) at t, from play-by-play."""
    frac, started = {}, set()
    for g in games:
        done = [p for p in g["plays"] if p[0] <= t_utc]
        if not done:
            left = 1.0
        elif done[-1][3] == "End Game" or (g["status"] == "STATUS_FINAL" and done[-1] is g["plays"][-1]):
            left = 0.0
        else:
            _, per, clk, _ = done[-1]
            left = max(0.0, ((4 - per) * 720 + clk) / 2880) if per <= 4 else 0.0
        on = g["status"] not in S.NEVER_STARTED and (g["tip"] <= t_utc or bool(done))
        for tm in g["teams"]:
            frac[tm] = left
            if on:
                started.add(tm)
    return frac, started


# ---------------------------------------------------------------- totals
STATS = ("pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fgm", "fga", "ftm", "fta")


def day_lines(g, sp):
    return {int(r.id): r for r in g[g.scoring_period == sp].itertuples(index=False)}


def start_of_day(g, hist, sps_before):
    """{team: {stat: total}} from completed days of the period: active players' lines (ESPN's locked lineups)."""
    out = {}
    for sp in sps_before:
        lines = day_lines(g, sp)
        for tid, lu in hist[sp].items():
            tot = out.setdefault(tid, dict.fromkeys(STATS, 0.0))
            for pid, slot in lu.items():
                if slot in ACTIVE and pid in lines:
                    for s in STATS:
                        tot[s] += float(getattr(lines[pid], s))
    return out


def cadence_totals(t_utc):
    path = CADENCE / "fantasy_changes.csv"
    if not path.exists():
        return None
    cur = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["t"]) > t_utc.timestamp():
                break
            if r["kind"] == "team":
                cur.setdefault(int(r["id"]), {})[r["stat"]] = float(r["value"])
    return cur


# ---------------------------------------------------------------- one chart point
@contextlib.contextmanager
def frozen(day, g, statuses):
    """The model as of `day`: its date, completed-day game logs and the statuses logged then."""
    class Day(date):
        @classmethod
        def today(cls):
            return day
    saved = (LV.date, I.gamelogs, I.all_statuses, I.participation, I._GAMELOGS, I._STATUSES)
    part = I.participation
    LV.date, I.gamelogs, I.all_statuses = Day, (lambda: g), (lambda: statuses)
    I.participation = lambda season, gg, today=None: part(season, gg, today=pd.Timestamp(day))
    I._GAMELOGS, I._STATUSES = g, statuses
    try:
        yield
    finally:
        LV.date, I.gamelogs, I.all_statuses, I.participation, I._GAMELOGS, I._STATUSES = saved


def replay(page):
    page = Path(page)
    base = json.loads(git(Path(__file__).parent, "show", "0536de4:data/league_2027.json"))   # Oct 21: matchup 1 open
    names = dict(pd.read_csv(M.DATA / "players_2027.csv", usecols=["id", "name"]).values)
    name_id = {t["name"]: t["id"] for t in json.loads((M.DATA / "league_2027.json").read_text())["teams"]}
    commits = {}
    for line in git(page, "log", "--format=%H %s").splitlines():
        h, subj = line.split(" ", 1)
        commits.setdefault(subj, h)
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    start, end = LV.period_dates(M.get_season(M.LIVE_SEASON, LV.LIVE_CONFIG["proj"]), PERIOD)
    sp_of = dict(sch.drop_duplicates("date")[["date", "scoring_period"]].values)
    period_sps = sorted({sp_of[d] for d in sp_of if start <= d <= end})
    g_all = I._fetch_all_gamelogs()
    hist = {sp: espn_lineups(sp) for sp in period_sps}
    games = {}
    points = chart_points(page)
    out = []
    for n, (t, published) in enumerate(points):
        day = t.date()
        d = pd.Timestamp(day)
        sp = sp_of[d]
        t_utc = (t + pd.Timedelta(seconds=30)).tz_convert("UTC")            # mid-minute of the build
        as_of = (t + pd.Timedelta(seconds=60)).tz_convert("UTC")             # logs written by that publish
        subj = f"Update odds {t:%a %b %-d, %-I:%M %p} ET"
        if day not in games:
            games[day] = games_on(day)
        frac, started = clocks_at(games[day], t_utc)
        if t >= CADENCE_FROM and cadence_totals(t_utc):
            tots, source = cadence_totals(t_utc), "cadence"
        elif subj in commits:
            pg = page_totals(page, commits[subj], name_id)
            sod = start_of_day(g_all, hist, [s for s in period_sps if s < sp])
            lines = day_lines(g_all, sp)
            tots = {}
            for tid, pt in pg.items():
                tot = {k: pt[k] for k in ("pts", "reb", "ast", "stl", "blk", "tpm", "tov")}
                for x, a, pct in (("fgm", "fga", "fg"), ("ftm", "fta", "ft")):
                    att = sod.get(tid, {}).get(a, 0.0)
                    for pid, slot in hist[sp].get(tid, {}).items():                # games in progress or over today
                        if slot in ACTIVE and pid in lines:
                            att += float(getattr(lines[pid], a)) * (1 - frac.get(team_of(pid), 1.0))
                    tot[a], tot[x] = att, (pt[pct] or 0.0) * att
                tots[tid] = tot
            source = "page"
        else:
            out.append({"t": t, "missing": True})
            continue
        lus = lineups_at(as_of)
        stats = statuses_at(as_of)
        lg = json.loads(json.dumps(base))
        for team in lg["teams"]:
            team["roster"] = [[pid, names.get(pid, str(pid)), slot, stats.get(pid, "")]
                              for pid, slot in sorted(lus.get(team["id"], {}).items())]
        live = {"period": PERIOD, "latest_scoring_period": sp,
                "matchups": [{"home": int(k.split("-")[0]), "away": int(k.split("-")[1]),
                              "home_tot": tots[int(k.split("-")[0])], "away_tot": tots[int(k.split("-")[1])]}
                             for k in published]}
        g_t = g_all[g_all.scoring_period < sp]
        with frozen(day, g_t, stats):
            lv = LV.Live(live=live, today_frac=frac, lg=lg, started=started, locks=False)
            arms = {}
            for arm, on in (("L4", False), ("K1", True)):
                lv.set_locks(on)
                arms[arm] = {f"{m['home']}-{m['away']}": lv.matchup(m["home"], m["away"])[:2] for m in live["matchups"]}
            team = lv.season.df["pro_team_id"].to_numpy()
            idx = {int(i): k for k, i in enumerate(lv.season.df["id"])}
        locked = {tid for tid, lu in lus.items()
                  if any(sl != I.IR_SLOT and pid in idx and team[idx[pid]] in started for pid, sl in lu.items())}
        out.append({"t": t, "published": published, "source": source, "locked_teams": locked, "started": started,
                    "L4": {k: (float(p), np.asarray(pc)) for k, (pc, p) in arms["L4"].items()},
                    "K1": {k: (float(p), np.asarray(pc)) for k, (pc, p) in arms["K1"].items()}})
        print(f"{n + 1}/{len(points)} {t:%a %H:%M} {source:7} " + " ".join(
            f"{k}:{published[k]:.3f}/{out[-1]['L4'][k][0]:.3f}/{out[-1]['K1'][k][0]:.3f}" for k in published), flush=True)
    REPLAY.write_bytes(pickle.dumps(out))
    return out


_TEAM = None


def team_of(pid):
    """A player's NBA team (projection pool), for matching him to today's game clocks."""
    global _TEAM
    if _TEAM is None:
        _TEAM = dict(pd.read_csv(M.DATA / "players_2027.csv", usecols=["id", "pro_team_id"]).values)
        lg = json.loads((M.DATA / "league_2027.json").read_text())
        _TEAM.update({int(k): v for k, v in lg.get("pro_team", {}).items() if v})
    return int(_TEAM.get(pid, 0))


# ---------------------------------------------------------------- reports
def published_arm(r):
    return "K1" if K1_LIVE is not None and r["t"] >= K1_LIVE else "L4"


def validity():
    """Replayed odds of the rule that was live at each point vs the published odds."""
    rows = [r for r in pickle.loads(REPLAY.read_bytes()) if not r.get("missing")]
    d = np.array([abs(r[published_arm(r)][k][0] - p) for r in rows for k, p in r["published"].items()]) * 100
    ok = d.mean() <= 1.0 and np.percentile(d, 95) <= 3.0
    print(f"chart points replayed: {len(rows)} ({sum(r['source'] == 'cadence' for r in rows)} from the cadence log); "
          f"pairs {len(d)}")
    print(f"|replayed (rule live then) - published|: mean {d.mean():.2f} pts, p95 {np.percentile(d, 95):.2f}, max {d.max():.2f} "
          f"-> {'PASS' if ok else 'FAIL'} (registered: mean <= 1.0, p95 <= 3.0)")
    worst = sorted(((abs(r[published_arm(r)][k][0] - p), r["t"], k, p, r[published_arm(r)][k][0], r["source"]) for r in rows
                    for k, p in r["published"].items()), reverse=True)[:8]
    for w in worst:
        print(f"  {w[1]:%a %H:%M} {w[2]:6} published {w[3]:.3f} replayed {w[4]:.3f} ({w[5]})")
    return ok


def outcomes():
    j = requests.get(S.BASE, params=[("view", "mMatchupScore"), ("view", "mScoreboard")], cookies=S.cookies(), timeout=60).json()
    res = {}
    for m in j["schedule"]:
        if m.get("matchupPeriodId") != PERIOD or not m.get("away"):
            continue
        key = f"{m['home']['teamId']}-{m['away']['teamId']}"
        win = {"HOME": 1.0, "AWAY": 0.0, "TIE": 0.5}.get(m.get("winner"))
        sbs = (m["home"].get("cumulativeScore") or {}).get("scoreByStat") or {}
        cats = {}
        for sid, c in (("19", "FG%"), ("20", "FT%"), ("17", "3PM"), ("6", "REB"), ("3", "AST"), ("2", "STL"),
                       ("1", "BLK"), ("11", "TO"), ("0", "PTS")):
            cats[c] = {"WIN": 1.0, "LOSS": 0.0, "TIE": 0.5}.get((sbs.get(sid) or {}).get("result"))
        res[key] = (win, cats)
    return res


def compare():
    rows = [r for r in pickle.loads(REPLAY.read_bytes()) if not r.get("missing")]
    res = outcomes()
    if any(v[0] is None for v in res.values()):
        sys.exit("matchup 1 isn't final on ESPN yet")
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
    ll = lambda p, y: -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))
    pairs = [(r, k) for r in rows for k in r["published"]]
    ident = [abs(r["K1"][k][0] - r["L4"][k][0]) < 1e-9 for r, k in pairs
             if not ({int(x) for x in k.split("-")} & r["locked_teams"])]
    print(f"pass 1, identical odds before any of a matchup's games start: {sum(ident)}/{len(ident)} "
          f"-> {'PASS' if all(ident) else 'FAIL'}")
    out = {}
    for arm in ("L4", "K1"):
        p = np.array([r[arm][k][0] for r, k in pairs])
        y = np.array([res[k][0] for r, k in pairs])
        pc = np.array([np.asarray(r[arm][k][1], float) for r, k in pairs])
        yc = np.array([[res[k][1][c] for c in M.CATS] for r, k in pairs], float)
        out[arm] = {"ll": ll(p, y).mean(), "brier": ((p - y) ** 2).mean(), "cat_ll": ll(pc, yc).mean(),
                    "by_cat": ll(pc, yc).mean(0), "per_matchup": {}}
        for k in res:
            sel = np.array([kk == k for _, kk in pairs])
            out[arm]["per_matchup"][k] = (ll(p[sel], y[sel]).sum(), ((p[sel] - y[sel]) ** 2).sum(), sel.sum())
    for arm in ("L4", "K1"):
        o = out[arm]
        print(f"{arm}: log loss {o['ll']:.4f}, Brier {o['brier']:.4f}, category log loss {o['cat_ll']:.4f}  by category "
              + " ".join(f"{c} {v:.3f}" for c, v in zip(M.CATS, o["by_cat"])))
    better = out["K1"]["ll"] <= out["L4"]["ll"] and out["K1"]["brier"] <= out["L4"]["brier"]
    print(f"pass 2, K1 no worse on log loss and Brier: {'PASS' if better else 'FAIL'}")
    rng = np.random.default_rng(7)
    keys = list(res)
    diffs = []
    for _ in range(2000):
        bk = rng.choice(keys, len(keys))
        num = sum(out["K1"]["per_matchup"][k][0] - out["L4"]["per_matchup"][k][0] for k in bk)
        diffs.append(num / sum(out["L4"]["per_matchup"][k][2] for k in bk))
    print(f"log loss difference K1 - L4 {out['K1']['ll'] - out['L4']['ll']:+.4f}, 95% bootstrap over the 6 matchups "
          f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}] (information only)")
    jumps = sorted(((abs(r["K1"][k][0] - r["L4"][k][0]), r["t"], k, r["L4"][k][0], r["K1"][k][0]) for r, k in pairs),
                   reverse=True)
    moved = np.array([j[0] for j in jumps if j[0] >= 1e-9]) * 100
    if len(moved):
        print(f"pairs where K1 differs: {len(moved)}/{len(pairs)}; |K1 - L4| median {np.median(moved):.1f} pts, "
              f"p90 {np.percentile(moved, 90):.1f}, max {moved.max():.1f}")
    for j in jumps[:10]:
        print(f"  {j[1]:%a %H:%M} {j[2]:6} L4 {j[3]:.3f} -> K1 {j[4]:.3f}")
    tip = [r for r in rows if r["t"] >= pd.Timestamp("2026-10-25 17:00", tz=ET)][:1]
    for r in tip:
        k = "5-13"
        print(f"Oct 25 first chart point after LAL @ UTAH's tip ({r['t']:%H:%M}): Chrissy {1 - r['L4'][k][0]:.1%} (L4) "
              f"-> {1 - r['K1'][k][0]:.1%} (K1); user's local estimate 41.1%")
    return out


def bench():
    """Nathan's team (5): change in ESPN's live totals since 16:50 vs his players' lines tonight, active only and all."""
    lu = lineups_at(pd.Timestamp.now(tz="UTC"))[5]
    rows = list(csv.DictReader(open(CADENCE / "fantasy_changes.csv")))
    base, last, lines = {}, {}, {}
    for r in rows:
        if r["kind"] == "team" and r["id"] == "5":
            base.setdefault(r["stat"], float(r["value"]))
            last[r["stat"]] = float(r["value"])
        elif r["kind"] == "player" and int(r["id"]) in lu:
            lines.setdefault(int(r["id"]), {})[r["stat"]] = float(r["value"])
    for s in ("pts", "reb", "ast", "fga"):
        act = sum(v.get(s, 0) for pid, v in lines.items() if lu[pid] in ACTIVE)
        allp = sum(v.get(s, 0) for v in lines.values())
        print(f"{s}: ESPN team total moved {last.get(s, 0) - base.get(s, 0):+.0f}; tonight's lines active-only {act:.0f}, "
              f"with bench {allp:.0f}")
    benched = [pid for pid, sl in lu.items() if sl not in ACTIVE and pid in lines]
    print("benched players with lines tonight:", benched)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validity"
    if cmd == "replay":
        replay(sys.argv[2] if len(sys.argv) > 2 else Path.home() / "cg-531d2447")
    elif cmd == "validity":
        validity()
    elif cmd == "compare":
        compare()
    elif cmd == "bench":
        bench()
