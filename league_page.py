"""Public league page: this week's matchups (live score, category totals, games, win chances and their trend,
categories still in play), who plays today, standings, week-by-week results, head-to-head, all-play record and
luck, league records, top performers, last week's odds against results, the model's season scorecard, and recent
adds/drops. Nothing player-level from the model (no projections, values or recommendations) and no playoff or
title odds.

    python3 league_page.py OUT_DIR        # writes a copy of the site (page, app, state) to OUT_DIR

build() gathers the page's data; api_state() turns it into the public JSON API (api/state.json), which web/app.js renders
in the browser and keeps current from api/events (api.py). serve.py runs it on the server (LIVE_URL); publish_league.py
is the manual fallback that writes a copy of the site to the GitHub Pages repo.
"""
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import inseason as I
import live as LV
import model as M

CATS = M.CATS
STAT_OF = {"3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "PTS": "pts"}
ET = ZoneInfo("America/New_York")
BENCH_SLOT = 12
IN_PLAY = (0.3, 0.7)                 # a category is a toss-up while either side has at least a 30% chance
LIVE_URL = "https://live.christopherjlee.ca/"   # the API (serve.py + api.py, behind Caddy: server/install.sh)
VOLATILE = ("updated", "v", "ts")    # build stamps, left out of the content key


def cat_values(tot):
    """Per category: the comparable value (percentages as made/attempted; turnovers negated) or None."""
    out = {}
    for c in CATS:
        if c in ("FG%", "FT%"):
            m, a = ("fgm", "fga") if c == "FG%" else ("ftm", "fta")
            out[c] = tot.get(m, 0) / tot[a] if tot.get(a) else None
        else:
            v = tot.get(STAT_OF[c], 0)
            out[c] = -v if c == "TO" else v
    return out


def compare(ta, tb):
    """(wins, losses, ties) for team a against team b from category totals."""
    va, vb = cat_values(ta), cat_values(tb)
    w = l = t = 0
    for c in CATS:
        a, b = va[c], vb[c]
        if a is None and b is None or (a is not None and b is not None and abs(a - b) < 1e-12):
            t += 1
        elif b is None or (a is not None and a > b):
            w += 1
        else:
            l += 1
    return w, l, t


def fmt(tot, c):
    if c in ("FG%", "FT%"):
        m, a = ("fgm", "fga") if c == "FG%" else ("ftm", "fta")
        return f"{tot.get(m, 0) / tot[a]:.3f}".lstrip("0") if tot.get(a) else "—"
    return f"{tot.get(STAT_OF[c], 0):.0f}"


def standings(lg):
    rows = []
    for t in lg["teams"]:
        w, l, ti = t.get("record") or [0, 0, 0]
        n = w + l + ti
        rows.append({"team": t["name"], "w": w, "l": l, "t": ti, "pct": (w + 0.5 * ti) / n if n else 0.0})
    rows.sort(key=lambda r: (-r["pct"], -r["w"]))
    lead = rows[0]
    for r in rows:
        r["gb"] = ((lead["w"] - r["w"]) + (r["l"] - lead["l"])) / 2
    return rows


def all_play(lg):
    """Each team against every other team in every finished week: matchup and category records, and luck."""
    names = {str(t["id"]): t["name"] for t in lg["teams"]}
    weeks = [str(w) for w in lg.get("completed_weeks", []) if str(w) in lg.get("week_totals", {})]
    if not weeks:
        return []
    rec = {tid: {"mw": 0, "ml": 0, "mt": 0, "cw": 0, "cl": 0, "ct": 0} for tid in names}
    for wk in weeks:
        tots = lg["week_totals"][wk]
        for a in tots:
            for b in tots:
                if a == b:
                    continue
                w, l, t = compare(tots[a], tots[b])
                r = rec[a]
                r["cw"], r["cl"], r["ct"] = r["cw"] + w, r["cl"] + l, r["ct"] + t
                key = "mw" if w > l else "ml" if l > w else "mt"
                r[key] += 1
    actual = {str(t["id"]): t.get("record") or [0, 0, 0] for t in lg["teams"]}
    rows = []
    for tid, r in rec.items():
        n = r["cw"] + r["cl"] + r["ct"]
        ap = (r["cw"] + 0.5 * r["ct"]) / n if n else 0.0
        w, l, t = actual[tid]
        an = w + l + t
        act = (w + 0.5 * t) / an if an else 0.0
        rows.append({"team": names[tid], "mw": r["mw"], "ml": r["ml"], "mt": r["mt"], "allplay_pct": ap,
                     "actual_pct": act, "luck": act - ap})
    rows.sort(key=lambda r: -r["allplay_pct"])
    return rows


def week_games(lv, g, start, end):
    """Per fantasy team: player-games this week so far (box scores of active players) and scheduled games left."""
    df = lv.season.df
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    sp = set(sch[(sch.date >= start) & (sch.date <= end)].scoring_period)
    played = g[g.scoring_period.isin(sp)] if len(g) else g
    idx = {int(i): k for k, i in enumerate(df["id"])}
    out = {}
    for t in lv.lg["teams"]:
        active = [e[0] for e in t["roster"] if e[2] != I.IR_SLOT]
        n_played = int(played["id"].isin(active).sum()) if len(played) else 0
        n_left = float(sum(lv.n_rem[idx[p]] for p in active if p in idx))
        out[t["id"]] = (n_played, int(round(n_left)))
    return out


def stat_line(r):
    """One player's week as text: counting stats, shooting percentages, turnovers."""
    pct = lambda m, a: f"{m / a:.3f}".lstrip("0") if a else "–"
    return (f"{r.pts:.0f} pts · {r.reb:.0f} reb · {r.ast:.0f} ast · {r.stl:.0f} stl · {r.blk:.0f} blk · {r.tpm:.0f} 3pm · "
            f"{pct(r.fgm, r.fga)} FG · {pct(r.ftm, r.fta)} FT · {r.tov:.0f} TO")


def performers(lv, g, start, end, n=6):
    """Rostered players' 9-category production this week against what the model expected from the same games.

    Production is the page's value scale without the replacement-level baseline (sum over categories of the
    stat over its spread, turnovers negative), so it stays positive and a percentage gap is meaningful.
    Expected = the model's current per-game rates x games played. Returns (best, worst)."""
    df = lv.season.df
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    sp = set(sch[(sch.date >= start) & (sch.date <= end)].scoring_period)
    if not len(g):
        return [], []
    owner = {e[0]: t["name"] for t in lv.lg["teams"] for e in t["roster"]}
    idx = {int(i): k for k, i in enumerate(df["id"])}
    wk = g[g.scoring_period.isin(sp) & g["id"].isin(owner) & g["id"].isin(idx)]
    if not len(wk):
        return [], []
    tot = wk.groupby("id")[["pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fgm", "fga", "ftm", "fta"]].sum()
    tot["g"] = wk.groupby("id").size()
    _, scale = M.static_value(df)
    mu = np.zeros((len(tot), len(M.STATK)))
    for st in ("fga", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        mu[:, M.SK[st]] = tot[st].to_numpy()
    mu[:, M.SK["fgx"]] = tot["fgm"] - M.LG_FG * tot["fga"]
    mu[:, M.SK["ftx"]] = tot["ftm"] - M.LG_FT * tot["fta"]
    exp = df[M.STATK].to_numpy(float)[[idx[int(p)] for p in tot.index]] * tot["g"].to_numpy()[:, None]

    def production(m):
        return sum(sign * m[:, j] / sd for j, _, sd, sign in scale.values())
    tot["actual"], tot["expected"] = production(mu), production(exp)
    tot["gap"] = tot["actual"] - tot["expected"]
    names = g.drop_duplicates("id").set_index("id")["name"]

    def rows(sel):
        return [{"player": names.get(pid, str(pid)), "team": owner[pid], "g": int(r.g), "line": stat_line(r),
                 "pct": float(r.gap / r.expected) if r.expected > 0 else None} for pid, r in sel.iterrows()]
    return (rows(tot[tot.gap > 0].sort_values("gap", ascending=False).head(n)),
            rows(tot[tot.gap < 0].sort_values("gap").head(n)))


def abbrevs(lg):
    return {t["id"]: t.get("abbrev") or "".join(w[0] for w in t["name"].split()[:3]).upper() for t in lg["teams"]}


def in_play(ta, tb, pc, ha, aa):
    """Categories either side can still plausibly win this week, with the current margin."""
    out = []
    va, vb = cat_values(ta), cat_values(tb)
    for c, p in zip(CATS, pc):
        if not IN_PLAY[0] < p < IN_PLAY[1]:
            continue
        a, b = va[c], vb[c]
        if a is None or b is None or abs(a - b) < 1e-12:
            out.append([c, "tied"])
        elif c in ("FG%", "FT%"):
            out.append([c, f"{ha if a > b else aa} {fmt(ta, c) if a > b else fmt(tb, c)} vs {fmt(tb, c) if a > b else fmt(ta, c)}"])
        else:
            out.append([c, f"{ha if a > b else aa} by {abs(a - b):.0f}"])
    return out


def record_trend(history, period, games, now):
    """Append this run's odds to history["trend"][period] when any matchup has moved half a point or more."""
    tr = history.setdefault("trend", {}).setdefault(str(period), [])
    cur = {g["key"]: round(g["p_home"], 3) for g in games}
    if not tr or any(abs(p - tr[-1][1].get(k, -1)) >= 0.005 for k, p in cur.items()):
        tr.append([now.strftime("%Y-%m-%dT%H:%M"), cur])
    return tr


def attach_trend(games, tr, now):
    """Per matchup: the odds path this week and the change since the last update before today."""
    today = now.strftime("%Y-%m-%d")
    base = next((snap for ts, snap in reversed(tr) if ts[:10] < today), None)
    for g in games:
        g["trend"] = [[ts, snap[g["key"]]] for ts, snap in tr if g["key"] in snap]
        b = None if base is None else base.get(g["key"])
        g["swing"] = None if b is None else round(g["p_home"] - b, 3)


def playing_today(lg, sb, now):
    """Per fantasy team: rostered (non-IR) players whose NBA team plays today, with tip times or game state."""
    pro = {str(k): v for k, v in pd.read_csv(M.DATA / "players_2027.csv", usecols=["id", "pro_team_id"]).values}
    pro.update(lg.get("pro_team", {}))
    if sb is None:                                       # no scoreboard: today's schedule, no times
        sch = pd.read_csv(M.DATA / "schedule_2027.csv")
        sb = {int(t): ("pre", 1.0, "", "", False) for t in sch[sch.date == I.game_day(now).strftime("%Y-%m-%d")].pro_team_id}

    def when(state, detail):
        if state == "post":
            return "final", 9e9
        if state == "in":
            return "live", -1
        m = re.search(r"(\d+):(\d+) ([AP]M)", detail or "")
        if not m:
            return "", 5e9
        h, mi = int(m[1]) % 12 + (12 if m[3] == "PM" else 0), int(m[2])
        return f"{int(m[1])}:{m[2]}{m[3][0].lower()}", h * 60 + mi

    rows = []
    for t in lg["teams"]:
        ps = []
        for pid, name, slot, inj in t["roster"]:
            team = pro.get(str(pid))
            if slot == I.IR_SLOT or team is None or int(team) not in sb:
                continue
            state, _, detail = sb[int(team)][:3]
            label, order = when(state, detail)
            first, _, last = name.partition(" ")
            ps.append({"name": f"{first[0]}. {last}" if last else name, "when": label, "order": order,
                       "bench": slot == BENCH_SLOT, "out": inj in ("OUT", "SUSPENSION")})
        ps.sort(key=lambda x: (x["order"], x["name"]))
        rows.append({"team": t["name"], "n": len(ps), "lineup": sum(not p["bench"] and not p["out"] for p in ps),
                     "players": ps})
    rows.sort(key=lambda r: (-r["lineup"], -r["n"], r["team"]))
    return rows


def week_grid(lg, games_now, period):
    """Each team's category score in every matchup so far (finals from ESPN, the current week live)."""
    names = {t["id"]: t["name"] for t in lg["teams"]}
    cells = {tid: {} for tid in names}
    for r in lg.get("results", []):
        wk, h, a, winner, hw, hl, ht = r[:7]
        cells[h][wk] = [hw, hl, ht, "w" if winner == "HOME" else "l" if winner == "AWAY" else "t", True]
        cells[a][wk] = [hl, hw, ht, "w" if winner == "AWAY" else "l" if winner == "HOME" else "t", True]
    for g in games_now:
        h, a = (int(x) for x in g["key"].split("-"))
        w, l, t = g["score"]
        if period not in cells[h]:
            cells[h][period] = [w, l, t, "w" if w > l else "l" if l > w else "t", False]
            cells[a][period] = [l, w, t, "w" if l > w else "l" if w > l else "t", False]
    weeks = sorted({wk for c in cells.values() for wk in c})
    rows = [{"team": names[tid], "cells": [c.get(wk) for wk in weeks],
             "rec": [sum(1 for x in c.values() if x[4] and x[3] == k) for k in "wlt"]} for tid, c in cells.items()]
    rows.sort(key=lambda r: (-(r["rec"][0] + 0.5 * r["rec"][2]), r["team"]))
    return {"weeks": weeks, "rows": rows} if any(x[4] for c in cells.values() for x in c.values()) else None


def head_to_head(lg):
    """Matchup record of every team against every other team, from finished matchups."""
    teams = sorted(lg["teams"], key=lambda t: t["name"].lower())
    ab = abbrevs(lg)
    rec = {t["id"]: {} for t in teams}
    for r in lg.get("results", []):
        _, h, a, winner = r[:4]
        for x, y, won in ((h, a, "HOME"), (a, h, "AWAY")):
            w, l, t = rec[x].get(y, [0, 0, 0])
            rec[x][y] = [w + (winner == won), l + (winner not in (won, "UNDECIDED", "TIE")), t + (winner in ("UNDECIDED", "TIE"))]
    if not any(rec.values()):
        return None
    return {"cols": [ab[t["id"]] for t in teams],
            "rows": [{"team": t["name"], "abbr": ab[t["id"]], "cells": [None if u["id"] == t["id"] else rec[t["id"]].get(u["id"])
                                                                      for u in teams]} for t in teams]}


def league_records(lg):
    """Single-matchup bests from finished matchups (counting stats skip the two-week All-Star matchup)."""
    names = {str(t["id"]): t["name"] for t in lg["teams"]}
    weeks = [w for w in lg.get("completed_weeks", []) if str(w) in lg.get("week_totals", {})]
    if not weeks:
        return []
    tw = [(tid, w, tot) for w in weeks if len(I.PERIOD_WEEKS.get(w, [0])) == 1
          for tid, tot in lg["week_totals"][str(w)].items() if tid in names]
    out = []
    for label, stat in (("Most points", "pts"), ("Most rebounds", "reb"), ("Most assists", "ast"), ("Most steals", "stl"),
                        ("Most blocks", "blk"), ("Most threes", "tpm"), ("Most turnovers", "tov")):
        if tw:
            tid, w, tot = max(tw, key=lambda x: (x[2].get(stat, 0), -x[1]))
            out.append({"what": label, "team": names[tid], "value": f"{tot.get(stat, 0):.0f}", "week": w})
    for label, m, a in (("Best FG%", "fgm", "fga"), ("Best FT%", "ftm", "fta")):
        att = [x[2].get(a, 0) for x in tw]
        floor = 0.75 * float(np.mean(att)) if att else 0
        ok = [x for x in tw if x[2].get(a, 0) >= floor and x[2].get(a, 0) > 0]
        if ok:
            tid, w, tot = max(ok, key=lambda x: (x[2][m] / x[2][a], -x[1]))
            out.append({"what": label, "team": names[tid], "week": w,
                        "value": f"{tot[m] / tot[a]:.3f}".lstrip("0") + f" ({tot[m]:.0f}/{tot[a]:.0f})"})
    res = [r for r in lg.get("results", []) if r[0] in weeks]
    if res:
        def side(r):
            _, h, a, winner, hw, hl, ht = r[:7]
            win, lose = (h, a) if hw >= hl else (a, h)
            return names[str(win)], names[str(lose)], f"{max(hw, hl)}–{min(hw, hl)}" + (f"–{ht}" if ht else "")
        r = max(res, key=lambda r: (abs(r[4] - r[5]), -r[0]))
        w_, l_, sc = side(r)
        out.append({"what": "Biggest blowout", "team": f"{w_} over {l_}", "value": sc, "week": r[0]})
        r = min(res, key=lambda r: (abs(r[4] - r[5]), -r[6], -r[0]))
        w_, l_, sc = side(r)
        out.append({"what": "Closest matchup", "team": f"{w_} vs {l_}" if r[4] == r[5] else f"{w_} over {l_}",
                    "value": sc, "week": r[0]})
    return out


def scorecard(lg, history):
    """How the odds at each week's first update did against the results, over the season so far."""
    rows = []
    for r in lg.get("results", []):
        wk, h, a, winner = r[:4]
        p = (history.get(str(wk)) or {}).get(f"{h}-{a}")
        if p is None:
            continue
        y = 1.0 if winner == "HOME" else 0.0 if winner == "AWAY" else 0.5
        rows.append((p, y))
    if not rows:
        return None
    p, y = np.array(rows).T
    decided = (y != 0.5) & (p != 0.5)
    fav_won = ((p > 0.5) == (y == 1))[decided]
    big = decided & (np.maximum(p, 1 - p) >= 0.75)
    big_won = ((p > 0.5) == (y == 1))[big]
    return {"n": len(rows), "fav": [int(fav_won.sum()), int((~fav_won).sum())],
            "big": [int(big_won.sum()), int((~big_won).sum())], "brier": float(np.mean((p - y) ** 2)),
            "conf": float(np.mean(np.maximum(p, 1 - p)))}


def recent_moves(tx, lg, now, days=7):
    """Executed adds, drops and trades from the last week, newest first."""
    if not tx:
        return []
    pname = {str(k): v for k, v in pd.read_csv(M.DATA / "players_2027.csv", usecols=["id", "name"]).values}
    pname.update({str(e[0]): e[1] for t in lg["teams"] for e in t["roster"]})
    tname = {t["id"]: t["name"] for t in lg["teams"]}
    cutoff = (now - timedelta(days=days)).timestamp() * 1000
    rows, seen = [], set()
    for t in tx:
        ts = t.get("processDate") or t.get("proposedDate") or 0
        if t.get("status") != "EXECUTED" or ts < cutoff:
            continue
        items = [i for i in t.get("items") or [] if i.get("type") in ("ADD", "DROP", "TRADE")]
        if not items:
            continue
        when = datetime.fromtimestamp(ts / 1000, ET).strftime("%a %-I:%M%p").replace("AM", "a").replace("PM", "p")
        nm = lambda i: pname.get(str(i["playerId"]), f"player {i['playerId']}")
        if any(i["type"] == "TRADE" for i in items):
            key = tuple(sorted((i["playerId"], i.get("fromTeamId"), i.get("toTeamId")) for i in items))
            if key in seen:
                continue
            seen.add(key)
            got = {}
            for i in items:
                got.setdefault(i.get("toTeamId"), []).append(nm(i))
            rows.append({"ts": ts, "when": when, "kind": "trade", "team": " ↔ ".join(tname.get(k, "?") for k in got),
                         "add": "; ".join(f"{tname.get(k, '?')} get {', '.join(v)}" for k, v in got.items()), "drop": ""})
            continue
        team = tname.get(t.get("teamId"), "?")
        rows.append({"ts": ts, "when": when, "kind": "waivers" if t.get("type") == "WAIVER" else "free agent" if
                     any(i["type"] == "ADD" for i in items) else "drop", "team": team,
                     "add": ", ".join(nm(i) for i in items if i["type"] == "ADD"),
                     "drop": ", ".join(nm(i) for i in items if i["type"] == "DROP")})
    rows.sort(key=lambda r: -r["ts"])
    return rows[:25]


def week_allplay(tots):
    """{team: (matchup wins, losses, ties)} for one week, every team against every other team."""
    out = {}
    for a in tots:
        r = [0, 0, 0]
        for b in tots:
            if a != b:
                w, l, _ = compare(tots[a], tots[b])
                r[0 if w > l else 1 if l > w else 2] += 1
        out[a] = r
    return out


def awards(lg, history):
    """Highlights of the latest finished matchup: upset, comeback, best week, unluckiest loss, luckiest win."""
    done = sorted(lg.get("completed_weeks", []))
    if not done:
        return None
    wk = done[-1]
    names = {t["id"]: t["name"] for t in lg["teams"]}
    res = [r for r in lg.get("results", []) if r[0] == wk and r[3] in ("HOME", "AWAY")]
    odds = history.get(str(wk)) or {}
    trend = (history.get("trend") or {}).get(str(wk)) or []
    out = []

    def won(r):                                          # (winner, loser, winner's odds key sign, score text)
        _, h, a, winner, hw, hl, ht = r[:7]
        tie = f"–{ht}" if ht else ""
        return (h, a, 1, f"{hw}–{hl}{tie}") if winner == "HOME" else (a, h, -1, f"{hl}–{hw}{tie}")

    ups = []
    for r in res:
        p = odds.get(f"{r[1]}-{r[2]}")
        if p is not None:
            w, l, s, sc = won(r)
            pw = p if s == 1 else 1 - p
            if pw < 0.5:
                ups.append((pw, w, l, sc))
    if ups:
        pw, w, l, sc = min(ups)
        out.append({"title": "Upset of the week", "team": names[w], "text": f"won {sc} over {names[l]} as a {pw:.0%} underdog"})
    elif odds and res:
        out.append({"title": "Upset of the week", "team": "None", "text": "every favorite won"})
    comebacks = []
    for r in res:
        key = f"{r[1]}-{r[2]}"
        path = [(ts, snap[key]) for ts, snap in trend if key in snap]
        if path:
            w, l, s, sc = won(r)
            ts, low = min(((ts, p if s == 1 else 1 - p) for ts, p in path), key=lambda x: x[1])
            if low < 0.5:
                comebacks.append((low, ts, w, l, sc))
    if comebacks:
        low, ts, w, l, sc = min(comebacks)
        day = datetime.fromisoformat(ts).strftime("%a")
        out.append({"title": "Comeback of the week", "team": names[w], "text": f"down to {low:.0%} on {day}, beat {names[l]} {sc}"})
    tots = (lg.get("week_totals") or {}).get(str(wk))
    if tots:
        ap = {int(k): v for k, v in week_allplay(tots).items()}
        n = len(ap) - 1
        best = max(ap, key=lambda k: (ap[k][0] + 0.5 * ap[k][2], -k))
        out.append({"title": "Best week", "team": names[best], "text": f"would have beaten {ap[best][0]} of {n} teams"})
        losers = [won(r)[1] for r in res]
        winners = [won(r)[0] for r in res]
        if losers:
            u = max(losers, key=lambda k: (ap[k][0] + 0.5 * ap[k][2], -k))
            out.append({"title": "Unluckiest loss", "team": names[u], "text": f"lost, but would have beaten {ap[u][0]} of {n} teams"})
        if winners:
            lw = min(winners, key=lambda k: (ap[k][0] + 0.5 * ap[k][2], k))
            out.append({"title": "Luckiest win", "team": names[lw], "text": f"won, but would have beaten only {ap[lw][0]} of {n} teams"})
    return {"period": wk, "items": out} if out else None


def category_ranks(lg):
    """Each team's season rank (1 = best) in every category over finished matchups, with the season value."""
    weeks = [str(w) for w in lg.get("completed_weeks", []) if str(w) in lg.get("week_totals", {})]
    if not weeks:
        return None
    names = {str(t["id"]): t["name"] for t in lg["teams"]}
    season = {tid: {} for tid in names}
    for w in weeks:
        for tid, tot in lg["week_totals"][w].items():
            if tid in season:
                for k, v in tot.items():
                    season[tid][k] = season[tid].get(k, 0) + v
    vals = {tid: cat_values(t) for tid, t in season.items()}
    rows = {tid: {"team": names[tid], "ranks": [], "shown": []} for tid in names}
    for c in CATS:
        key = {tid: -1e9 if vals[tid][c] is None else vals[tid][c] for tid in names}
        for tid in names:
            rows[tid]["ranks"].append(1 + sum(1 for u in names if key[u] > key[tid] + 1e-12))
            rows[tid]["shown"].append(fmt(season[tid], c))
    out = sorted(rows.values(), key=lambda r: (np.mean(r["ranks"]), r["team"]))
    for r in out:
        r["avg"] = float(np.mean(r["ranks"]))
    return {"n": len(names), "rows": out}


def activity(lg, tx, history):
    """Season adds, drops and trades per team, counted from ESPN's transaction log. Each run sees only the last
    week or so of transactions, so counted ones are kept in history["moves"] ({transaction key: [team, adds,
    drops]} and, for trades, {key: [teams]})."""
    seen = history.setdefault("moves", {})
    for t in tx or []:
        items = [i for i in t.get("items") or [] if i.get("type") in ("ADD", "DROP", "TRADE")]
        if t.get("status") != "EXECUTED" or not items:
            continue
        if any(i["type"] == "TRADE" for i in items):
            key = "trade:" + ",".join(sorted(f'{i["playerId"]}>{i.get("toTeamId")}' for i in items))
            seen.setdefault(key, sorted({i.get("toTeamId") for i in items} | {i.get("fromTeamId") for i in items}))
        elif t.get("id"):
            seen.setdefault(str(t["id"]), [t.get("teamId"), sum(i["type"] == "ADD" for i in items),
                                           sum(i["type"] == "DROP" for i in items)])
    rec = {t["id"]: [0, 0, 0] for t in lg["teams"]}
    for key, v in seen.items():
        if key.startswith("trade:"):
            for tid in v:
                if tid in rec:
                    rec[tid][2] += 1
        elif v[0] in rec:
            rec[v[0]][0] += v[1]
            rec[v[0]][1] += v[2]
    if not any(any(r) for r in rec.values()):
        return []
    names = {t["id"]: t["name"] for t in lg["teams"]}
    rows = [{"team": names[tid], "adds": a, "drops": d, "trades": tr} for tid, (a, d, tr) in rec.items()]
    rows.sort(key=lambda r: (-(r["adds"] + r["drops"] + 2 * r["trades"]), r["team"]))
    return rows


def build(lv, history=None, sb=None, tx=None):
    """Page data. `history` ({period: {matchup: odds at the first update}, "trend": {period: [[time, {matchup:
    odds}], ...]}}) is updated in place; `sb` is today's NBA scoreboard (live_server.scoreboard) and `tx` ESPN's
    recent transactions (live_server.transactions)."""
    now = datetime.now(ET)
    names = {t["id"]: t["name"] for t in lv.lg["teams"]}
    abbr = abbrevs(lv.lg)
    start, end = LV.period_dates(lv.season, lv.period)
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    in_period = sch[(sch.date >= start) & (sch.date <= end)].date
    if len(in_period):                                  # ESPN's matchup dates: first to last game day
        start, end = in_period.min(), in_period.max()
    g = I.gamelogs()
    gp = week_games(lv, g, start, end)
    games = []
    for m in lv.live["matchups"]:
        pc, pw, _, _ = lv.matchup(m["home"], m["away"])
        games.append({"key": f"{m['home']}-{m['away']}", "home": names[m["home"]], "away": names[m["away"]],
                      "home_abbr": abbr[m["home"]], "away_abbr": abbr[m["away"]],
                      "p_home": float(pw), "score": list(compare(m["home_tot"], m["away_tot"])),
                      "lean": [(c, float(p)) for c, p in zip(CATS, pc)],
                      "totals": [[c, fmt(m["home_tot"], c), fmt(m["away_tot"], c)] for c in CATS],
                      "lead": [compare_one(m["home_tot"], m["away_tot"], c) for c in CATS],
                      "in_play": in_play(m["home_tot"], m["away_tot"], pc, abbr[m["home"]], abbr[m["away"]]),
                      "games": [gp[m["home"]], gp[m["away"]]]})
    games.sort(key=lambda x: abs(x["p_home"] - 0.5))
    history = {} if history is None else history
    history.setdefault(str(lv.period), {x["key"]: round(x["p_home"], 3) for x in games})
    attach_trend(games, record_trend(history, lv.period, games, now), now)
    data = {"period": lv.period, "dates": f"{start:%b %-d} – {end:%b %-d}", "start": f"{start:%Y-%m-%d}",
            "end": f"{end:%Y-%m-%d}", "games": games, "today": playing_today(lv.lg, sb, now),
            "standings": standings(lv.lg), "grid": week_grid(lv.lg, games, lv.period), "h2h": head_to_head(lv.lg),
            "allplay": all_play(lv.lg), "records": league_records(lv.lg), **dict(zip(("top", "worst"), performers(lv, g, start, end))),
            "last": last_week(lv.lg, history), "scorecard": scorecard(lv.lg, history),
            "moves": recent_moves(tx, lv.lg, now), "awards": awards(lv.lg, history),
            "ranks": category_ranks(lv.lg), "activity": activity(lv.lg, tx, history), "updated": now.strftime("%a %b %-d, %-I:%M %p ET")}
    # v identifies the content (the page refreshes when it changes, even twice in a minute); ts orders builds, so a
    # stale copy (the GitHub fallback) never replaces newer numbers
    data["v"] = hashlib.sha1(content_key(data).encode()).hexdigest()[:12]
    data["ts"] = int(now.timestamp())
    return data


def content_key(data):
    """Everything on the page except the build stamps: equal keys mean nothing to republish."""
    return json.dumps({k: v for k, v in data.items() if k not in VOLATILE}, sort_keys=True, default=str)


def compare_one(ta, tb, c):
    va, vb = cat_values(ta)[c], cat_values(tb)[c]
    if va is None or vb is None or abs(va - vb) < 1e-12:
        return 0
    return 1 if va > vb else -1


def last_week(lg, history):
    done = sorted(lg.get("completed_weeks", []))
    if not done:
        return None
    wk = done[-1]
    names = {t["id"]: t["name"] for t in lg["teams"]}
    odds = history.get(str(wk), {})
    rows = []
    for r in lg.get("results", []):
        if r[0] != wk:
            continue
        _, h, a, winner, hw, hl, ht = r[:7]
        rows.append({"home": names[h], "away": names[a], "score": [hw, hl, ht],
                     "p_home": odds.get(f"{h}-{a}"), "home_won": winner == "HOME", "tie": winner == "TIE"})
    return {"period": wk, "rows": rows}


SHELL_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>The Cool Guys · League page</title>
<style>
:root { --bg: #f6f6f2; --card: #fcfcfb; --ink: #15171a; --ink2: #4d5056; --muted: #84878c; --rule: #e1e0d9;
  --a: #2a78d6; --b: #c8501e; --mid: #ece9e2; --good: #2e7a57; --bad: #b2402e; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg: #0f1011; --card: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
  --muted: #8e8d87; --rule: #2c2c2a; --a: #5b9ce8; --b: #ec7a45; --mid: #2c2c2a; --good: #63c497; --bad: #e8796a; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 820px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 14px; grid-template-columns: minmax(0, 1fr); }
h1 { font-size: 26px; margin: 0; letter-spacing: -.01em; }
h2 { font-size: 18px; margin: 18px 0 0; }
.sub { color: var(--muted); font-size: 13.5px; margin: 2px 0 6px; }
.headline { font-size: 14px; color: var(--ink2); }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; min-width: 0; }
.matchup { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
.side { display: grid; gap: 3px; min-width: 0; }
.side.r { text-align: right; }
.meta { color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.caption { color: var(--muted); font-size: 12px; text-align: center; margin-top: -4px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
.da { background: var(--a); } .db { background: var(--b); }
table.cats td:first-child { font-weight: 600; }
table.cats th:not(:first-child), table.cats td:not(:first-child) { min-width: 44px; }
.team { font-weight: 600; overflow-wrap: anywhere; }
.r { text-align: right; }
.pct { font-size: 30px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.05; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--mid); }
.bar i:first-child { background: var(--a); } .bar i:last-child { background: var(--b); }
.trend summary { cursor: pointer; list-style: none; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center;
  gap: 10px; padding: 6px 8px; border: 1px solid var(--rule); border-radius: 8px; background: var(--bg); user-select: none; }
.trend summary::-webkit-details-marker { display: none; }
.trend summary:hover { border-color: var(--muted); }
.trend summary:hover .pill { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.trend summary:focus-visible { outline: 2px solid var(--a); outline-offset: 2px; }
.trend .mini { height: 44px; min-width: 0; }
.trend .mini svg { display: block; width: 100%; height: 100%; }
.trend[open] .mini { display: none; }
.trend[open] summary { display: flex; justify-content: flex-end; border-color: transparent; background: none; padding: 0; }
.trend .pill { font-size: 12.5px; font-weight: 600; color: var(--ink2); border: 1px solid var(--rule); background: var(--card);
  border-radius: 999px; padding: 5px 11px; white-space: nowrap; transition: background .12s, color .12s; }
.trend .pill .chev { display: inline-block; margin-left: 4px; transition: transform .15s; }
.trend[open] .pill .chev { transform: rotate(180deg); }
.trend .pill .open { display: none; }
.trend[open] .pill .open { display: inline; }
.trend[open] .pill .closed { display: none; }
.tplot { height: 240px; margin-top: 6px; position: relative; touch-action: pan-y; }
.tplot svg { display: block; width: 100%; height: 100%; overflow: visible; }
.tplot text { font: 11.5px system-ui, sans-serif; fill: var(--muted); }
.tplot .end { font-weight: 700; font-size: 12.5px; }
.ttip { position: fixed; z-index: 20; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 12.5px;
  line-height: 1.4; padding: 6px 9px; border-radius: 6px; white-space: nowrap; }
.inplay { font-size: 12.5px; color: var(--ink2); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.lbl { color: var(--muted); }
.chip { border: 1px solid var(--rule); border-radius: 999px; padding: 1px 8px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.chip b { font-weight: 600; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13.5px; }
th, td { padding: 5px 7px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
th:first-child, td:first-child { text-align: left; }
th.l, td.l { text-align: left; }
tr:last-child td { border-bottom: 0; }
td.lead { font-weight: 700; }
td.who { white-space: normal; text-align: left; min-width: 250px; font-size: 12.5px; color: var(--ink2); line-height: 1.7; }
.who span { white-space: nowrap; margin-right: 10px; }
.who .bn { color: var(--muted); } .who .out { color: var(--muted); text-decoration: line-through; }
table.grid td:not(:first-child), table.grid th:not(:first-child) { text-align: center; min-width: 46px; }
td.gw { background: color-mix(in srgb, var(--good) 20%, transparent); }
td.gl { background: color-mix(in srgb, var(--bad) 18%, transparent); }
td.gt { background: var(--mid); }
td.live { background: none; color: var(--muted); font-style: italic; }
td.self { color: var(--muted); }
.ha { color: var(--a); } .hb { color: var(--b); }
.pos { color: var(--good); } .neg { color: var(--bad); }
.note { color: var(--muted); font-size: 12.5px; margin: 0; }
.card.tbl { padding: 6px 10px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile { gap: 2px; padding: 12px 14px; }
.tile .big { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.award .big { font-size: 17px; overflow-wrap: anywhere; }
nav.jump { position: sticky; top: 0; z-index: 5; margin: 0 -16px; padding: 8px 16px; display: flex; gap: 6px;
  overflow-x: auto; scrollbar-width: none; background: color-mix(in srgb, var(--bg) 90%, transparent);
  -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule); }
nav.jump::-webkit-scrollbar { display: none; }
nav.jump a { flex: none; padding: 6px 12px; border-radius: 999px; border: 1px solid var(--rule); background: var(--card);
  color: var(--ink2); font-size: 13px; font-weight: 500; text-decoration: none; white-space: nowrap; }
nav.jump a:hover { border-color: var(--muted); color: var(--ink); }
nav.jump a.on { background: var(--ink); border-color: var(--ink); color: var(--bg); }
nav.jump a:focus-visible { outline: 2px solid var(--a); outline-offset: 2px; }
.foot { display: flex; justify-content: center; padding-top: 18px; border-top: 1px solid var(--rule); margin-top: 8px; }
.btn { display: inline-block; padding: 9px 16px; border-radius: 999px; border: 1px solid var(--rule); background: var(--card);
  color: var(--ink); font-size: 14px; font-weight: 600; text-decoration: none; }
.btn:hover { border-color: var(--muted); }
.btn:focus-visible { outline: 2px solid var(--a); outline-offset: 2px; }
[id] { scroll-margin-top: 60px; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
td.rk { text-align: center; min-width: 40px; }
.card.tbl:has(> table.grid) { padding-left: 0; }
table.grid tr > :first-child { position: sticky; left: 0; z-index: 1; background: var(--card); padding-left: 10px; }
@media (max-width: 560px) { td:first-child { white-space: normal; min-width: 116px; } }
</style>
</head>
<body>
<div class="wrap"><p class="note">Loading…</p></div>
<script>
const CG = {};
function initNav() {
  if (CG.io) CG.io.disconnect();
  const nav = document.querySelector("nav.jump");
  if (!nav || !("IntersectionObserver" in window)) return;
  const links = [...nav.querySelectorAll("a")];
  const byId = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
  const seen = new Map();
  let hold = 0;
  const mark = id => {
    links.forEach(a => a.classList.toggle("on", a === byId.get(id)));
    const a = byId.get(id);
    if (a) nav.scrollTo({ left: a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2, behavior: "smooth" });
  };
  const update = () => {
    if (Date.now() < hold) return;
    const ids = [...byId.keys()];
    const atEnd = innerHeight + scrollY >= document.documentElement.scrollHeight - 4;
    const first = atEnd ? ids[ids.length - 1] : ids.find(id => seen.get(id));
    if (first) mark(first);
  };
  const io = CG.io = new IntersectionObserver(es => {
    es.forEach(e => seen.set(e.target.id, e.isIntersecting));
    update();
  }, { rootMargin: "-64px 0px -55% 0px" });
  byId.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  CG.onScroll && removeEventListener("scroll", CG.onScroll);
  CG.onScroll = () => { if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) update(); };
  addEventListener("scroll", CG.onScroll, { passive: true });
  links.forEach(a => a.addEventListener("click", () => { mark(a.getAttribute("href").slice(1)); hold = Date.now() + 900; }));
}
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const tip = Object.assign(document.createElement("div"), { className: "ttip", hidden: true });
  document.body.appendChild(tip);
  CG.hideTip = () => { tip.hidden = true; };
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const day = d => new Date(d + "T00:00:00");
  const fmt = t => t.toLocaleString("en-US", { weekday: "short", hour: "numeric", minute: "2-digit" });
  function draw(box) {
    const pts = JSON.parse(box.dataset.trend).map(([ts, p]) => [new Date(ts), p]);
    const t0 = day(box.dataset.start), t1 = new Date(day(box.dataset.end).getTime() + 86400000);
    const ha = box.dataset.ha, aa = box.dataset.aa;
    box.innerHTML = "";
    const W = box.clientWidth, H = box.clientHeight, L = 34, R = 64, T = 10, B = 24;
    const x = t => L + (t - t0) / (t1 - t0) * (W - L - R), y = p => T + (1 - p) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    for (const g of [0, .25, .5, .75, 1]) {
      el("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), stroke: css("--rule"), "stroke-dasharray": g === .5 ? "4 3" : "" }, svg);
      el("text", { x: L - 6, y: y(g) + 4, "text-anchor": "end" }, svg).textContent = Math.round(g * 100) + "%";
    }
    for (let d = new Date(t0); d < t1; d = new Date(d.getTime() + 86400000)) {
      el("line", { x1: x(d), x2: x(d), y1: T, y2: H - B, stroke: css("--rule"), "stroke-opacity": .6 }, svg);
      el("text", { x: x(new Date(d.getTime() + 43200000)), y: H - 6, "text-anchor": "middle" }, svg).textContent =
        d.toLocaleDateString("en-US", { weekday: "short" });
    }
    const path = f => pts.map(([t, p], i) => (i ? `H${x(t)}V${y(f(p))}` : `M${x(t)},${y(f(p))}`)).join("");
    const lines = [[p => p, "--a", ha], [p => 1 - p, "--b", aa]];
    const last = pts[pts.length - 1];
    const endY = lines.map(([f]) => y(f(last[1])));
    if (Math.abs(endY[0] - endY[1]) < 15) { const m = (endY[0] + endY[1]) / 2, s = endY[0] <= endY[1] ? -1 : 1;
      endY[0] = m + s * 8; endY[1] = m - s * 8; }
    lines.forEach(([f, c, name], k) => {
      if (pts.length > 1) el("path", { d: path(f), fill: "none", stroke: css(c), "stroke-width": 2.25, "stroke-linejoin": "round" }, svg);
      el("circle", { cx: x(last[0]), cy: y(f(last[1])), r: 3.5, fill: css(c) }, svg);
      const lab = el("text", { x: x(last[0]) + 8, y: endY[k] + 4, class: "end", fill: css(c) }, svg);
      lab.style.fill = css(c);
      const hp = Math.round(last[1] * 100);
      lab.textContent = `${name} ${k === 0 ? hp : 100 - hp}%`;     // labels always sum to 100
    });
    const cross = el("line", { y1: T, y2: H - B, stroke: css("--muted"), "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const dots = lines.map(([, c]) => el("circle", { r: 4.5, fill: css(c), stroke: css("--card"), "stroke-width": 2, visibility: "hidden" }, svg));
    const hit = el("rect", { x: L, y: T, width: Math.max(W - L - R, 1), height: H - T - B, fill: "transparent" }, svg);
    const move = ev => {
      const r = svg.getBoundingClientRect(), px = Math.min(Math.max(ev.clientX - r.left, x(pts[0][0])), x(last[0]));
      let k = 0;
      while (k + 1 < pts.length && x(pts[k + 1][0]) <= px) k++;
      const [t, p] = pts[k];
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      lines.forEach(([f], i) => { dots[i].setAttribute("cx", px); dots[i].setAttribute("cy", y(f(p))); dots[i].setAttribute("visibility", "visible"); });
      tip.innerHTML = `<b>${fmt(t)}</b><br>${ha} ${(p * 100).toFixed(1)}% · ${aa} ${((1 - p) * 100).toFixed(1)}%`;
      tip.hidden = false;
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (ev.clientY - tip.offsetHeight - 12) + "px";
    };
    const leave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", leave);
  }
  function mini(box) {
    const pts = JSON.parse(box.dataset.trend).map(([ts, p]) => [new Date(ts), p]);
    const t0 = day(box.dataset.start), t1 = new Date(day(box.dataset.end).getTime() + 86400000);
    box.innerHTML = "";
    const W = box.clientWidth, H = box.clientHeight, P = 3;
    if (!W) return;
    const x = t => (t - t0) / (t1 - t0) * W, y = p => P + (1 - p) * (H - 2 * P);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
    el("line", { x1: 0, x2: W, y1: y(.5), y2: y(.5), stroke: css("--rule"), "stroke-dasharray": "3 3" }, svg);
    for (let d = new Date(t0.getTime() + 86400000); d < t1; d = new Date(d.getTime() + 86400000))
      el("line", { x1: x(d), x2: x(d), y1: 0, y2: H, stroke: css("--rule"), "stroke-opacity": .5 }, svg);
    [[p => p, "--a"], [p => 1 - p, "--b"]].forEach(([f, c]) => {
      const d = pts.map(([t, p], i) => (i ? `H${x(t)}V${y(f(p))}` : `M${x(t)},${y(f(p))}`)).join("");
      if (pts.length > 1) el("path", { d, fill: "none", stroke: css(c), "stroke-width": 1.75 }, svg);
      const [t, p] = pts[pts.length - 1];
      el("circle", { cx: x(t), cy: y(f(p)), r: 2.5, fill: css(c) }, svg);
    });
  }
  CG.draw = draw;
  CG.mini = mini;
  CG.bindCharts = () => document.querySelectorAll("details.trend").forEach(d => {
    if (d._cg) return;                     // once per element (live refresh keeps elements)
    d._cg = true;
    d.addEventListener("toggle", () => { if (d.open) draw(d.querySelector(".tplot")); else tip.hidden = true; });
  });
  CG.initCharts = () => {
    document.querySelectorAll(".trend .mini").forEach(mini);
    CG.bindCharts();
    document.querySelectorAll("details.trend[open] .tplot").forEach(draw);
  };
  let w = innerWidth;
  addEventListener("resize", () => { if (innerWidth === w) return; w = innerWidth;
    document.querySelectorAll("details.trend[open] .tplot").forEach(draw);
    document.querySelectorAll("details.trend:not([open]) .mini").forEach(mini); });
})();
</script>
<script>window.CG_API = window.CG_API ?? "%(live_url)s";</script>
<script src="app.js"></script>
</body>
</html>
"""


GAME_FIELDS = ("key", "home", "away", "home_abbr", "away_abbr", "p_home", "score", "games", "totals", "lead", "swing",
               "in_play", "trend")
STATE_FIELDS = ("period", "dates", "start", "end", "updated", "v", "ts", "standings", "grid", "h2h", "allplay", "records",
                "top", "worst", "last", "scorecard", "awards", "ranks", "activity")


def _plain(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON: {type(o).__name__}")


def api_state(data):
    """The public API (api/state.json; web/app.js renders it): exactly what the page shows, as plain JSON. Leaves out what
    the page doesn't display: each matchup's category probabilities, sort keys, raw transaction times."""
    games = [{k: g.get(k) for k in GAME_FIELDS} for g in data["games"]]
    for g in games:
        g["trend"] = [[ts, round(float(p), 4)] for ts, p in g["trend"] or []]
    today = None if data.get("today") is None else [
        {**r, "players": [{k: p[k] for k in ("name", "when", "bench", "out")} for p in r["players"]]} for r in data["today"]]
    moves = [{k: v for k, v in m.items() if k != "ts"} for m in data.get("moves") or []]
    state = json.loads(json.dumps({**{k: data.get(k) for k in STATE_FIELDS}, "games": games, "today": today, "moves": moves},
                                  default=_plain))
    shown = {k: v for k, v in state.items() if k not in VOLATILE}       # v names what's displayed, nothing else
    state["v"] = hashlib.sha1(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:12]
    return state


def shell():
    """The page itself: styles, chart and section-nav code, and web/app.js, which builds everything else in the browser
    from the JSON API (LIVE_URL, then this site's own api/state.json if that doesn't answer)."""
    return SHELL_TEMPLATE.replace("%(live_url)s", LIVE_URL)


APP_JS = Path(__file__).parent / "web" / "app.js"
INSIGHTS = Path(__file__).parent / "web" / "insights.html"


def site_files(state=None):
    """{path: text} for a copy of the site: the page, the app, the insights page, and the state if given."""
    files = {"index.html": shell(), "app.js": APP_JS.read_text()}
    if INSIGHTS.exists():
        files["insights.html"] = INSIGHTS.read_text()
    if state is not None:
        files["api/state.json"] = json.dumps(state, separators=(",", ":"))
    return files


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "public_page")
    import live_server as S
    lv = LV.Live()
    try:
        sb = S.scoreboard(datetime.now(ET))
    except Exception:
        sb = None
    try:
        tx = S.transactions(lv.live["latest_scoring_period"])
    except Exception:
        tx = None
    for name, text in site_files(api_state(build(lv, history={}, sb=sb, tx=tx))).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text)
    print("wrote", out, "(open index.html through a local web server; set window.CG_API = '' to use this copy's state)")
