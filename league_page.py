"""Public league page: this week's matchups (live score, category totals, games, win chances and their trend,
categories still in play), who plays today, standings, week-by-week results, head-to-head, all-play record and
luck, league records, top performers, last week's odds against results, the model's season scorecard, and recent
adds/drops. Nothing player-level from the model (no projections, values or recommendations) and no playoff or
title odds.

    python3 league_page.py OUT_DIR        # writes OUT_DIR/index.html

Used by publish_league.py, which syncs ESPN, rebuilds this page and pushes it to the GitHub Pages repo.
"""
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
        sb = {int(t): ("pre", 1.0, "", "") for t in sch[sch.date == now.strftime("%Y-%m-%d")].pro_team_id}

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
            state, _, detail, _ = sb[int(team)]
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
    return {"period": lv.period, "dates": f"{start:%b %-d} – {end:%b %-d}", "start": f"{start:%Y-%m-%d}",
            "end": f"{end:%Y-%m-%d}", "games": games, "today": playing_today(lv.lg, sb, now),
            "standings": standings(lv.lg), "grid": week_grid(lv.lg, games, lv.period), "h2h": head_to_head(lv.lg),
            "allplay": all_play(lv.lg), "records": league_records(lv.lg), **dict(zip(("top", "worst"), performers(lv, g, start, end))),
            "last": last_week(lv.lg, history), "scorecard": scorecard(lv.lg, history),
            "moves": recent_moves(tx, lv.lg, now), "awards": awards(lv.lg, history),
            "ranks": category_ranks(lv.lg), "activity": activity(lv.lg, tx, history), "updated": now.strftime("%a %b %-d, %-I:%M %p ET")}


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
                     "p_home": odds.get(f"{h}-{a}"), "home_won": winner == "HOME", "tie": winner == "UNDECIDED"})
    return {"period": wk, "rows": rows}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="300">
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
.spark { display: grid; gap: 2px; }
.spark svg { width: 100%; height: 34px; display: block; border-radius: 4px; background: var(--mid); }
.days { display: grid; grid-template-columns: repeat(var(--n), 1fr); font-size: 10.5px; color: var(--muted); text-align: center; }
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
<div class="wrap">
  <div><h1>The Cool Guys · Matchup %(period)s</h1>
  <div class="sub">%(dates)s · updated %(updated)s</div>%(headline)s</div>
  %(nav)s
  <div id="matchups" style="display:grid;gap:14px">%(cards)s</div>
  <p class="note">Win chance: each team's chance to win at least 5 of the 9 categories by Sunday, from what's happened so
  far plus a projection of every remaining game. The strip under each bar is that chance through the week (blue share
  at the bottom, one column per day). Toss-ups: categories where either side has at least a 30% chance to win.
  Games: player-games already played this week · scheduled games left. Updated every 10 minutes during games.</p>
  %(today)s
  %(awards)s
  %(standings)s
  %(grid)s
  %(ranks)s
  %(h2h)s
  %(allplay)s
  %(records)s
  %(top)s
  %(worst)s
  %(last)s
  %(scorecard)s
  %(moves)s
  %(activity)s
  %(later)s
  <footer class="foot"><a class="btn" href="insights.html" target="_blank" rel="noopener">Insights: how the odds work ↗</a></footer>
</div>
<script>
(() => {
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
  const io = new IntersectionObserver(es => {
    es.forEach(e => seen.set(e.target.id, e.isIntersecting));
    update();
  }, { rootMargin: "-64px 0px -55% 0px" });
  byId.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  addEventListener("scroll", () => { if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) update(); }, { passive: true });
  links.forEach(a => a.addEventListener("click", () => { mark(a.getAttribute("href").slice(1)); hold = Date.now() + 900; }));
})();
</script>
</body>
</html>
"""


def spark(trend, start, end):
    """Inline SVG of the home side's win chance through the week (shaded like the bar), plus day labels."""
    t0 = datetime.fromisoformat(start)
    days = (datetime.fromisoformat(end) - t0).days + 1
    span = days * 86400
    pts = [(min(300, max(0, (datetime.fromisoformat(ts) - t0).total_seconds() / span * 300)), 34 * (1 - p)) for ts, p in trend]
    if not pts:
        return ""
    if len(pts) == 1:
        pts = pts + [(min(300, pts[0][0] + 3), pts[0][1])]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    x0, xn = pts[0][0], pts[-1][0]
    grid = "".join(f'<line x1="{300 * k / days:.1f}" y1="0" x2="{300 * k / days:.1f}" y2="34" stroke="var(--rule)" '
                   f'vector-effect="non-scaling-stroke"/>' for k in range(1, days))
    labels = "".join(f"<span>{(t0 + timedelta(days=k)):%a}</span>" for k in range(days))
    return (f'<div class="spark"><svg viewBox="0 0 300 34" preserveAspectRatio="none" aria-hidden="true">{grid}'
            f'<polygon points="{x0:.1f},34 {line} {xn:.1f},34" fill="var(--a)" fill-opacity=".35"/>'
            f'<polygon points="{x0:.1f},0 {line} {xn:.1f},0" fill="var(--b)" fill-opacity=".35"/>'
            f'<line x1="0" y1="17" x2="300" y2="17" stroke="var(--muted)" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
            f'<polyline points="{line}" fill="none" stroke="var(--ink2)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'
            f'</svg><div class="days" style="--n:{days}">{labels}</div></div>')


def card(gm, start, end):
    e = html.escape
    ph = gm["p_home"]
    w, l, t = gm["score"]
    (hp, hl), (ap, al) = gm["games"]
    ha, aa = e(gm.get("home_abbr") or gm["home"][:3].upper()), e(gm.get("away_abbr") or gm["away"][:3].upper())
    head = "".join(f"<th>{c}</th>" for c in CATS)
    rh = "".join(f'<td class="{"lead ha" if s == 1 else ""}">{h}</td>' for (c, h, a), s in zip(gm["totals"], gm["lead"]))
    ra = "".join(f'<td class="{"lead hb" if s == -1 else ""}">{a}</td>' for (c, h, a), s in zip(gm["totals"], gm["lead"]))
    bits = ["chance to win the week"]
    sw = gm.get("swing")
    if sw is not None and abs(sw) >= 0.01:
        bits.append(f'<span class="{"ha" if sw > 0 else "hb"}">{ha if sw > 0 else aa} +{100 * abs(sw):.0f}</span> since yesterday')
    if t:
        bits.append(f"{t} tied")
    ip = gm.get("in_play")
    if ip:
        chips = "".join(f'<span class="chip"><b>{e(c)}</b> {e(txt)}</span>' for c, txt in ip)
        inplay = f'<div class="inplay"><span class="lbl">Toss-ups</span>{chips}</div>'
    elif ip is not None:
        inplay = '<div class="inplay"><span class="lbl">No toss-up categories left.</span></div>'
    else:
        inplay = ""
    return (f'<div class="card">'
            f'<div class="matchup">'
            f'<div class="side"><div class="team ha">{e(gm["home"])}</div><div class="pct ha">{ph:.0%}</div>'
            f'<div class="meta">leads {w} of 9 · {hp} played, {hl} left</div></div>'
            f'<div class="side r"><div class="team hb">{e(gm["away"])}</div><div class="pct hb">{1 - ph:.0%}</div>'
            f'<div class="meta">leads {l} of 9 · {ap} played, {al} left</div></div>'
            f'</div>'
            f'<div class="bar" aria-hidden="true"><i style="width:{100 * ph:.1f}%"></i><i style="width:{100 * (1 - ph):.1f}%"></i></div>'
            f'<div class="caption">{" · ".join(bits)}</div>'
            f'{spark(gm.get("trend") or [], start, end)}'
            f'<div class="scroll"><table class="cats"><tr><th></th>{head}</tr>'
            f'<tr><td><span class="dot da"></span>{ha}</td>{rh}</tr>'
            f'<tr><td><span class="dot db"></span>{aa}</td>{ra}</tr></table></div>'
            f'{inplay}'
            f'</div>')


def section(title, body, note="", sid=""):
    return (f'<section id="{sid}" style="display:grid;gap:14px"><h2>{title}</h2><div class="card tbl scroll">{body}</div>'
            + (f'<p class="note">{note}</p>' if note else "") + "</section>")


NAV = [("matchups", "Matchups"), ("today", "Today"), ("awards", "Awards"), ("standings", "Standings"),
       ("weeks", "Weekly"), ("ranks", "Cat ranks"), ("h2h", "Head to head"), ("allplay", "All-play"),
       ("records", "Records"), ("top", "Best players"), ("worst", "Worst players"), ("odds", "Odds"), ("moves", "Moves"), ("activity", "Activity")]


def render(data):
    e = html.escape
    finished = any(r["w"] + r["l"] + r["t"] for r in data["standings"])

    headline = ""
    moved = [g for g in data["games"] if g.get("swing") is not None and abs(g["swing"]) >= 0.05]
    if moved:
        g = max(moved, key=lambda g: abs(g["swing"]))
        sw, p = g["swing"], g["p_home"]
        team, now, before = (g["home"], p, p - sw) if sw > 0 else (g["away"], 1 - p, 1 - p + sw)
        headline = (f'<div class="headline">Biggest swing since yesterday: <b>{e(team)}</b> '
                    f'{before:.0%} → {now:.0%}</div>')

    today = ""
    if data.get("today") and any(r["n"] for r in data["today"]):
        rows = []
        for r in data["today"]:
            who = "".join(f'<span class="{"out" if p["out"] else "bn" if p["bench"] else ""}">{e(p["name"])} '
                          f'{e(p["when"])}{" (bench)" if p["bench"] and not p["out"] else ""}</span>' for p in r["players"])
            rows.append(f'<tr><td>{e(r["team"])}</td><td>{r["lineup"]}</td><td>{r["n"]}</td><td class="who">{who or "—"}</td></tr>')
        today = section("Playing today", f'<table><tr><th>Team</th><th>In lineup</th><th>With a game</th><th class="l">Who · tip-off</th></tr>'
                        f'{"".join(rows)}</table>',
                        "In lineup: players with a game today in an active slot and not ruled out. Bench in grey, "
                        "ruled out struck through. Times Eastern.", "today")
    elif data.get("today") is not None:
        today = '<section id="today"><h2>Playing today</h2><p class="note">No NBA games today.</p></section>'

    standings_html = ""
    if finished:
        st = "".join(f'<tr><td>{i + 1}. {e(r["team"])}</td><td>{r["w"]}–{r["l"]}–{r["t"]}</td><td>{r["pct"]:.3f}</td>'
                     f'<td>{"—" if r["gb"] == 0 else f"{r["gb"]:g}"}</td></tr>' for i, r in enumerate(data["standings"]))
        standings_html = section("Standings", f'<table><tr><th>Team</th><th>Categories W–L–T</th><th>Pct</th><th>GB</th></tr>{st}</table>', sid="standings")

    grid_html = ""
    gd = data.get("grid")
    if gd:
        head = "".join(f"<th>{w}</th>" for w in gd["weeks"])
        rows = []
        for r in gd["rows"]:
            cells = "".join("<td></td>" if c is None else
                            f'<td class="{"g" + c[3] if c[4] else "live"}">{c[0]}–{c[1]}{f"–{c[2]}" if c[2] else ""}</td>'
                            for c in r["cells"])
            rows.append(f'<tr><td>{e(r["team"])}</td><td>{r["rec"][0]}–{r["rec"][1]}{f"–{r["rec"][2]}" if r["rec"][2] else ""}</td>{cells}</tr>')
        grid_html = section("Week by week", f'<table class="grid"><tr><th>Team</th><th>Matchups</th>{head}</tr>{"".join(rows)}</table>',
                            "Category score in each matchup. Green won, red lost; grey italic is this week, still in progress.", "weeks")

    h2h_html = ""
    hh = data.get("h2h")
    if hh:
        head = "".join(f"<th>{e(c)}</th>" for c in hh["cols"])
        rows = []
        for r in hh["rows"]:
            cells = []
            for c in r["cells"]:
                if c is None:
                    cells.append("<td></td>")
                    continue
                w, l, t = c
                cls = "gw" if w > l else "gl" if l > w else "gt"
                cells.append(f'<td class="{cls}">{w}–{l}{f"–{t}" if t else ""}</td>')
            rows.append(f'<tr><td>{e(r["team"])} <span class="lbl">{e(r["abbr"])}</span></td>{"".join(cells)}</tr>')
        h2h_html = section("Head to head", f'<table class="grid"><tr><th>Team</th>{head}</tr>{"".join(rows)}</table>',
                           "Each row team's matchup record against each opponent.", "h2h")

    allplay_html = ""
    if data["allplay"]:
        ap = "".join(f'<tr><td>{e(r["team"])}</td><td>{r["mw"]}–{r["ml"]}–{r["mt"]}</td><td>{r["allplay_pct"]:.3f}</td>'
                     f'<td>{r["actual_pct"]:.3f}</td><td class="{"pos" if r["luck"] > 0 else "neg" if r["luck"] < 0 else ""}">'
                     f'{100 * r["luck"]:+.1f}</td></tr>' for r in data["allplay"])
        allplay_html = section("All-play record and luck",
                               f'<table><tr><th>Team</th><th>All-play W–L–T</th><th>All-play cat pct</th><th>Actual cat pct</th>'
                               f'<th>Luck</th></tr>{ap}</table>',
                               "All-play: every team against every other team, every finished week. Luck: actual category "
                               "win percentage minus all-play, in points. Positive means your schedule has been kind.", "allplay")

    records_html = ""
    if data.get("records"):
        rr = "".join(f'<tr><td>{e(r["what"])}</td><td class="l">{e(r["team"])}</td><td>{e(r["value"])}</td><td>{r["week"]}</td></tr>'
                     for r in data["records"])
        records_html = section("League records", f'<table><tr><th>Record</th><th class="l">Team</th><th>Value</th><th>Matchup</th></tr>{rr}</table>',
                               "Best single matchups so far. Percentages need at least 75% of the league's average attempts; "
                               "counting stats leave out the two-week All-Star matchup.", "records")

    def players_table(rows, title, sid, note):
        if not rows:
            return ""
        tr = "".join(f'<tr><td>{e(r["player"])}</td><td class="l">{e(r["team"])}</td><td>{r["g"]}</td>'
                     f'<td class="{"pos" if (r["pct"] or 0) > 0 else "neg"}">{"—" if r["pct"] is None else f"{100 * r['pct']:+.0f}%"}</td>'
                     f'<td class="l">{e(r["line"])}</td></tr>' for r in rows)
        return section(title, f'<table><tr><th>Player</th><th class="l">Fantasy team</th><th>G</th><th>vs expected</th>'
                              f'<th class="l">Line</th></tr>{tr}</table>', note, sid)

    how = ("vs expected: 9-category production against what the model expected from the same number of games. "
           "Ranked by the size of the gap, so a big game from a big role outranks a big percentage from a small one. "
           "Rostered players, games through last night.")
    top_html = players_table(data["top"], "Best performers this week", "top", how)
    worst_html = players_table(data.get("worst") or [], "Worst performers this week", "worst",
                               how + " Below −100%: turnovers and missed shots outweighed everything else.")

    last_html = ""
    if data["last"]:
        lr = []
        for r in data["last"]["rows"]:
            p = r["p_home"]
            odds = "—" if p is None else f"{p:.0%} / {1 - p:.0%}"
            win = "Tie" if r["tie"] else (r["home"] if r["home_won"] else r["away"])
            fav_won = p is not None and not r["tie"] and ((p >= 0.5) == r["home_won"])
            lr.append(f'<tr><td>{e(r["home"])} vs {e(r["away"])}</td><td>{odds}</td><td>{r["score"][0]}–{r["score"][1]}–{r["score"][2]}</td>'
                      f'<td class="l">{e(win)}{"" if p is None else (" ✓" if fav_won else " (upset)")}</td></tr>')
        last_html = section(f'Matchup {data["last"]["period"]}: odds against results',
                            f'<table><tr><th>Matchup</th><th>Odds at first update</th><th>Final</th><th class="l">Winner</th></tr>'
                            f'{"".join(lr)}</table>', sid="odds")

    score_html = ""
    sc = data.get("scorecard")
    if sc:
        tiles = [("Favorites", f'{sc["fav"][0]}–{sc["fav"][1]}', "won–lost, ties left out"),
                 ("Strong favorites", f'{sc["big"][0]}–{sc["big"][1]}', "75% or more"),
                 ("Brier score", f'{sc["brier"]:.3f}', "coin flip .250 · lower is better"),
                 ("Avg. confidence", f'{sc["conf"]:.0%}', f'in the favorite, {sc["n"]} matchups')]
        score_html = (f'<section id="{"" if last_html else "odds"}" style="display:grid;gap:14px"><h2>How the odds have done</h2><div class="tiles">' +
                      "".join(f'<div class="card tile"><div class="lbl">{a}</div><div class="big">{b}</div><div class="meta">{c}</div></div>'
                              for a, b, c in tiles) +
                      '</div><p class="note">Every finished matchup, scored on the odds at its first update of the week.</p></section>')

    moves_html = ""
    if data.get("moves"):
        mv = []
        for r in data["moves"]:
            if r["kind"] == "trade":
                mv.append(f'<tr><td>{e(r["when"])}</td><td class="l">Trade</td><td class="who" colspan="2">{e(r["add"])}</td></tr>')
            else:
                mv.append(f'<tr><td>{e(r["when"])}</td><td class="l">{e(r["team"])} <span class="lbl">· {e(r["kind"])}</span></td>'
                          f'<td class="l">{e(r["add"]) or "—"}</td><td class="l">{e(r["drop"]) or "—"}</td></tr>')
        moves_html = section("Recent moves", f'<table><tr><th>When</th><th class="l">Team</th><th class="l">Added</th>'
                             f'<th class="l">Dropped</th></tr>{"".join(mv)}</table>', "Executed adds, drops and trades from the last 7 days.", "moves")

    awards_html = ""
    aw = data.get("awards")
    if aw:
        awards_html = (f'<section id="awards" style="display:grid;gap:14px"><h2>Matchup {aw["period"]} awards</h2><div class="tiles">' +
                       "".join(f'<div class="card tile award"><div class="lbl">{e(a["title"])}</div><div class="big">{e(a["team"])}</div>'
                               f'<div class="meta">{e(a["text"])}</div></div>' for a in aw["items"]) + '</div></section>')

    ranks_html = ""
    rk = data.get("ranks")
    if rk:
        n = rk["n"]

        def heat(r):
            x = (n - r) / (n - 1) if n > 1 else 0.5          # 1 for first, 0 for last
            col, amt = ("--good", (x - 0.5) * 2) if x >= 0.5 else ("--bad", (0.5 - x) * 2)
            return f'background: color-mix(in srgb, var({col}) {6 + 34 * amt:.0f}%, transparent)'
        head = "".join(f"<th>{c}</th>" for c in CATS)
        rows = "".join(f'<tr><td>{e(r["team"])}</td>' + "".join(f'<td class="rk" style="{heat(k)}" title="{e(v)}">{k}</td>'
                                                               for k, v in zip(r["ranks"], r["shown"])) +
                       f'<td>{r["avg"]:.1f}</td></tr>' for r in rk["rows"])
        ranks_html = section("Category ranks", f'<table class="grid"><tr><th>Team</th>{head}<th>Avg</th></tr>{rows}</table>',
                             "Season rank in each category over finished matchups (1 is best; fewest turnovers ranks first). "
                             "Hover or long-press a cell for the season figure.", "ranks")

    activity_html = ""
    if data.get("activity"):
        ac = "".join(f'<tr><td>{e(r["team"])}</td><td>{r["adds"]}</td><td>{r["drops"]}</td><td>{r["trades"]}</td></tr>'
                     for r in data["activity"])
        activity_html = section("Most active managers", f'<table><tr><th>Team</th><th>Adds</th><th>Drops</th><th>Trades</th>'
                                f'</tr>{ac}</table>', "Since the draft, from ESPN's transaction log.", "activity")

    later = "" if finished else ('<p class="note">After the first matchup is final, this page adds standings, week-by-week '
                                 'results, head-to-head records, all-play and luck, league records, and a running scorecard '
                                 'for the odds.</p>')
    fill = {"period": str(data["period"]), "dates": e(data["dates"]), "updated": e(data["updated"]), "headline": headline,
            "cards": "\n  ".join(card(gm, data["start"], data["end"]) for gm in data["games"]), "today": today,
            "standings": standings_html, "grid": grid_html, "h2h": h2h_html, "allplay": allplay_html,
            "records": records_html, "top": top_html, "worst": worst_html, "last": last_html, "scorecard": score_html, "moves": moves_html,
            "later": later, "awards": awards_html, "ranks": ranks_html, "activity": activity_html}
    present = {k for k in ("today", "awards", "standings", "weeks", "ranks", "h2h", "allplay", "records", "top", "worst", "odds",
                           "moves", "activity") if f'id="{k}"' in "".join(fill.values())} | {"matchups"}
    fill["nav"] = ('<nav class="jump" aria-label="Sections">' +
                   "".join(f'<a href="#{k}">{lbl}</a>' for k, lbl in NAV if k in present) + "</nav>")
    page = PAGE
    for k, v in fill.items():
        page = page.replace(f"%({k})s", v)
    return page


def public_json(data):
    return {k: data[k] for k in ("period", "dates", "updated")} | {
        "games": [{k: gm[k] for k in ("home", "away", "p_home", "score")} for gm in data["games"]]}


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "public_page")
    out.mkdir(parents=True, exist_ok=True)
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
    data = build(lv, history={}, sb=sb, tx=tx)
    (out / "index.html").write_text(render(data))
    print("wrote", out / "index.html")
