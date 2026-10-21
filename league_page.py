"""Public league page: this week's six matchups with live score and win chances (model L3).

    python3 league_page.py OUT_DIR        # writes OUT_DIR/index.html (self-contained, no player-level data)

Used by publish_league.py, which syncs ESPN, rebuilds this page and pushes it to the GitHub Pages repo.
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import live as LV
import model as M

CATS = M.CATS


def current_score(home_tot, away_tot):
    """Categories won so far (home, away, ties) from live totals."""
    h, a = LV.known_totals(home_tot), LV.known_totals(away_tot)
    w = l = t = 0
    for i, c in enumerate(CATS):
        if c in ("FG%", "FT%"):
            x, at = ("fgx", "fga") if c == "FG%" else ("ftx", "fta")
            if h[M.SK[at]] == 0 and a[M.SK[at]] == 0:
                t += 1
                continue
            hv = h[M.SK[x]] / h[M.SK[at]] if h[M.SK[at]] else -9
            av = a[M.SK[x]] / a[M.SK[at]] if a[M.SK[at]] else -9
        else:
            s = {"3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "PTS": "pts"}[c]
            hv, av = h[M.SK[s]], a[M.SK[s]]
            if c == "TO":
                hv, av = -hv, -av
        if abs(hv - av) < 1e-12:
            t += 1
        elif hv > av:
            w += 1
        else:
            l += 1
    return w, l, t


def build(lv):
    names = {t["id"]: t["name"] for t in lv.lg["teams"]}
    rec = {t["id"]: t.get("record") or [0, 0, 0] for t in lv.lg["teams"]}
    start, end = LV.period_dates(lv.season, lv.period)
    import pandas as pd
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    in_period = sch[(sch.date >= start) & (sch.date <= end)].date
    if len(in_period):                                  # ESPN's matchup dates: first to last game day
        start, end = in_period.min(), in_period.max()
    games = []
    for m in lv.live["matchups"]:
        pc, pw, _, _ = lv.matchup(m["home"], m["away"])
        w, l, t = current_score(m["home_tot"], m["away_tot"])
        lean = [(c, float(p)) for c, p in zip(CATS, pc)]
        games.append({"home": names[m["home"]], "away": names[m["away"]], "p_home": float(pw),
                      "score": [w, l, t], "lean": lean,
                      "rec_home": rec[m["home"]], "rec_away": rec[m["away"]]})
    games.sort(key=lambda g: -abs(g["p_home"] - 0.5))
    return {"period": lv.period, "dates": f"{start:%b %-d} – {end:%b %-d}", "games": games,
            "updated": datetime.now().strftime("%a %b %-d, %-I:%M %p ET")}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="600">
<title>The Cool Guys · Matchup odds</title>
<style>
:root { --bg: #f6f6f2; --card: #fcfcfb; --ink: #15171a; --ink2: #4d5056; --muted: #84878c; --rule: #e1e0d9;
  --a: #2a78d6; --b: #c8501e; --mid: #ece9e2; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg: #0f1011; --card: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
  --muted: #8e8d87; --rule: #2c2c2a; --a: #5b9ce8; --b: #ec7a45; --mid: #2c2c2a; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 14px; }
h1 { font-size: 26px; margin: 0; letter-spacing: -.01em; }
.sub { color: var(--muted); font-size: 13.5px; margin: 2px 0 6px; }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); gap: 10px; align-items: baseline; }
.team { font-weight: 600; overflow-wrap: anywhere; }
.team.r { text-align: right; }
.pct { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
.score { color: var(--muted); font-size: 13px; text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--mid); }
.bar i:first-child { background: var(--a); } .bar i:last-child { background: var(--b); }
.cats { display: flex; flex-wrap: wrap; gap: 4px; }
.cat { font-size: 11.5px; padding: 2px 6px; border-radius: 9px; font-variant-numeric: tabular-nums; border: 1px solid var(--rule); color: var(--ink2); }
.cat.h { border-color: var(--a); color: var(--a); } .cat.w { border-color: var(--b); color: var(--b); }
.note { color: var(--muted); font-size: 12.5px; }
</style>
</head>
<body>
<div class="wrap">
  <div><h1>The Cool Guys · Matchup %(period)s</h1>
  <div class="sub">%(dates)s · chance each team wins its matchup this week · updated %(updated)s</div></div>
  %(cards)s
  <p class="note">Each team's chance to win at least 5 of the 9 categories by Sunday: what's happened so far this week
  plus a projection of every remaining game (player form, schedules, injuries, the 10-starter daily limit). Colored chips
  are categories leaning clearly one way. Model estimates, refreshed every 30 minutes.</p>
</div>
</body>
</html>
"""


def card(g):
    ph = g["p_home"]
    chips = []
    for c, p in g["lean"]:
        if p >= 0.65:
            chips.append(f'<span class="cat h">{c} {p:.0%}</span>')
        elif p <= 0.35:
            chips.append(f'<span class="cat w">{c} {1 - p:.0%}</span>')
    w, l, t = g["score"]
    e = html.escape
    return (f'<div class="card"><div class="row"><div class="team">{e(g["home"])}</div>'
            f'<div class="score">{w}–{l}–{t} now</div><div class="team r">{e(g["away"])}</div></div>'
            f'<div class="row"><div class="pct" style="color:var(--a)">{ph:.0%}</div><div></div>'
            f'<div class="pct" style="color:var(--b);text-align:right">{1 - ph:.0%}</div></div>'
            f'<div class="bar" aria-hidden="true"><i style="width:{100 * ph:.1f}%"></i><i style="width:{100 * (1 - ph):.1f}%"></i></div>'
            f'<div class="cats">{"".join(chips) or "<span class=note>No category is leaning clearly yet.</span>"}</div></div>')


def render(data):
    return PAGE % {"period": data["period"], "dates": html.escape(data["dates"]), "updated": html.escape(data["updated"]),
                   "cards": "\n  ".join(card(g) for g in data["games"])}


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "public_page")
    out.mkdir(parents=True, exist_ok=True)
    lv = LV.Live()
    data = build(lv)
    (out / "index.html").write_text(render(data))
    (out / "data.json").write_text(json.dumps({k: data[k] for k in ("period", "dates", "updated")} |
                                              {"games": [{k: g[k] for k in ("home", "away", "p_home", "score")} for g in data["games"]]}))
    print("wrote", out / "index.html")
