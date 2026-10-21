"""Public league page: this week's matchups (live score, category totals, games, win chances), standings,
all-play record and luck, top performers, and last week's odds against results. Nothing player-level from the
model (no projections, values or recommendations) and no playoff or title odds.

    python3 league_page.py OUT_DIR        # writes OUT_DIR/index.html

Used by publish_league.py, which syncs ESPN, rebuilds this page and pushes it to the GitHub Pages repo.
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import inseason as I
import live as LV
import model as M

CATS = M.CATS
STAT_OF = {"3PM": "tpm", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "tov", "PTS": "pts"}


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


def top_performers(lv, g, start, end, n=6):
    df = lv.season.df
    sch = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
    sp = set(sch[(sch.date >= start) & (sch.date <= end)].scoring_period)
    if not len(g):
        return []
    wk = g[g.scoring_period.isin(sp)]
    owner = {e[0]: t["name"] for t in lv.lg["teams"] for e in t["roster"]}
    wk = wk[wk["id"].isin(owner)]
    if not len(wk):
        return []
    tot = wk.groupby("id")[["pts", "reb", "ast", "stl", "blk", "tpm", "tov", "fgm", "fga", "ftm", "fta"]].sum()
    tot["g"] = wk.groupby("id").size()
    _, scale = M.static_value(df)
    mu = np.zeros((len(tot), len(M.STATK)))
    for s in ("fga", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
        mu[:, M.SK[s]] = tot[s].to_numpy()
    mu[:, M.SK["fgx"]] = tot["fgm"] - M.LG_FG * tot["fga"]
    mu[:, M.SK["ftx"]] = tot["ftm"] - M.LG_FT * tot["fta"]
    tot["score"] = M.value_from(mu, np.ones(len(tot)), scale)
    names = g.drop_duplicates("id").set_index("id")["name"]
    out = []
    for pid, r in tot.sort_values("score", ascending=False).head(n).iterrows():
        out.append({"player": names.get(pid, str(pid)), "team": owner[pid], "g": int(r.g),
                    "line": f"{r.pts:.0f} pts · {r.reb:.0f} reb · {r.ast:.0f} ast · {r.stl:.0f} stl · {r.blk:.0f} blk · "
                            f"{r.tpm:.0f} 3pm · {r.fgm:.0f}-{r.fga:.0f} FG"})
    return out


def build(lv, history=None):
    """Page data; `history` ({period: {matchup: odds at the first update}}) is updated in place."""
    names = {t["id"]: t["name"] for t in lv.lg["teams"]}
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
                      "p_home": float(pw), "score": list(compare(m["home_tot"], m["away_tot"])),
                      "lean": [(c, float(p)) for c, p in zip(CATS, pc)],
                      "totals": [[c, fmt(m["home_tot"], c), fmt(m["away_tot"], c)] for c in CATS],
                      "lead": [compare_one(m["home_tot"], m["away_tot"], c) for c in CATS],
                      "games": [gp[m["home"]], gp[m["away"]]]})
    games.sort(key=lambda x: abs(x["p_home"] - 0.5))
    if history is not None:
        history.setdefault(str(lv.period), {x["key"]: round(x["p_home"], 3) for x in games})
    last = last_week(lv.lg, history or {})
    return {"period": lv.period, "dates": f"{start:%b %-d} – {end:%b %-d}", "games": games,
            "standings": standings(lv.lg), "allplay": all_play(lv.lg), "top": top_performers(lv, g, start, end),
            "last": last, "updated": datetime.now().strftime("%a %b %-d, %-I:%M %p ET")}


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
<meta http-equiv="refresh" content="600">
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
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; min-width: 0; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); gap: 10px; align-items: baseline; }
.team { font-weight: 600; overflow-wrap: anywhere; }
.r { text-align: right; }
.pct { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
.mid { color: var(--muted); font-size: 13px; text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--mid); }
.bar i:first-child { background: var(--a); } .bar i:last-child { background: var(--b); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13.5px; }
th, td { padding: 5px 7px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: 0; }
td.lead { font-weight: 700; }
.ha { color: var(--a); } .hb { color: var(--b); }
.pos { color: var(--good); } .neg { color: var(--bad); }
.note { color: var(--muted); font-size: 12.5px; }
.card.tbl { padding: 6px 10px; }
</style>
</head>
<body>
<div class="wrap">
  <div><h1>The Cool Guys · Matchup %(period)s</h1>
  <div class="sub">%(dates)s · updated %(updated)s</div></div>
  %(cards)s
  <p class="note">Win chance: each team's chance to win at least 5 of the 9 categories by Sunday, from what's happened so
  far plus a projection of every remaining game. Games: player-games already played this week · scheduled games left.
  Refreshed every 30 minutes.</p>
  %(standings)s
  %(allplay)s
  %(top)s
  %(last)s
</div>
</body>
</html>
"""


def card(gm):
    e = html.escape
    ph = gm["p_home"]
    w, l, t = gm["score"]
    (hp, hl), (ap, al) = gm["games"]
    head = "".join(f"<th>{c}</th>" for c in CATS)
    rh = "".join(f'<td class="{"lead" if s == 1 else ""}">{h}</td>' for (c, h, a), s in zip(gm["totals"], gm["lead"]))
    ra = "".join(f'<td class="{"lead" if s == -1 else ""}">{a}</td>' for (c, h, a), s in zip(gm["totals"], gm["lead"]))
    return (f'<div class="card"><div class="row"><div class="team ha">{e(gm["home"])}</div>'
            f'<div class="mid">{w}–{l}–{t} now</div><div class="team hb r">{e(gm["away"])}</div></div>'
            f'<div class="row"><div class="pct ha">{ph:.0%}</div><div class="mid">win chance</div>'
            f'<div class="pct hb r">{1 - ph:.0%}</div></div>'
            f'<div class="bar" aria-hidden="true"><i style="width:{100 * ph:.1f}%"></i><i style="width:{100 * (1 - ph):.1f}%"></i></div>'
            f'<div class="scroll"><table><tr><th></th>{head}</tr>'
            f'<tr><td class="ha">{e(gm["home"][:16])}</td>{rh}</tr><tr><td class="hb">{e(gm["away"][:16])}</td>{ra}</tr></table></div>'
            f'<div class="row"><div class="mid" style="text-align:left">games {hp} played · {hl} left</div><div></div>'
            f'<div class="mid r">games {ap} played · {al} left</div></div></div>')


def render(data):
    e = html.escape
    st = "".join(f'<tr><td>{i + 1}. {e(r["team"])}</td><td>{r["w"]}–{r["l"]}–{r["t"]}</td><td>{r["pct"]:.3f}</td>'
                 f'<td>{"—" if r["gb"] == 0 else f"{r["gb"]:g}"}</td></tr>' for i, r in enumerate(data["standings"]))
    if all(r["w"] + r["l"] + r["t"] == 0 for r in data["standings"]):
        standings_html = '<h2>Standings</h2><p class="note">Appear once the first matchup is final.</p>'
    else:
        standings_html = (f'<h2>Standings</h2><div class="card tbl scroll"><table><tr><th>Team</th><th>Categories W–L–T</th>'
                          f'<th>Pct</th><th>GB</th></tr>{st}</table></div>')
    if data["allplay"]:
        ap = "".join(f'<tr><td>{e(r["team"])}</td><td>{r["mw"]}–{r["ml"]}–{r["mt"]}</td><td>{r["allplay_pct"]:.3f}</td>'
                     f'<td>{r["actual_pct"]:.3f}</td><td class="{"pos" if r["luck"] > 0 else "neg" if r["luck"] < 0 else ""}">'
                     f'{100 * r["luck"]:+.1f}</td></tr>' for r in data["allplay"])
        allplay_html = (f'<h2>All-play record and luck</h2><div class="card tbl scroll"><table><tr><th>Team</th>'
                        f'<th>All-play W–L–T</th><th>All-play cat pct</th><th>Actual cat pct</th><th>Luck</th></tr>{ap}</table></div>'
                        f'<p class="note">All-play: every team against every other team, every finished week. Luck: actual category '
                        f'win percentage minus all-play, in points. Positive means your schedule has been kind.</p>')
    else:
        allplay_html = '<h2>All-play record and luck</h2><p class="note">Appears once the first matchup is final.</p>'
    if data["top"]:
        tp = "".join(f'<tr><td>{e(r["player"])}</td><td style="text-align:left">{e(r["team"])}</td><td>{r["g"]}</td>'
                     f'<td style="text-align:left">{e(r["line"])}</td></tr>' for r in data["top"])
        top_html = (f'<h2>Top performers this week</h2><div class="card tbl scroll"><table><tr><th>Player</th>'
                    f'<th style="text-align:left">Fantasy team</th><th>G</th><th style="text-align:left">Line</th></tr>{tp}</table></div>'
                    f'<p class="note">Rostered players, games through last night, ranked by 9-category production.</p>')
    else:
        top_html = ""
    if data["last"]:
        lr = []
        for r in data["last"]["rows"]:
            p = r["p_home"]
            odds = "—" if p is None else f"{p:.0%} / {1 - p:.0%}"
            win = "Tie" if r["tie"] else (r["home"] if r["home_won"] else r["away"])
            fav_won = p is not None and not r["tie"] and ((p >= 0.5) == r["home_won"])
            lr.append(f'<tr><td>{e(r["home"])} vs {e(r["away"])}</td><td>{odds}</td><td>{r["score"][0]}–{r["score"][1]}–{r["score"][2]}</td>'
                      f'<td style="text-align:left">{e(win)}{"" if p is None else (" ✓" if fav_won else " (upset)")}</td></tr>')
        last_html = (f'<h2>Matchup {data["last"]["period"]}: odds against results</h2><div class="card tbl scroll"><table>'
                     f'<tr><th>Matchup</th><th>Odds at first update</th><th>Final</th><th style="text-align:left">Winner</th></tr>'
                     f'{"".join(lr)}</table></div>')
    else:
        last_html = ""
    fill = {"period": str(data["period"]), "dates": e(data["dates"]), "updated": e(data["updated"]),
            "cards": "\n  ".join(card(gm) for gm in data["games"]), "standings": standings_html,
            "allplay": allplay_html, "top": top_html, "last": last_html}
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
    lv = LV.Live()
    data = build(lv, history={})
    (out / "index.html").write_text(render(data))
    print("wrote", out / "index.html")
