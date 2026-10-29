"""Weekly runner: evaluate performance, update the odds from the real standings, and rank actions.

    python3 sync.py        # or run web/espn_sync.js in a signed-in browser -> data/league_2027.json
    python3 weekly.py      # report -> data/weekly/report_<date>.md, prediction log, odds history
    python3 build_lab.py && python3 build_lab.py --espn   # refreshed dashboards (weekly.py writes their data)

Every action is scored on the same simulated seasons as the baseline (common random numbers), so the
change in expected prize money is precise even when the level is not.
"""
import itertools
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

import dashboard_data
import inseason as I
import league as L
import model as M
import sim as S
from draft import LIVE_CONFIG

MY_TEAM = 13
POSITION_LIMITS = {"C": 4}        # ESPN rosterSettings.positionLimits (by default position), this league
R, K, SEED = 600, 32, 2027
OUT = M.DATA / "weekly"
CAT_LIST = M.CATS


class Ctx:
    """Everything one run needs: in-season season, rosters, truth draws, schedule state."""

    def __init__(self, proj=LIVE_CONFIG["proj"]):
        M.set_kappa(LIVE_CONFIG["kappa"])
        self.lg = I.load_league()
        self.g = I.gamelogs()
        self.season = I.inseason_season(M.get_season(M.LIVE_SEASON, proj), self.g)
        self.df = self.season.df
        B = I.build(self.season, self.lg, self.g)               # availability: participation, injuries, IR
        self.B = B
        self.rost, self.tids, self.names, self.unmodeled, self.ir = B["rost"], B["tids"], B["names"], B["unmodeled"], B["ir_dict"]
        self.irs = B["ir"]
        self.me = self.tids.index(MY_TEAM)
        self.pairs, self.tot0, self.weeks_left, self.sched = B["pairs"], B["tot0"], B["weeks_left"], B["sched"]
        self.h2h0 = B["h2h0"]
        self.G = B["Geff"]                                       # each remaining matchup's games x availability
        import live as LV
        self.lv = None                                           # the live engine for the week in progress
        if LV.LIVE_FILE.exists() and self.weeks_left:
            lv = LV.Live()
            if lv.period == self.weeks_left[0] and np.array_equal(lv.season.df["id"], self.df["id"]):
                self.lv = lv
        self.week0 = None if self.lv is None else self.lv.week0_pc(self.pairs[0])   # from live totals
        self.ev = M.Evaluator(self.season, proj_unc=True)
        self.truth = self.ev.draw_truth(R, np.random.default_rng(SEED))
        self.val, _ = M.static_value(self.df)
        self.Z = np.stack([M.static_value.last_by_cat[c] for c in CAT_LIST], axis=1)
        self.base = self.outcomes(self.rost)
        self.wts = self.team_weights()
        self.limits = self.lg.get("position_limits") or POSITION_LIMITS
        self.ir_ids = {t: {e[0] for e in v} for t, v in self.ir.items()}
        sched = pd.read_csv(M.DATA / "schedule_2027.csv", parse_dates=["date"])
        self.days = sched[(sched.date >= I.game_day()) & (sched.date < self.season.playoff_dates[0])]
        self.bench_base = self.benched_games(self.rost[self.me])

    def roster_ok(self, roster, team_slot):
        """ESPN position limits on the whole roster (active + IR), by each player's default position."""
        pos = list(self.df["pos"].to_numpy()[[p for p in roster if p >= 0]])
        idx = {int(i): k for k, i in enumerate(self.df["id"])}
        pos += [self.df["pos"][idx[i]] for i in self.ir_ids.get(self.tids[team_slot], set()) if i in idx]
        return all(pos.count(k) <= v for k, v in self.limits.items())

    def benched_games(self, roster, n_sims=4):
        """Expected rest-of-regular-season games our players play but cannot start (10 slots, eligibility)."""
        import research_lineup as RL
        r = [p for p in roster if p >= 0]
        team = self.df["pro_team_id"].to_numpy()[r]
        avail = self.df["avail"].to_numpy()[r]
        E = np.array([[RL.eligible(e, sl) for sl in RL.SLOTS] for e in self.df["elig"].to_numpy()[r]], float)
        rng = np.random.default_rng(7)
        lost = 0.0
        for _ in range(n_sims):
            for _, day in self.days.groupby("date"):
                plays = np.isin(team, day["pro_team_id"].to_numpy()) & (rng.random(len(r)) < avail)
                if plays.sum() > 0:
                    from scipy.optimize import linear_sum_assignment
                    a, b = linear_sum_assignment(-E[plays])
                    lost += plays.sum() - E[plays][a, b].sum()
        return lost / n_sims

    def games(self, rost):
        """Each remaining matchup's games x availability for these rosters (IR returns displace each roster's own
        weakest players; for the league's rosters this is self.G)."""
        return I.displaced_games(self.B["G"], self.B["A"], rost, self.irs, self.val)

    def week0_for(self, rost):
        """The week in progress (live category odds) with these rosters. A player who changes team (added or traded)
        starts on the bench, as ESPN places him, so today's started games are locked for him (K1)."""
        if self.lv is None or np.array_equal(rost, self.rost):
            return self.week0
        team0, team1 = np.full(len(self.df), -1), np.full(len(self.df), -1)
        for s in range(len(rost)):
            team0[self.rost[s][self.rost[s] >= 0]] = s
            team1[rost[s][rost[s] >= 0]] = s
        moved = np.where((team1 >= 0) & (team1 != team0))[0]
        keep = self.lv.slot_now[moved].copy()
        self.lv.slot_now[moved] = 12
        try:
            return self.lv.week0_pc(self.pairs[0], rost)
        finally:
            self.lv.slot_now[moved] = keep

    def outcomes(self, rost):
        full = I.full(rost, self.irs)
        RR = np.broadcast_to(full[None], (R,) + full.shape).copy()
        mS, vS = self.ev.truth_team_stats(self.truth, RR, g=self.games(rost))
        return L.outcomes_by_rollout(mS, vS, np.random.default_rng(SEED + 1), K=K, pairs=self.pairs, tot0=self.tot0,
                                     per_week=True, week0_pc=self.week0_for(rost), h2h0=self.h2h0)

    def delta(self, rost, teams):
        o = self.outcomes(rost)
        res = {}
        for t in teams:
            d = o["dollars"][:, t] - self.base["dollars"][:, t]
            res[t] = {"dE$": d.mean(), "se": d.std(ddof=1) / np.sqrt(R),
                      "dTitle": (o["title"][:, t] - self.base["title"][:, t]).mean(),
                      "dPlayoffs": (o["playoffs"][:, t] - self.base["playoffs"][:, t]).mean()}
        return res

    def team_weights(self):
        """Per team: marginal value of each category (d P(win week) / d +0.25 z), normalised to mean 1."""
        d = S.Draft(self.season)
        mA, vA = self.ev.team_stats(self.rost)                          # [T, W, S]
        mB, vB = mA.mean(0, keepdims=True), vA.mean(0, keepdims=True)
        base = d._J(mA, vA, mB, vB)
        w = np.zeros((12, 9))
        for c, (j, shift) in enumerate(d._dz):
            m2 = mA.copy()
            m2[..., j] += shift
            w[:, c] = d._J(m2, vA, mB, vB) - base
        w = np.maximum(w, 1e-4 * w.max(1, keepdims=True))
        return w / w.mean(1, keepdims=True)

    def name(self, p):
        return self.df.name[p]


# ---------------------------------------------------------------- performance
def performance(cx):
    lines = []
    res = [r for r in cx.lg.get("results", []) if MY_TEAM in (r[1], r[2])]
    teams = {t["id"]: t["name"] for t in cx.lg["teams"]}
    preds = {int(p.stem.split("_wk")[1]): json.loads(p.read_text()) for p in OUT.glob("pred_wk*.json")}
    if res:
        lines.append("| Week | Opponent | Result (W-L-T cats) | Predicted P(win) | Categories won |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(res):
            wk, h, a, winner, hw, hl, ht, by = r
            mine_home = h == MY_TEAM
            w, l, t = (hw, hl, ht) if mine_home else (hl, hw, ht)
            won = [c for c, v in by.items() if (v == "WIN") == mine_home and v != "TIE"]
            pr = preds.get(wk, {}).get("p_win")
            lines.append(f"| {wk} | {teams[a if mine_home else h]} | {w}-{l}-{t} | "
                         f"{'' if pr is None else f'{pr:.0%}'} | {', '.join(sorted(won))} |")
        scored = [(preds[r[0]], r) for r in res if r[0] in preds]
        if scored:
            br = np.mean([(p["p_win"] - float(((r[4] > r[5]) if r[1] == MY_TEAM else (r[5] > r[4])))) ** 2 for p, r in scored])
            lines.append(f"\nMatchup predictions scored so far: {len(scored)}; Brier {br:.3f} (0.25 = coin flip).")
    else:
        lines.append("No completed matchups yet.")
    # players: to-date production vs projection
    if len(cx.g):
        g = cx.g.groupby("id").agg(n=("min", "size"), mpg=("min", "mean"))
        rows = []
        for p in cx.rost[cx.me][cx.rost[cx.me] >= 0]:
            pid = int(cx.df["id"][p])
            if pid in g.index and g.loc[pid, "n"] >= 3:
                proj_min = float(cx.df["espn_min"][p]) if "espn_min" in cx.df else np.nan
                rows.append((cx.name(p), int(g.loc[pid, "n"]), g.loc[pid, "mpg"], proj_min))
        if rows:
            lines.append("\n| Player | Games | Minutes/g | Projected min | Flag |\n|---|---|---|---|---|")
            for nm, n, mpg, pm in sorted(rows, key=lambda x: x[2] - x[3]):
                flag = "minutes down" if pm and mpg < 0.85 * pm else ("minutes up" if pm and mpg > 1.15 * pm else "")
                lines.append(f"| {nm} | {n} | {mpg:.1f} | {pm:.1f} | {flag} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- this week
def week_preview(cx):
    period = cx.lg.get("current_period", 1)
    done = set(cx.lg.get("completed_weeks", []))
    if period in done:
        period += 1
    opp_id = next((a if h == MY_TEAM else h for w, h, a in cx.sched if w == period and MY_TEAM in (h, a)), None)
    if opp_id is None:
        return period, None, None
    o = cx.tids.index(opp_id)
    wk = I.PERIOD_WEEKS.get(period, [0])
    k = cx.weeks_left.index(period) if period in cx.weeks_left else None
    if k is not None:
        gw = cx.G[:, k]                                          # this week's games x availability
    else:
        gw = I.games_by_period(cx.season, [period])[:, 0] if period in I.PERIOD_WEEKS else np.zeros(len(cx.df))
    ev = cx.ev
    ev.wmu, ev.wvar, ev.own = ev.weekly(ev.mu, ev.avail, np.repeat(gw[:, None], 4, 1))
    m, v = ev.team_stats(cx.rost[[cx.me, o]])
    pc = M.Evaluator.cat_probs(m[0, 0], v[0, 0], m[1, 0], v[1, 0])
    pw = float(M.Evaluator.matchup_win(pc))
    games = {"me": int(round(gw[cx.rost[cx.me][cx.rost[cx.me] >= 0]].sum())), "opp": int(round(gw[cx.rost[o][cx.rost[o] >= 0]].sum()))}
    cx.week = {"g": gw, "opp": o, "k": k}
    pred = {"period": period, "opponent": cx.names[o], "p_win": pw, "cats": dict(zip(CAT_LIST, map(float, pc))),
            "games": games, "made": pd.Timestamp.now().isoformat(), "model_weeks": wk}
    f = OUT / f"pred_wk{period:02d}.json"
    if not f.exists():                    # logged once, before the week, so it can be scored honestly
        f.write_text(json.dumps(pred, indent=1))
    ev.wmu, ev.wvar, ev.own = ev.weekly(ev.mu, ev.avail)    # restore the typical-week view
    return period, pred, gw


# ---------------------------------------------------------------- actions
def week_win(cx, my_roster):
    """P(win this week's matchup) for an alternative roster of ours (this week's games)."""
    ev, wk = cx.ev, getattr(cx, "week", None)
    if wk is None:
        return None
    g = wk["g"]
    if wk.get("k") is not None:                                  # this roster's own IR displacement
        r = cx.rost.copy()
        r[cx.me] = my_roster
        g = cx.games(r)[:, wk["k"]]
    ev.wmu, ev.wvar, ev.own = ev.weekly(ev.mu, ev.avail, np.repeat(g[:, None], 4, 1))
    m, v = ev.team_stats(np.stack([my_roster, cx.rost[wk["opp"]]]))
    p = float(M.Evaluator.matchup_win(M.Evaluator.cat_probs(m[0, 0], v[0, 0], m[1, 0], v[1, 0])))
    ev.wmu, ev.wvar, ev.own = ev.weekly(ev.mu, ev.avail)
    return p


def waiver_actions(cx, gw, n_add=18, n_drop=4):
    taken = set(cx.rost[cx.rost >= 0].ravel()) | {p for t in cx.ir.values() for p in []}
    rostered_ids = {e[0] for t in cx.lg["teams"] for e in t["roster"]}
    fa = np.array([p for p in range(len(cx.df)) if p not in taken and int(cx.df["id"][p]) not in rostered_ids])
    w = cx.wts[cx.me]
    adds = fa[np.argsort(-(cx.Z[fa] @ w))][:n_add]
    mine = cx.rost[cx.me][cx.rost[cx.me] >= 0]
    drops = mine[np.argsort(cx.val[mine])][:n_drop]
    if gw is not None:                     # streaming candidates: most games this week among decent free agents
        decent = fa[np.argsort(-cx.val[fa])][:80]
        adds = np.unique(np.r_[adds, decent[np.argsort(-gw[decent], kind="stable")][:10]])
    base_week = week_win(cx, cx.rost[cx.me])
    out = []
    for a, d in itertools.product(adds, drops):
        r = cx.rost.copy()
        r[cx.me][r[cx.me] == d] = a
        if not cx.roster_ok(r[cx.me], cx.me):
            continue
        res = cx.delta(r, [cx.me])[cx.me]
        res["benched"] = cx.benched_games(r[cx.me]) - cx.bench_base
        wk = week_win(cx, r[cx.me])
        out.append({"type": "Add/drop", "move": f"Add {cx.name(a)}, drop {cx.name(d)}",
                    "games_this_week": None if gw is None else int(round(gw[a] - gw[d])),
                    "dWeek": None if wk is None else wk - base_week, **res})
    return out


def trade_ideas(cx, per_team=4, top=36):
    me, w_me = cx.me, cx.wts[cx.me]
    mine = [p for p in cx.rost[me] if p >= 0]
    mkt = cx.df["mkt"].to_numpy(float)
    cands = []
    for o in range(12):
        if o == me:
            continue
        theirs = [p for p in cx.rost[o] if p >= 0]
        w_o = cx.wts[o]
        local = []
        for a in mine:                                               # 1-for-1
            for b in theirs:
                gu, gt = cx.Z[b] @ w_me - cx.Z[a] @ w_me, cx.Z[a] @ w_o - cx.Z[b] @ w_o
                if gu > 0.15 and gt > -0.3 and mkt[a] <= mkt[b] + 12:
                    local.append((gu + 0.5 * min(gt, 0), o, (a,), (b,)))
        top_b = sorted(theirs, key=lambda p: -cx.val[p])[:5]
        for b in top_b:                                              # 2-for-1: consolidate depth into a star
            for a1, a2 in itertools.combinations(mine, 2):
                gu = cx.Z[b] @ w_me - (cx.Z[a1] + cx.Z[a2]) @ w_me * 0.5
                gt = (cx.Z[a1] + cx.Z[a2]) @ w_o * 0.5 - cx.Z[b] @ w_o
                if gu > 0.3 and gt > -0.3 and min(mkt[a1], mkt[a2]) <= mkt[b] + 6:
                    local.append((gu + 0.5 * min(gt, 0), o, (a1, a2), (b,)))
        cands += sorted(local, key=lambda x: -x[0])[:per_team]
    cands = sorted(cands, key=lambda x: -x[0])[:top]
    out = []
    for _, o, give, get in cands:
        r = cx.rost.copy()
        r[me] = apply_trade(cx, r[me], give, get, receiving=True, team_slot=me)
        r[o] = apply_trade(cx, r[o], get, give, receiving=True, team_slot=o)
        if not (cx.roster_ok(r[me], me) and cx.roster_ok(r[o], o)):
            continue
        res = cx.delta(r, [me, o])
        res[me]["benched"] = cx.benched_games(r[me]) - cx.bench_base
        out.append({"type": "Trade", "partner": cx.names[o],
                    "move": f"Give {' + '.join(cx.name(p) for p in give)} to {cx.names[o]} for {' + '.join(cx.name(p) for p in get)}",
                    **res[me], "partner_dE$": res[o]["dE$"],
                    "market": "fair" if sum(1 / mkt[p] for p in give) >= 0.8 * sum(1 / mkt[p] for p in get) else "you win on ADP"})
    return out


def apply_trade(cx, roster, out_players, in_players, receiving=True, team_slot=None):
    """Swap players; with more incoming than outgoing, drop the lowest-value players; with fewer, add the best FA
    (a player on no league roster, IR slots included) that keeps team_slot's roster within the position limits."""
    r = [p for p in roster if p >= 0 and p not in out_players] + list(in_players)
    r = sorted(r, key=lambda p: -cx.val[p])[: M.ROSTER]
    if len(r) < M.ROSTER:
        rostered = {e[0] for t in cx.lg["teams"] for e in t["roster"]}
        ids = cx.df["id"].to_numpy()
        for p in np.argsort(-cx.val):
            if len(r) == M.ROSTER:
                break
            if int(ids[p]) in rostered or p in r:
                continue
            if team_slot is not None and not cx.roster_ok(np.array(r + [p]), team_slot):
                continue
            r.append(p)
    arr = np.full(M.ROSTER, -1, int)
    arr[: len(r)] = r
    return arr


def lineup_flags(cx):
    flags = []
    me_team = next(t for t in cx.lg["teams"] if t["id"] == MY_TEAM)
    ir_used = sum(1 for e in me_team["roster"] if e[2] == I.IR_SLOT)
    for e in me_team["roster"]:
        if e[3] in ("OUT", "INJURY_RESERVE") and e[2] != I.IR_SLOT:
            flags.append(f"{e[1]} is {e[3]}: move to IR ({2 - ir_used} IR slot(s) free) and use the open spot.")
    return flags


# ---------------------------------------------------------------- report
def refresh_schedule():
    """Re-pull the NBA schedule so newly scheduled games (e.g. NBA Cup dates) are counted."""
    import fetch
    try:
        sched, _ = fetch.fetch_schedule(M.LIVE_SEASON)
        if len(sched) >= 2400:
            sched.to_csv(M.DATA / "schedule_2027.csv", index=False)
    except Exception as e:
        print("schedule refresh failed, using the saved schedule:", e)


def main():
    OUT.mkdir(exist_ok=True)
    refresh_schedule()
    cx = Ctx()
    period, pred, gw = week_preview(cx)
    b = cx.base
    t_me = cx.me
    E = b["dollars"].mean(0)
    headline = None
    if "--no-dash" not in sys.argv:                 # one source of truth for the odds: the dashboards' run
        dashboard_data.main()
        dashboard_data.main(proj="espn", out_name="dashboard_espn.json")
        import playoff_data
        playoff_data.main()
        headline = next(t for t in json.loads((M.DATA / "dashboard.json").read_text())["teams"] if t["id"] == MY_TEAM)
    hist = OUT / "history.csv"
    row = pd.DataFrame([{"date": str(date.today()), "period": period, "exp_dollars": E[t_me],
                         "p_title": b["title"][:, t_me].mean(), "p_playoffs": b["playoffs"][:, t_me].mean(),
                         "games_played_median": float(np.median(cx.df["games_to_date"]))}])
    row.to_csv(hist, mode="a", header=not hist.exists(), index=False)

    acts = waiver_actions(cx, gw) + trade_ideas(cx)
    A = pd.DataFrame(acts)
    good = A[(A["dE$"] > 2 * A["se"]) & (A["dE$"] > 0.25)].sort_values("dE$", ascending=False)
    trades = A[A.type == "Trade"].sort_values("dE$", ascending=False)
    lines = [f"# Weekly report — {date.today()} (matchup period {period})", ""]
    if headline is not None:
        allE = [t["exp_dollars"] for t in json.loads((M.DATA / "dashboard.json").read_text())["teams"]]
        e_me, ti, po = headline["exp_dollars"], headline["p_1st"], headline["p_playoffs"]
        rank = sum(x > e_me for x in allE) + 1
    else:
        e_me, ti, po = E[t_me], b["title"][:, t_me].mean(), b["playoffs"][:, t_me].mean()
        rank = int((E > E[t_me]).sum()) + 1
    lines += ["## Odds", "",
              f"Expected winnings **${e_me:.2f}** (average team $25), title **{ti:.1%}**, playoffs **{po:.1%}**. "
              f"League rank by expected winnings: {rank} of 12. (Same simulation as the dashboards; the action "
              f"changes below come from a separate paired simulation, so compare them to each other, not to this level.)", ""]
    if pred:
        live_line = ""
        try:
            import live as LV
            lv = LV.Live()
            if lv.period == pred["period"]:
                mine = next(x for x in lv.live["matchups"] if MY_TEAM in (x["home"], x["away"]))
                pw = lv.matchup(mine["home"], mine["away"])[1]
                pw = pw if mine["home"] == MY_TEAM else 1 - pw
                live_line = (f" **Live now: {pw:.0%}** (with this week's results so far; games left you "
                             f"{lv.games_left(MY_TEAM):.0f}, them {lv.games_left(next(t for t in (mine['home'], mine['away']) if t != MY_TEAM)):.0f}).")
        except Exception:
            pass
        lines += [f"## This week vs {pred['opponent']}", "",
                  f"Pre-week P(win) {pred['p_win']:.0%} (games this week: you {pred['games']['me']}, them {pred['games']['opp']}).{live_line}", "",
                  "| " + " | ".join(CAT_LIST) + " |", "|" + "---|" * 9,
                  "| " + " | ".join(f"{pred['cats'][c]:.0%}" for c in CAT_LIST) + " |", ""]
    lines += ["## Performance", "", performance(cx), ""]
    flags = lineup_flags(cx)
    lines += ["## Actions ranked by change in expected winnings", ""]
    if flags:
        lines += ["Lineup: " + " ".join(flags), ""]
    if len(good):
        lines += ["| Action | Δ expected $ | ± | Δ title | Δ playoffs | Partner Δ$ | Δ benched games | Notes |", "|---|---|---|---|---|---|---|---|"]
        for _, r in good.head(12).iterrows():
            note = r.get("market") if r.type == "Trade" else (f"{int(r.games_this_week):+d} games this week" if pd.notna(r.get("games_this_week")) else "")
            pdl = f"{r['partner_dE$']:+.2f}" if r.type == "Trade" else ""
            lines.append(f"| {r.move} | {r['dE$']:+.2f} | {r.se:.2f} | {r.dTitle:+.1%} | {r.dPlayoffs:+.1%} | {pdl} | {r.benched:+.1f} | {note} |")
    else:
        lines.append("No move clears the bar (a gain of more than $0.25 and two standard errors). Hold.")
    ad = A[A.type == "Add/drop"]
    if "dWeek" in ad and ad["dWeek"].notna().any():
        st = ad[(ad["dWeek"] > 0.01) & (ad["dE$"] > -0.5)].sort_values("dWeek", ascending=False).head(6)
        lines += ["", "### Streaming for this week's matchup", "",
                  "Adds that raise this week's win chance without costing more than $0.50 of season value.", ""]
        if len(st):
            lines += ["| Move | Δ P(win this week) | Games this week | Δ season $ |", "|---|---|---|---|"]
            for _, r in st.iterrows():
                lines.append(f"| {r.move} | {r.dWeek:+.1%} | {int(r.games_this_week):+d} | {r['dE$']:+.2f} ± {r.se:.2f} |")
        else:
            lines.append("None worth an add this week.")
    lines += ["", "### Best trade ideas (any sign)", "",
              "| Trade | Δ your $ | ± | Δ their $ | ADP view |", "|---|---|---|---|---|"]
    for _, r in trades.head(8).iterrows():
        lines.append(f"| {r.move} | {r['dE$']:+.2f} | {r.se:.2f} | {r['partner_dE$']:+.2f} | {r.market} |")
    lines += ["", f"_Rules enforced: position limits {cx.limits} (whole roster incl. IR). Benched games: change in rest-of-season "
              f"games your players play but can't start (PG, SG, SF, PF, C, G, F, UTIL x3); currently {cx.bench_base:.0f}._"]
    lines += ["", f"_Simulation: {R} simulated seasons × {K} league runs on the real remaining schedule; "
              f"rates blend projections with {int(np.median(cx.df['games_to_date']))} median games of data._"]
    report = "\n".join(lines)
    f = OUT / f"report_{date.today()}.md"
    f.write_text(report)
    A.to_csv(OUT / f"actions_{date.today()}.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
