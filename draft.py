"""Live draft assistant.

    python3 draft.py            # resumes data/draft_state.json if present
    python3 draft.py --new      # fresh draft

Each line you type is either a command or the name of the player just drafted (fuzzy matched;
recorded for whoever is on the clock). When you are on the clock it prints recommendations.

Commands:
  rec [n]       recommendations now (n rollouts, default 400)
  board [n]     best available by value, with P(still there at your next pick)
  team [t]      roster + projected category win probs (t = team slot, default you)
  undo          remove last pick
  other         record an unmodeled player for the team on the clock
  order         show upcoming picks
  quit
"""
import difflib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import model as M
import sim as S

STATE = Path(__file__).parent / "data" / "draft_state.json"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


ALIASES = {"wemby": "victor wembanyama", "sga": "shai gilgeousalexander", "shai": "shai gilgeousalexander",
           "kat": "karlanthony towns", "ant": "anthony edwards", "giannis": "giannis antetokounmpo",
           "jjj": "jaren jackson jr", "nav": "nickeil alexanderwalker", "jdub": "jalen williams",
           "lebron": "lebron james", "luka": "luka doncic", "jokic": "nikola jokic", "kd": "kevin durant",
           "cp3": "chris paul", "pg13": "paul george", "spida": "donovan mitchell", "zion": "zion williamson"}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch == " ").strip()


# Shipped configuration (set from the acceptance tests in DECISIONS.md).
# Shipped 2026-10-05 (DECISIONS.md): belief layer V2 (season-varying market weights), P(championship),
# calibrated matchup sd.
LIVE_CONFIG = {"objective": "payout", "kappa": 1.182, "proj": "stack_belief"}
TIME_BUDGET_S = 35  # per recommendation; ESPN pick clock is 90s


class App:
    def __init__(self, new: bool, slot: int):
        M.set_kappa(LIVE_CONFIG["kappa"])
        self.budget = TIME_BUDGET_S
        self.season = M.get_season(M.LIVE_SEASON, LIVE_CONFIG["proj"])
        self.df = self.season.df
        picks = []
        if STATE.exists() and not new:
            st = json.loads(STATE.read_text())
            slot, picks = st["slot"], [self.id_to_idx(i) for i in st["pick_ids"]]
        self.d = S.Draft(self.season, my_slot=slot, picks=picks, objective=LIVE_CONFIG["objective"])
        self.names = [norm_name(n) for n in self.df["name"]]

    def id_to_idx(self, pid):
        if pid is None:
            return -1
        hit = np.where(self.df["id"].to_numpy() == pid)[0]
        return int(hit[0]) if len(hit) else -1

    def save(self):
        ids = [None if p < 0 else int(self.df.at[p, "id"]) for p in self.d.picks]
        STATE.write_text(json.dumps({"slot": self.d.my_slot, "pick_ids": ids}))

    # ---------------------------------------------------------------
    def on_clock(self):
        k = len(self.d.picks)
        if k >= self.d.n_picks:
            return None, None
        return k, self.d.order[k]

    def header(self):
        k, t = self.on_clock()
        if k is None:
            print("Draft complete.")
            return
        r, i = divmod(k, M.N_TEAMS)
        who = "YOU" if t == self.d.me else f"team {t + 1}"
        mine = self.d.my_next_picks()
        until = (mine[0] - k) if mine else None
        print(f"\nPick {k + 1} (round {r + 1}.{i + 1}) — {who} on the clock"
              + ("" if t == self.d.me else f"  | you pick in {until}"))

    def match(self, text: str):
        q = norm_name(text)
        q = ALIASES.get(q.replace(" ", ""), q)
        avail = self.d.available()
        exact = [i for i, n in enumerate(self.names) if n == q]
        subs = [i for i, n in enumerate(self.names) if q in n]
        cands = exact or subs
        if not cands:
            close = difflib.get_close_matches(q, self.names, n=5, cutoff=0.6)
            cands = [self.names.index(c) for c in close]
        cands = [c for c in cands if avail[c]] or cands
        return sorted(cands, key=lambda c: self.df.at[c, "mkt"])

    def record(self, idx: int):
        k, t = self.on_clock()
        self.d.picks.append(idx)
        name = "(unmodeled player)" if idx < 0 else self.df.at[idx, "name"]
        print(f"  #{k + 1} team {t + 1}{' (you)' if t == self.d.me else ''}: {name}")
        self.save()

    # ---------------------------------------------------------------
    def rec(self, n_roll: int = 300, top: int = 10):
        k, t = self.on_clock()
        if t != self.d.me:
            print("  (not your pick — showing what you'd take if it were)")
            # temporarily evaluate as if on the clock is not supported; show board instead
            return self.board()
        cands = self.d.candidate_pool(16)
        res = self.d.evaluate_adaptive(cands, budget_s=self.budget, batch=n_roll)
        surv = self.d.survival(n_roll=n_roll)
        res["surv_next"] = surv[res["idx"].to_numpy()]
        res["mkt"] = self.df.loc[res["idx"], "mkt"].to_numpy()
        res["team"] = self.df.loc[res["idx"], "team"].to_numpy()
        res["elig"] = self.df.loc[res["idx"], "elig"].to_numpy()
        res["dJ"] = res["J"] - res["J"].iloc[0]
        a = res.attrs
        print(f"  {a['batches']} batches, {a['seconds']}s — "
              + ("top choice is clearly best" if a["resolved"] else "top choices are a statistical tie (pick on fit)"))
        cols = ["name", "team", "elig", "mkt", "J", "dJ", "se_vs_best", "cats_won", "surv_next"]
        out = res[cols].head(top).copy()
        metric = {"champ": "P(title)", "payout": "E[$ won]"}.get(self.d.objective, "P(win wk)")
        out.columns = ["player", "tm", "pos", "ADP", metric, "vs best", "±se", "cats/wk", "P(avail next turn)"]
        print(out.round(3).to_string(index=False))
        w = res["weights"].iloc[0]
        print("  category weights after top pick:", "  ".join(f"{c} {x:.2f}" for c, x in zip(M.CATS, w)))

        top_row = res.iloc[0]
        tip = ("  tip: top choice likely there next time too — consider the 2nd-best now"
               if top_row["surv_next"] > 0.7 and len(res) > 1 and res["surv_next"].iloc[1] < 0.4 else "")
        if tip:
            print(tip)

    def board(self, n: int = 25):
        avail = np.where(self.d.available())[0]
        mine = self.d.roster(self.d.me)
        w = self.d.need_weights(mine) if mine else np.ones(9)
        v = self.d.Z[avail] @ w
        o = avail[np.argsort(-v)][:n]
        surv = self.d.survival(n_roll=300)
        tbl = self.df.loc[o, ["name", "team", "elig", "mkt", "gp"]].copy()
        tbl["value"] = (self.d.Z[o] @ w).round(2)
        for i, c in enumerate(M.CATS):
            tbl[c] = self.d.Z[o, i].round(1)
        tbl["P(avail next turn)"] = surv[o].round(2)
        print(tbl.to_string(index=False))

    def team(self, slot=None):
        t = self.d.me if slot is None else slot - 1
        r = self.d.roster(t)
        if not r:
            print("  (empty)")
            return
        print("  " + ", ".join("(other)" if p < 0 else self.df.at[p, "name"] for p in r))
        rosters = [self.d.roster(x) for x in range(M.N_TEAMS)]
        R = np.full((M.N_TEAMS, M.ROSTER), -1)
        for x, rr in enumerate(rosters):
            R[x, : len(rr)] = rr
        m, v = self.d.ev.team_stats(R)
        opp = [x for x in range(M.N_TEAMS) if x != t]
        p = self.d.ev.cat_probs(m[t : t + 1, 0], v[t : t + 1, 0], m[opp, 0], v[opp, 0])
        print("  vs current rosters (empty slots = replacement level):")
        print("  " + "  ".join(f"{c} {x:.2f}" for c, x in zip(M.CATS, p.mean(0))))
        print(f"  P(win wk) {self.d.ev.matchup_win(p).mean():.3f}")

    def order(self, n: int = 15):
        k = len(self.d.picks)
        print("  " + "  ".join(f"{j + 1}:{'YOU' if self.d.order[j] == self.d.me else self.d.order[j] + 1}"
                               for j in range(k, min(k + n, self.d.n_picks))))

    # ---------------------------------------------------------------
    def loop(self):
        print(f"Draft assistant — you are slot {self.d.my_slot}. {len(self.df)} players modeled. 'help' for commands.")
        while True:
            self.header()
            k, t = self.on_clock()
            if k is not None and t == self.d.me:
                self.rec()
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            cmd, *args = line.split()
            c = cmd.lower()
            if c in ("quit", "exit", "q"):
                break
            elif c == "help":
                print(__doc__)
            elif c == "rec":
                self.rec(int(args[0]) if args else 800)
            elif c == "board":
                self.board(int(args[0]) if args else 25)
            elif c == "team":
                self.team(int(args[0]) if args else None)
            elif c == "order":
                self.order()
            elif c == "undo":
                if self.d.picks:
                    p = self.d.picks.pop()
                    print("  undid", "(other)" if p < 0 else self.df.at[p, "name"])
                    self.save()
            elif c == "other":
                self.record(-1)
            elif k is None:
                print("  draft complete")
            else:
                cands = self.match(line)
                avail = self.d.available()
                if len(cands) == 1 and avail[cands[0]]:
                    self.record(cands[0])
                elif not cands:
                    print("  no match. Use 'other' if the player isn't modeled.")
                else:
                    opts = [c for c in cands if avail[c]][:6]
                    if not opts:
                        print("  already drafted:", ", ".join(self.df.loc[cands[:3], "name"]))
                        continue
                    for j, c2 in enumerate(opts):
                        print(f"   {j + 1}) {self.df.at[c2, 'name']} ({self.df.at[c2, 'team']})")
                    sel = input("  which #? ").strip()
                    if sel.isdigit() and 1 <= int(sel) <= len(opts):
                        self.record(opts[int(sel) - 1])


if __name__ == "__main__":
    new = "--new" in sys.argv
    slot = 12
    for a in sys.argv[1:]:
        if a.startswith("--slot="):
            slot = int(a.split("=")[1])
    App(new=new, slot=slot).loop()
