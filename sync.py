"""Pull the league snapshot from ESPN into data/league_2027.json (the schema inseason.py reads).

The league is private, so ESPN needs your login cookies. Put them in data/espn_auth.json yourself
(never commit or share it): {"espn_s2": "...", "SWID": "{...}"} - copy both from your browser's cookies for
espn.com. Without that file, run web/espn_sync.js in a signed-in browser tab and save its output instead.

    python3 sync.py
"""
import json
import sys
from datetime import datetime, timezone

import requests

import model as M

LEAGUE, SEASON = 2113064615, 2027
POS = {"1": "PG", "2": "SG", "3": "SF", "4": "PF", "5": "C"}
CAT = {"0": "PTS", "1": "BLK", "2": "STL", "3": "AST", "6": "REB", "11": "TO", "17": "3PM", "19": "FG%", "20": "FT%"}
URL = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{SEASON}/segments/0/leagues/{LEAGUE}"
       "?view=mTeam&view=mRoster&view=mMatchupScore&view=mSettings&view=mStatus")


STAT = {"0": "pts", "1": "blk", "2": "stl", "3": "ast", "6": "reb", "11": "tov", "13": "fgm", "14": "fga",
        "15": "ftm", "16": "fta", "17": "tpm"}


def week_totals(games):
    """{period: {team_id: {stat: total}}} for every matchup with scores (finished weeks and the live one)."""
    out = {}
    for m in games:
        for side in ("home", "away"):
            s = m[side]
            sbs = ((s.get("cumulativeScoreLive") or s.get("cumulativeScore") or {}).get("scoreByStat") or {})
            tot = {STAT[k]: (v.get("score") or 0) for k, v in sbs.items() if k in STAT}
            if tot:
                out.setdefault(str(m["matchupPeriodId"]), {})[str(s["teamId"])] = tot
    return out


def normalize(j):
    def side(s):
        c = s.get("cumulativeScore") or {}
        return [c.get("wins", 0), c.get("losses", 0), c.get("ties", 0)]

    def by_stat(s):
        sbs = (s.get("cumulativeScore") or {}).get("scoreByStat") or {}
        return {CAT[k]: v.get("result") for k, v in sbs.items() if k in CAT}

    games = [m for m in j["schedule"] if m.get("away")]
    decided = [m for m in games if m.get("winner") not in (None, "UNDECIDED")]
    periods = sorted({m["matchupPeriodId"] for m in games})
    return {
        "generated": datetime.now(timezone.utc).isoformat(), "season": SEASON, "league_id": LEAGUE,
        "current_period": j["status"]["currentMatchupPeriod"],
        "position_limits": {POS[k]: v for k, v in j["settings"]["rosterSettings"]["positionLimits"].items()
                            if k in POS and v >= 0},
        "order": j["settings"]["draftSettings"]["pickOrder"],
        "teams": [{"id": t["id"], "name": t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip(),
                   "abbrev": t.get("abbrev"),
                   "record": side({"cumulativeScore": (t.get("record") or {}).get("overall") or {}}),
                   "waiver_rank": t.get("waiverRank"),
                   "roster": [[e["playerId"], e["playerPoolEntry"]["player"]["fullName"], e["lineupSlotId"],
                               e["playerPoolEntry"]["player"].get("injuryStatus") or ""]
                              for e in t["roster"]["entries"]]} for t in j["teams"]],
        "pro_team": {str(e["playerId"]): e["playerPoolEntry"]["player"].get("proTeamId")
                     for t in j["teams"] for e in t["roster"]["entries"]},
        "schedule": [[m["matchupPeriodId"], m["home"]["teamId"], m["away"]["teamId"]] for m in games],
        "results": [[m["matchupPeriodId"], m["home"]["teamId"], m["away"]["teamId"], m["winner"], *side(m["home"]),
                     by_stat(m["home"])] for m in decided],
        "week_totals": week_totals(games),
        "completed_weeks": [p for p in periods
                            if all(m.get("winner") not in (None, "UNDECIDED") for m in games if m["matchupPeriodId"] == p)],
    }


def main():
    auth_file = M.DATA / "espn_auth.json"
    if not auth_file.exists():
        sys.exit("data/espn_auth.json not found: add your ESPN cookies there, or use web/espn_sync.js in the browser.")
    auth = json.loads(auth_file.read_text())
    r = requests.get(URL, cookies={"espn_s2": auth["espn_s2"], "SWID": auth["SWID"]}, timeout=60)
    r.raise_for_status()
    (M.DATA / "league_2027.json").write_text(json.dumps(normalize(r.json())))
    print("synced", URL.split("?")[0])


if __name__ == "__main__":
    main()
