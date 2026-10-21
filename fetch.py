"""Pull ESPN fantasy basketball player data (projections, actuals, ADP, eligibility) and pro schedules.

Writes data/players_{season}.csv for each season and data/schedule_{season}.csv.
ESPN season id = year the season ends (2027 = 2026-27).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
DATA = Path(__file__).parent / "data"

# ESPN basketball stat ids
STAT_IDS = {
    "0": "pts", "1": "blk", "2": "stl", "3": "ast", "6": "reb", "11": "tov",
    "13": "fgm", "14": "fga", "15": "ftm", "16": "fta", "17": "tpm", "18": "tpa",
    "40": "min", "42": "gp",
}
SLOT_NAMES = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}
POS_NAMES = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}


def fetch_players(season: int, limit: int = 1000) -> list[dict]:
    flt = {
        "players": {
            "limit": limit,
            "filterStatsForSourceIds": {"value": [0, 1]},
            "filterStatsForSplitTypeIds": {"value": [0]},
            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
        }
    }
    r = requests.get(
        BASE.format(season=season) + "/segments/0/leaguedefaults/3",
        params={"view": "kona_player_info"},
        headers={"X-Fantasy-Filter": json.dumps(flt)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["players"]


def flatten(players: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for p in players:
        pl = p["player"]
        own = pl.get("ownership", {}) or {}
        ranks = pl.get("draftRanksByRankType", {}) or {}
        row = {
            "id": pl["id"],
            "name": pl["fullName"],
            "pro_team_id": pl.get("proTeamId"),
            "pos": POS_NAMES.get(pl.get("defaultPositionId")),
            "elig": "/".join(SLOT_NAMES[s] for s in pl.get("eligibleSlots", []) if s in SLOT_NAMES),
            "injury": pl.get("injuryStatus"),
            "adp": own.get("averageDraftPosition"),
            "pct_owned": own.get("percentOwned"),
            "espn_rank": (ranks.get("STANDARD") or {}).get("rank"),
            "espn_roto_rank": (ranks.get("ROTO") or {}).get("rank"),
            "outlook": pl.get("seasonOutlook"),
        }
        for s in pl.get("stats", []):
            if s.get("statSplitTypeId") != 0:
                continue
            avg = s.get("averageStats") or {}
            if not avg:
                continue
            tag = ("proj" if s["statSourceId"] == 1 else "act") + str(s["seasonId"])
            for k, name in STAT_IDS.items():
                if k == "42":
                    row[f"{tag}_gp"] = (s.get("stats") or {}).get("42")
                elif k in avg:
                    row[f"{tag}_{name}"] = avg[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["season"] = season
    return df


def fetch_schedule(season: int) -> pd.DataFrame:
    r = requests.get(BASE.format(season=season), params={"view": "proTeamSchedules_wl"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    rows, teams = [], []
    for t in js["settings"]["proTeams"]:
        teams.append({"pro_team_id": t["id"], "abbrev": t["abbrev"], "team": t["name"]})
        for period, games in (t.get("proGamesByScoringPeriod") or {}).items():
            for g in games:
                rows.append({"pro_team_id": t["id"], "scoring_period": int(period), "date": g["date"], "game_id": g["id"]})
    sched = pd.DataFrame(rows).drop_duplicates(["pro_team_id", "game_id"])
    sched["date"] = pd.to_datetime(sched["date"], unit="ms", utc=True).dt.tz_convert("US/Eastern").dt.date
    return sched, pd.DataFrame(teams)


def main(seasons=(2027, 2026, 2025)):
    DATA.mkdir(exist_ok=True)
    for season in seasons:
        df = flatten(fetch_players(season), season)
        df.to_csv(DATA / f"players_{season}.csv", index=False)
        print(f"{season}: {len(df)} players, cols={[c for c in df.columns if c.startswith(('proj', 'act'))][:3]}...")
    sched, teams = fetch_schedule(2027)
    sched.to_csv(DATA / "schedule_2027.csv", index=False)
    teams.to_csv(DATA / "teams.csv", index=False)
    print(f"schedule: {len(sched)} team-games, {sched.date.min()} .. {sched.date.max()}")


if __name__ == "__main__":
    main(tuple(int(s) for s in sys.argv[1:]) or (2027, 2026, 2025))


def fetch_gamelogs(season: int, limit: int = 600) -> pd.DataFrame:
    """Per-game box scores (statSplitTypeId 5) for the top `limit` players of `season`."""
    flt = {
        "players": {
            "limit": limit,
            "filterStatsForSourceIds": {"value": [0]},
            "filterStatsForSplitTypeIds": {"value": [5]},
            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
        }
    }
    r = requests.get(
        BASE.format(season=season) + "/segments/0/leaguedefaults/3",
        params={"view": "kona_player_info"},
        headers={"X-Fantasy-Filter": json.dumps(flt)},
        timeout=180,
    )
    r.raise_for_status()
    rows = []
    for p in r.json()["players"]:
        pl = p["player"]
        for s in pl.get("stats", []):
            st = s.get("stats") or {}
            # the endpoint also serves last season's logs until games are played: keep the requested season only
            if s.get("statSplitTypeId") != 5 or not st or not st.get("40") or s.get("seasonId", season) != season:
                continue
            row = {"id": pl["id"], "name": pl["fullName"], "scoring_period": s["scoringPeriodId"]}
            row.update({name: st.get(k, 0.0) for k, name in STAT_IDS.items() if k != "42"})
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_birthdates(ids) -> pd.DataFrame:
    """Date of birth per ESPN athlete id (core API), fetched in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    def one(pid):
        try:
            r = requests.get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{pid}", timeout=30)
            return {"id": pid, "dob": r.json().get("dateOfBirth")}
        except Exception:
            return {"id": pid, "dob": None}

    with ThreadPoolExecutor(16) as ex:
        return pd.DataFrame(list(ex.map(one, ids)))


DARKO_ASOF = {2021: "2020-12-15", 2022: "2021-10-12", 2023: "2022-10-11", 2024: "2023-10-17",
              2025: "2024-10-15", 2026: "2025-10-14"}  # 7 days before each season's first game


def _devalue(data, i, depth=0):
    """Resolve SvelteKit's devalue-encoded payload (flat array of values referencing each other by index)."""
    v = data[i]
    if depth > 10:
        return v
    if isinstance(v, dict):
        return {k: _devalue(data, j, depth + 1) for k, j in v.items()}
    if isinstance(v, list):
        if v and isinstance(v[0], str) and v[0] in ("Date", "Set", "Map"):
            return v
        return [_devalue(data, j, depth + 1) for j in v]
    return v


def fetch_darko(asof: str | None = None) -> tuple[pd.DataFrame, dict]:
    """DARKO leaderboard snapshot as of a date (per-100 points/assists, pace, shooting projections, DPM)."""
    url = "https://www.darko.app/__data.json" + (f"?asof={asof}" if asof else "")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    r.raise_for_status()
    node = [n for n in r.json()["nodes"] if n and n.get("type") == "data"][-1]
    obj = _devalue(node["data"], 0)
    pl = obj["players"]
    return pd.DataFrame(dict(zip(pl["keys"], pl["values"]))), obj["asOf"]


def fetch_darko_fantasy_lab() -> str:
    """Raw payload of DARKO's current full per-game projections (archived for in-season validation)."""
    r = requests.get("https://www.darko.app/projections/__data.json", headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    r.raise_for_status()
    return r.text
