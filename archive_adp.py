"""Clean preseason ESPN ADP from archived FantasyPros consensus pages (Wayback Machine).

ESPN's API stores ADP that was updated in-season for 2024-25 / 2025-26. FantasyPros' preseason snapshot
carries an ESPN ADP column; we match names to ESPN ids and write data/adp_archive_{t}.csv.
Snapshots (data/archive/fp_adp_{t}.html): 2022 (data 10/18/2021), 2024 (10/14/2023), 2026 (9/26/2025).
2025 is excluded: the only snapshot (Aug 2024) still shows 2023-24 data (last update 10/22/2023).
"""
import re
import unicodedata

import pandas as pd

from pathlib import Path

DATA = Path(__file__).parent / "data"
SNAPSHOTS = {2022: "fp_adp_2022.html", 2024: "fp_adp_2024.html", 2026: "fp_adp_2026.html"}


def key(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


def parse(t: int) -> pd.DataFrame:
    tab = pd.read_html(DATA / "archive" / SNAPSHOTS[t])[0]
    tab["name"] = tab["Player"].str.replace(r"\s*\(.*$", "", regex=True).str.strip()
    tab = tab[pd.to_numeric(tab["ESPN"], errors="coerce").notna()].copy()
    tab["adp_espn"] = tab["ESPN"].astype(float)
    return tab[["name", "adp_espn"]]


def build(t: int) -> pd.DataFrame:
    tab = parse(t)
    src = t + 1 if t == 2026 else t
    players = pd.read_csv(DATA / f"players_{src}.csv")[["id", "name"]]
    players["k"] = players["name"].map(key)
    tab["k"] = tab["name"].map(key)
    m = tab.merge(players.drop_duplicates("k"), on="k", how="left", suffixes=("_fp", ""))
    unmatched = m[m["id"].isna()]["name_fp"].tolist()
    out = m.dropna(subset=["id"])[["id", "name", "adp_espn"]]
    out["id"] = out["id"].astype(int)
    out.to_csv(DATA / f"adp_archive_{t}.csv", index=False)
    return out, unmatched


if __name__ == "__main__":
    from scipy.stats import spearmanr
    for t in SNAPSHOTS:
        out, um = build(t)
        print(f"{t}: matched {len(out)}, unmatched {len(um)} {um[:8]}")
        if t in (2022, 2024):  # validate against ESPN API ADP (clean for these seasons)
            api = pd.read_csv(DATA / f"players_{t}.csv")[["id", "adp"]]
            v = out.merge(api, on="id")
            v = v[v.adp < 135]
            r = spearmanr(v.adp_espn, v.adp).correlation
            print(f"   vs ESPN API ADP (n={len(v)}): rank corr {r:.3f}, median |diff| {(v.adp_espn - v.adp).abs().median():.1f} picks")
