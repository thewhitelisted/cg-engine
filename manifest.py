"""Record (or verify) SHA-256 checksums of every input data file, so results are tied to exact data.

    python3 manifest.py          # write data_manifest.json
    python3 manifest.py --check  # verify current files against it
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
INPUTS = sorted(p for p in (ROOT / "data").glob("*.csv")
                if p.name.startswith(("players_", "gamelogs_", "schedule_", "teams", "birthdates", "team_records")))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    out = ROOT / "data_manifest.json"
    if "--check" in sys.argv:
        ref = json.loads(out.read_text())
        bad = [k for k, v in ref.items() if not (ROOT / k).exists() or digest(ROOT / k) != v]
        print("OK" if not bad else f"CHANGED: {bad}")
    else:
        out.write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in INPUTS}, indent=1))
        print(f"wrote {out.name} ({len(INPUTS)} files)")
