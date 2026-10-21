"""Build the League Lab artifacts from web/lab_template.html:
    python3 build_lab.py          # league_lab.html  (our model)   from data/dashboard.json
    python3 build_lab.py --espn   # espn_lab.html    (ESPN-only)   from data/dashboard_espn.json"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VARIANTS = {"model": ("dashboard.json", "league_lab.html", "Cool Guys League Lab", "League Lab"),
            "espn": ("dashboard_espn.json", "espn_lab.html", "Cool Guys ESPN Lab", "ESPN Lab")}


def main(variant="model"):
    src, out, title, heading = VARIANTS[variant]
    data = (ROOT / "data" / src).read_text().replace("</", "<\\/")
    html = (ROOT / "web" / "lab_template.html").read_text()
    html = html.replace("<title>Cool Guys League Lab</title>", f"<title>{title}</title>")
    html = html.replace("<h1>League Lab</h1>", f"<h1>{heading}</h1>").replace("__DATA__", data)
    (ROOT / out).write_text(html)
    print("wrote", out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main("espn" if "--espn" in sys.argv else "model")
