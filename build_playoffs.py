"""Build playoff_picture.html (the Playoff Picture artifact) from web/playoffs_template.html + data/playoffs.json."""
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    data = (ROOT / "data" / "playoffs.json").read_text().replace("</", "<\\/")
    html = (ROOT / "web" / "playoffs_template.html").read_text().replace("__DATA__", data)
    (ROOT / "playoff_picture.html").write_text(html)
    print("wrote playoff_picture.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
