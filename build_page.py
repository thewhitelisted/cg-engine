"""Inject data/cheatsheet.json into web/template.html -> cheatsheet.html."""
from pathlib import Path

root = Path(__file__).parent
html = (root / "web/template.html").read_text().replace("__DATA__", (root / "data/cheatsheet.json").read_text())
(root / "cheatsheet.html").write_text(html)
print("wrote cheatsheet.html", len(html) // 1024, "KB")
