"""The hosted page lists every source file it loads; keep that list in step with the packages."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_web_page_lists_every_source_file():
    html = (ROOT / "web" / "index.html").read_text()
    block = re.search(r"const sources = \[(.*?)\];", html, re.S).group(1)
    listed = set(re.findall(r'"([^"]+\.py)"', block))
    expected = {"app.py"} | {
        path.relative_to(ROOT).as_posix()
        for package in ("moonsim", "moonsim_app")
        for path in (ROOT / package).rglob("*.py")
    }
    assert listed == expected
