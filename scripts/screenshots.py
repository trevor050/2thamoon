"""Capture the README screenshots from a running app.

streamlit run app.py --server.headless true --server.port 8599 &
pip install playwright && playwright install chromium
python scripts/screenshots.py http://localhost:8599
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def open_scenario(page: Page, base: str, key: str) -> None:
    page.goto(f"{base}/?scenario={key}")
    page.wait_for_selector("text=What happened", timeout=60_000)
    for _ in range(80):
        if not page.locator('[data-testid="stStatusWidget"]').count():
            break
        time.sleep(0.25)
    time.sleep(1.5)  # let plotly finish drawing


def scroll_to(page: Page, selector: str, offset: int = 16) -> None:
    page.evaluate(
        """([sel, off]) => {
            const el = document.querySelector(sel);
            const box = el.getBoundingClientRect();
            const main = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
            main.scrollBy(0, box.top - off);
        }""",
        [selector, offset],
    )
    time.sleep(0.8)


def main(base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        page.set_viewport_size({"width": 1440, "height": 1320})
        open_scenario(page, base, "pump_and_dump")
        scroll_to(page, "h1", offset=64)
        page.screenshot(path=OUT / "pump_and_dump.png")
        page.set_viewport_size({"width": 1440, "height": 900})

        open_scenario(page, base, "death_spiral")
        scroll_to(page, "h3")
        page.screenshot(path=OUT / "death_spiral.png")

        page.get_by_role("tab", name="Peg mechanics").click()
        time.sleep(1.5)
        scroll_to(page, '[role="tablist"]')
        page.screenshot(path=OUT / "death_spiral_mechanics.png")

        open_scenario(page, base, "whale_exit")
        page.get_by_role("tab", name="Liquidity").click()
        time.sleep(1.5)
        scroll_to(page, '[role="tablist"]')
        page.screenshot(path=OUT / "whale_exit_liquidity.png")

        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        open_scenario(phone, base, "peg_defense")
        scroll_to(phone, ".st-key-stats", offset=72)
        phone.screenshot(path=OUT / "phone.png")
        browser.close()
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599")
