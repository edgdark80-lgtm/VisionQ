"""Capturas de evidencia de la UI (Playwright). Uso: python scripts/capture_ui.py [BASE_URL]"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).resolve().parents[1] / "docs/evidence"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1366, "height": 900})
    pg.goto(f"{BASE}/ui/"); pg.wait_for_timeout(1500)
    pg.screenshot(path=OUT / "ui_01_inicio.png", full_page=True)
    for i, name in [(1, "ui_02_inspeccion_inclusion.png"), (5, "ui_03_inspeccion_rayaduras.png")]:
        pg.locator("#sample-row button").nth(i).click(); pg.wait_for_timeout(1500)
        pg.screenshot(path=OUT / name, full_page=True)
    pg.click("#tab-failure"); pg.click("#preset-risk"); pg.click("button[type=submit]"); pg.wait_for_timeout(1200)
    pg.screenshot(path=OUT / "ui_04_falla_riesgo.png", full_page=True)
    pg.fill("input[name=rotational_speed_rpm]", "-50"); pg.click("button[type=submit]"); pg.wait_for_timeout(1000)
    pg.screenshot(path=OUT / "ui_05_falla_validacion.png", full_page=True)
    pg.click("#tab-system"); pg.wait_for_timeout(1200)
    pg.screenshot(path=OUT / "ui_06_sistema.png", full_page=True)
    pg.goto(f"{BASE}/docs"); pg.wait_for_timeout(2500)
    pg.screenshot(path=OUT / "ui_07_swagger.png", full_page=False)
    m = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    m.goto(f"{BASE}/ui/"); m.wait_for_timeout(1200)
    m.locator("#sample-row button").nth(0).click(); m.wait_for_timeout(1500)
    m.screenshot(path=OUT / "ui_08_movil.png", full_page=True)
    b.close()
print("capturas OK")
