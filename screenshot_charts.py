#!/usr/bin/env python3
"""Generate PNG screenshots of all example charts for README."""

from playwright.sync_api import sync_playwright
import os

# All charts from the README
CHARTS = [
    # NVDA
    ("nvda_2023-12-20_WIN_TP3", "src/samples/nvda_daily_1214_2105/charts/04_2023-12-20_WIN_TP3.html"),
    ("nvda_2024-03-08_WIN_TP1", "src/samples/nvda_daily_1214_2105/charts/35_2024-03-08_WIN_TP1.html"),
    ("nvda_2024-04-11_LOSS_SL", "src/samples/nvda_daily_1214_2105/charts/44_2024-04-11_LOSS_SL.html"),
    # GOOGL
    ("googl_2023-12-15_WIN_TP3", "src/samples/googl_daily_1214_2115/charts/01_2023-12-15_WIN_TP3.html"),
    ("googl_2024-01-02_WIN_TP3", "src/samples/googl_daily_1214_2115/charts/03_2024-01-02_WIN_TP3.html"),
    # AAPL
    ("aapl_2025-01-02_WIN_TP1", "src/samples/aapl_daily_1214_2123/charts/100_2025-01-02_WIN_TP1.html"),
    ("aapl_2023-12-16_LOSS_SL", "src/samples/aapl_daily_1214_2123/charts/01_2023-12-16_LOSS_SL.html"),
]

OUTPUT_DIR = "docs/screenshots"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        
        for name, path in CHARTS:
            full_path = f"file://{os.path.abspath(path)}"
            print(f"📸 Capturing {name}...")
            
            page.goto(full_path)
            page.wait_for_timeout(4000)  # Wait for TradingView chart to render
            
            output_path = f"{OUTPUT_DIR}/{name}.png"
            page.screenshot(path=output_path)
            print(f"   ✅ Saved to {output_path}")
        
        browser.close()
    
    print(f"\n🎉 Done! {len(CHARTS)} screenshots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
