"""
Export Slide Presentation to PDF using Playwright (headless Chrome).
This gives pixel-perfect quality — the same approach used by Google Slides, Canva, Notion.

Usage:
    pip install playwright
    playwright install chromium
    python export_slides_pdf.py

Output: Sentiment_Analysis_Presentation.pdf
"""

import asyncio
from playwright.async_api import async_playwright

SLIDE_URL = "http://localhost:4200/present"
OUTPUT_FILE = "Sentiment_Analysis_Presentation.pdf"
SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720


async def export_pdf():
    print("🚀 Starting headless Chrome...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": SLIDE_WIDTH, "height": SLIDE_HEIGHT})

        print(f"📄 Loading {SLIDE_URL} ...")
        await page.goto(SLIDE_URL, wait_until="networkidle")

        # Wait for fonts and images to fully load
        await page.wait_for_timeout(3000)

        print("🖨️  Generating PDF...")
        await page.pdf(
            path=OUTPUT_FILE,
            width=f"{SLIDE_WIDTH}px",
            height=f"{SLIDE_HEIGHT}px",
            print_background=True,       # CRITICAL: preserve dark backgrounds & gradients
            landscape=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=True,    # Use @page size from CSS
        )

        await browser.close()

    print(f"✅ PDF saved: {OUTPUT_FILE}")
    print(f"📁 File location: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(export_pdf())
