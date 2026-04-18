import sys
from playwright.sync_api import sync_playwright
import time

def export_pdf(url, output_path):
    print(f"Starting PDF export for: {url}")
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Set viewport to match slide size
        page.set_viewport_size({"width": 1280, "height": 720})
        
        # Navigate to slides
        page.goto(url, wait_until="networkidle")
        
        # Thêm một chút delay để animation (nếu có) hoàn tất
        time.sleep(2)
        
        # Xuất PDF khớp chính xác với kích thước slide (1280x720)
        page.pdf(
            path=output_path,
            width="1280px",
            height="720px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )
        
        browser.close()
    print(f"Successfully exported PDF to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_pdf_worker.py <url> <output_path>")
        sys.exit(1)
    
    export_url = sys.argv[1]
    output_file = sys.argv[2]
    export_pdf(export_url, output_file)
