"""Mở app như một người dùng thật để Streamlit Community Cloud không cho app "ngủ" sau 12 giờ không có truy cập.
Chạy bởi .github/workflows/keep_alive.yml (cần secret APP_URL = địa chỉ app, ví dụ https://ten-app.streamlit.app)."""
import os
import re
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "").strip()
if not URL:
    sys.exit("Thiếu secret APP_URL")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(8_000)
    wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
    if wake.count():
        wake.first.click()
        print("App đang ngủ: đã bấm nút đánh thức")
    app = page.frame_locator('iframe[title="streamlitApp"]').locator('[data-testid="stApp"]')
    try:
        app.wait_for(timeout=120_000)
    except PWTimeout:
        try:
            page.locator('[data-testid="stApp"]').wait_for(timeout=20_000)  # app không nằm trong iframe
        except PWTimeout:
            browser.close()
            sys.exit("Không thấy giao diện app sau 2 phút")
    page.wait_for_timeout(5_000)
    print("App đang chạy bình thường")
    browser.close()
