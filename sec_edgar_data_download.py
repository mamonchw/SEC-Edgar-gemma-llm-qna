import os
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# --- SEC requires polite headers ---
HEADERS = {
    "User-Agent": "Chowdhury Sabir Morshed (mamonchw@gmail.com) - Financial QnA Project",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

# --- Filings we want (HTML versions for easier parsing) ---
FILINGS = {
    "GOOGL": [
        ("2024", "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm"),
        ("2023", "https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm"),
        ("2022", "https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm"),
    ],
    "MSFT": [
        ("2024", "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"),
        ("2023", "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm"),
        ("2022", "https://www.sec.gov/Archives/edgar/data/789019/000156459022026876/msft-10k_20220630.htm"),
    ],
    "NVDA": [
        ("2024", "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm"),
        ("2023", "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm"),
        ("2022", "https://www.sec.gov/Archives/edgar/data/1045810/000104581022000036/nvda-20220130.htm"),
    ],
}

# --- Output directories ---
HTML_DIR = Path("data/raw_html")
TEXT_DIR = Path("data/raw")

HTML_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def download_filings():
    for ticker, docs in FILINGS.items():
        for year, url in docs:
            html_file = HTML_DIR / f"{ticker}_{year}.htm"
            text_file = TEXT_DIR / f"{ticker}_{year}.txt"

            # Skip if already converted
            if text_file.exists():
                print(f"✅ Already converted: {text_file}")
                continue

            # Download HTML if missing
            if not html_file.exists():
                print(f"⬇️ Downloading {ticker} {year} ...")
                try:
                    r = requests.get(url, headers=HEADERS, timeout=30)
                    r.raise_for_status()
                    html_file.write_text(r.text, encoding="utf-8")
                    print(f"   Saved HTML to {html_file}")
                    time.sleep(0.5)  # polite delay
                except Exception as e:
                    print(f"   ❌ Failed {ticker} {year}: {e}")
                    continue

            # --- Convert HTML → Text ---
            try:
                html_content = html_file.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html_content, "html.parser")

                # Remove scripts/styles
                for tag in soup(["script", "style", "head", "meta", "noscript"]):
                    tag.decompose()

                text = soup.get_text(separator="\n")
                text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

                text_file.write_text(text, encoding="utf-8")
                print(f"   ✅ Converted to text: {text_file}")
            except Exception as e:
                print(f"   ❌ Conversion failed for {ticker} {year}: {e}")

download_filings()
