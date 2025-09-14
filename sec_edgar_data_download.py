import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

HEADERS = {
    "User-Agent": "Chowdhury Sabir Morshed (mamonchw@gmail.com) - Financial QnA Project",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}

OUTDIR = Path("data/raw")
OUTDIR.mkdir(parents=True, exist_ok=True)


def download_main_filing(cik, accession_number, ticker, year):
    """Download main 10-K HTML, extract text, and save"""
    acc_no_nodash = accession_number.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_nodash}"
    index_url = f"{base_url}/{accession_number}-index.html"

    print(f"🔎 Fetching filing index for {ticker} {year} ...")
    resp = requests.get(index_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all document rows
    candidates = []
    for row in soup.select("table tr"):
        cols = row.find_all("td")
        if len(cols) >= 3:
            file_link = cols[2].find("a")["href"] if cols[2].find("a") else None
            if file_link and file_link.endswith(".htm"):
                size_text = cols[-1].text.strip().replace(",", "")
                try:
                    size = int(size_text.split()[0])
                except:
                    size = 0
                candidates.append((file_link, size))

    # Pick the largest HTML file (most likely the full 10-K)
    if not candidates:
        raise Exception(f"No HTML candidates found for {ticker} {year}")

    candidates.sort(key=lambda x: x[1], reverse=True)
    link = candidates[0][0]  # largest file
    filing_url = "https://www.sec.gov" + link

    print(f"⬇️ Downloading main filing: {filing_url}")
    resp = requests.get(filing_url, headers=HEADERS)
    resp.raise_for_status()
    filing_soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style
    for tag in filing_soup(["script", "style", "noscript"]):
        tag.extract()

    text = filing_soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    outfile = OUTDIR / f"{ticker}_{year}.txt"
    outfile.write_text(text, encoding="utf-8")
    print(f"✅ Saved {outfile} ({len(text)} chars)")


# Example: Microsoft 2024 (CIK 789019, accession 0000950170-24-087843)
download_main_filing("789019", "0000950170-24-087843", "MSFT", "2024")
