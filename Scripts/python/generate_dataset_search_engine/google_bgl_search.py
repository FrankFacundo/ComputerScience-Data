#!/usr/bin/env python3
"""
google_bgl_search.py
Search Google for queries like "site:bgl.lu <keyword>" read from a file,
and save the results to CSV.

⚠️ Note: Automated scraping of Google may violate Google's Terms of Service and
can trigger CAPTCHAs or blocking. Prefer the official Programmable Search Engine
(Custom Search JSON API) when possible.

Usage:
  python google_bgl_search.py -i keywords.txt -o results.csv --pages 2 --site bgl.lu --hl fr

Requirements:
  pip install requests beautifulsoup4
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


def read_keywords(path: Path) -> List[str]:
    keywords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keywords.append(line)
    return keywords


def google_search(
    query: str, hl: str, num: int, start: int, session: requests.Session
) -> str:
    """
    Perform a single Google search request and return the HTML text.
    """
    params = {
        "q": query,
        "hl": hl,
        "num": str(num),
        "start": str(start),
        "source": "hp",
        "sca_esv": "dummy",
    }
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept-Language": f"{hl},en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }
    resp = session.get(
        "https://www.google.com/search", params=params, headers=headers, timeout=20
    )
    # Handle soft-blocks
    if resp.status_code == 429:
        raise RuntimeError("HTTP 429 Too Many Requests from Google")
    if (
        "Our systems have detected unusual traffic" in resp.text
        or "/sorry/index" in resp.url
    ):
        raise RuntimeError("Blocked by Google (unusual traffic/CAPTCHA)")
    return resp.text


def extract_results(html: str, site: str) -> List[Dict[str, str]]:
    """
    Parse the HTML and extract organic result links, titles, and snippets.
    Filters only URLs from the given site (including subdomains).
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    # Google organic results often use /url?q=<dest> links
    for a in soup.select('a[href^="/url?q="]'):
        href = a.get("href", "")
        # Extract destination URL from "/url?q=...&..."
        try:
            dest = parse_qs(urlparse(href).query).get("q", [None])[0]
        except Exception:
            dest = None
        if not dest:
            continue

        netloc = urlparse(dest).netloc.lower()
        site_l = site.lower().lstrip(".")
        if not (netloc == site_l or netloc.endswith("." + site_l)):
            # Even with site: operator, double-check domain
            continue

        title_tag = a.find("h3")
        title = (
            title_tag.get_text(strip=True)
            if title_tag
            else a.get_text(strip=True) or "(no title)"
        )

        # Snippet is near the result; try a few common containers
        snippet = ""
        # Move up to a sensible container, then look for known snippet classes
        container = a
        for _ in range(3):
            if container and container.parent:
                container = container.parent
        if container:
            # Common snippet class
            snip_div = container.find(
                "div", attrs={"class": lambda c: c and "VwiC3b" in c}
            )
            if snip_div:
                snippet = snip_div.get_text(" ", strip=True)

        results.append({"title": title, "url": dest, "snippet": snippet})

    # Deduplicate by URL while preserving order
    seen = set()
    unique = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        unique.append(r)
    return unique


def run_search_for_keyword(
    keyword: str,
    site: str,
    pages: int,
    hl: str,
    per_page: int,
    delay_range: tuple,
    session: requests.Session,
) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for page_idx in range(pages):
        start = page_idx * per_page
        query = f"site:{site} {keyword}".strip()
        html = google_search(
            query=query, hl=hl, num=per_page, start=start, session=session
        )
        parsed = extract_results(html, site=site)
        fetched_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for rank, r in enumerate(parsed, start=1 + start):
            all_rows.append(
                {
                    "keyword": keyword,
                    "query": query,
                    "position": rank,
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["snippet"],
                    "fetched_at_utc": fetched_at,
                }
            )
        # Polite delay
        if page_idx < pages - 1:
            sleep_s = random.uniform(*delay_range)
            time.sleep(sleep_s)
    return all_rows


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query Google for site-restricted results and save to CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="keywords.txt",
        type=Path,
        help="Path to keywords file (one per line).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results.csv",
        type=Path,
        help="Where to write the CSV results.",
    )
    parser.add_argument(
        "--site",
        default="bgl.lu",
        help="Site to restrict with Google 'site:' operator (default: bgl.lu).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
        help="Number of pages to fetch (10 results per page).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=10,
        help="Results per page (Google supports up to ~10).",
    )
    parser.add_argument(
        "--hl", default="fr", help="Interface language for Google (e.g., fr, en, de)."
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=2.0,
        help="Minimum delay between requests in seconds.",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=5.0,
        help="Maximum delay between requests in seconds.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    if args.pages < 1:
        print("--pages must be >= 1", file=sys.stderr)
        return 2

    if args.per_page < 1 or args.per_page > 10:
        print("--per-page must be between 1 and 10 for Google", file=sys.stderr)
        return 2

    delay_range = (
        min(args.min_delay, args.max_delay),
        max(args.min_delay, args.max_delay),
    )

    keywords = read_keywords(args.input)
    if not keywords:
        print("No keywords found in input file.", file=sys.stderr)
        return 2

    session = requests.Session()
    rows: List[Dict[str, str]] = []
    try:
        for kw in keywords:
            # Random delay between keywords to be polite
            time.sleep(random.uniform(*delay_range))
            rows.extend(
                run_search_for_keyword(
                    keyword=kw,
                    site=args.site,
                    pages=args.pages,
                    hl=args.hl,
                    per_page=args.per_page,
                    delay_range=delay_range,
                    session=session,
                )
            )
    except RuntimeError as e:
        print(f"Stopped due to error: {e}", file=sys.stderr)

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "keyword",
                "query",
                "position",
                "title",
                "url",
                "snippet",
                "fetched_at_utc",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
