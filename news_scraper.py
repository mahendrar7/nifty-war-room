"""
news_scraper.py — Scrapes economic calendar + market news, writes data/market_feed.json

Run standalone:   python news_scraper.py          (scrapes once)
                  python news_scraper.py --loop    (scrapes every 5 min)

Or call from war room:
    from news_scraper import scrape_and_save
    threading.Thread(target=scrape_and_save, args=(True,), daemon=True).start()
"""

import json
import os
import time
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT   = os.path.join(DATA_DIR, "market_feed.json")
IST      = timezone(timedelta(hours=5, minutes=30))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ── Economic Calendar (ForexFactory JSON — free, no auth) ────────

def fetch_economic_calendar():
    """Fetch this week's economic events, filter for today + relevant countries."""
    events = []
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        today = datetime.now(IST).strftime("%Y-%m-%d")

        for e in data:
            e_date = (e.get("date") or "")[:10]
            country = (e.get("country") or "").upper()
            if e_date == today and country in ("INR", "USD", "EUR", "CNY", "JPY", "GBP"):
                # Parse time to IST
                raw_time = e.get("date", "")
                try:
                    utc_dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    ist_dt = utc_dt.astimezone(IST)
                    time_str = ist_dt.strftime("%H:%M")
                except Exception:
                    time_str = "—"

                events.append({
                    "time":    time_str,
                    "country": country,
                    "event":   e.get("title", ""),
                    "impact":  (e.get("impact") or "low").upper(),
                    "actual":  e.get("actual", ""),
                    "forecast": e.get("forecast", ""),
                    "previous": e.get("previous", ""),
                })
    except Exception as ex:
        print(f"  [news] Calendar fetch failed: {ex}")

    return events


# ── Market News (RSS feeds — no CORS issue from Python) ──────────

BREAKING_KEYWORDS = [
    "breaking", "flash", "just in", "urgent", "rbi", "rate cut", "rate hike",
    "repo rate", "fed ", "fomc", "powell", "crash", "circuit", "halt",
    "war ", "attack", "bomb", "missile", "sanction", "tariff",
    "nifty crash", "sensex crash", "market crash", "black swan",
    "emergency", "lockdown", "default", "bankruptcy",
]


def _is_breaking(title):
    """Check if a headline looks like breaking/high-impact news."""
    lower = title.lower()
    return any(kw in lower for kw in BREAKING_KEYWORDS)


RSS_FEEDS = [
    ("ET Markets",     "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Moneycontrol",   "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("Livemint",       "https://www.livemint.com/rss/markets"),
]


def _parse_rss(source_name, url):
    """Parse an RSS feed and return list of news items."""
    items = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        # Handle both RSS 2.0 and Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        rss_items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        for item in rss_items[:8]:
            title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
            pub_date = (item.findtext("pubDate") or item.findtext("atom:updated", namespaces=ns) or "").strip()
            link = (item.findtext("link") or "").strip()
            if not link:
                link_el = item.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""

            # Parse time
            time_str = ""
            if pub_date:
                try:
                    # RSS date: "Tue, 25 Mar 2026 10:30:00 +0530"
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date).astimezone(IST)
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    time_str = ""

            if title:
                items.append({
                    "time":     time_str,
                    "title":    title,
                    "source":   source_name,
                    "link":     link,
                    "breaking": _is_breaking(title),
                })
    except Exception as ex:
        print(f"  [news] RSS failed ({source_name}): {ex}")

    return items


def fetch_news():
    """Fetch news from all RSS sources, deduplicate, sort by time."""
    all_items = []
    for name, url in RSS_FEEDS:
        all_items.extend(_parse_rss(name, url))

    # Deduplicate by title similarity (exact match)
    seen = set()
    unique = []
    for item in all_items:
        key = item["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort: items with time first (descending), then no-time items
    with_time = [i for i in unique if i["time"]]
    no_time   = [i for i in unique if not i["time"]]
    with_time.sort(key=lambda x: x["time"], reverse=True)

    return (with_time + no_time)[:20]


# ── Main ─────────────────────────────────────────────────────────

def scrape_and_save(loop=False):
    """Scrape everything and write to data/market_feed.json."""
    while True:
        try:
            cal  = fetch_economic_calendar()
            news = fetch_news()

            feed = {
                "updated": datetime.now(IST).strftime("%H:%M:%S"),
                "date":    datetime.now(IST).strftime("%Y-%m-%d"),
                "calendar": cal,
                "news":     news,
            }

            os.makedirs(DATA_DIR, exist_ok=True)
            with open(OUTPUT, "w") as f:
                json.dump(feed, f, indent=2, ensure_ascii=False)

            print(f"  [news] Saved {len(cal)} calendar events, {len(news)} headlines → {OUTPUT}")

        except Exception as ex:
            print(f"  [news] Scrape error: {ex}")

        if not loop:
            break
        time.sleep(300)  # 5 minutes


if __name__ == "__main__":
    loop = "--loop" in sys.argv
    print(f"  [news] Starting scraper {'(loop mode)' if loop else '(one-shot)'}...")
    scrape_and_save(loop=loop)
