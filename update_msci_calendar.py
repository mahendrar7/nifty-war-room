#!/usr/bin/env python3
"""
update_msci_calendar.py — Fetch MSCI quarterly rebalancing details and update econ_calendar.py

Run on or after the MSCI announcement day (~2nd week of Feb/May/Aug/Nov):

    # Preferred: give it the article URL directly (univest.in has the best structured data)
    python update_msci_calendar.py "Aug 2026" https://univest.in/blogs/msci-rebalancing-aug-2026-...

    # No URL: script fetches the MSCI official PDF (global summary) then asks for an India
    #          stock article URL or tries to auto-discover one
    python update_msci_calendar.py "Aug 2026"

    # No quarter arg: script fetches MSCI quarterly-index-review page and reports
    #                 what the current quarter is (useful for "what just got announced?")
    python update_msci_calendar.py

The script:
  1. Scrapes https://www.msci.com/indexes/quarterly-index-review for the latest PDF
  2. Downloads the PDF and extracts the announcement date / quarter label
  3. Fetches India-specific stock details from direct URL (preferred) or auto-discovers
  4. Extracts India flow direction/amount, added/removed stocks with per-stock flows
  5. Validates stock symbols against Kite instruments
  6. Shows findings and asks for confirmation before writing
  7. Patches MSCI_EVENTS in econ_calendar.py for the matching quarter

Source note:
  The official MSCI PDF (step 1-2) only has global totals, NOT India stock breakdown.
  India-specific additions/removals come from broker research published on univest.in,
  businessupturn.com, etc.  univest.in/blogs/msci-rebalancing-{month}-{year} is the
  most reliable structured source — it has per-stock flows in a table.
"""

import re
import sys
import os
import time
import urllib.parse
import requests
from html.parser import HTMLParser

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

FLOW_EXCLUDE_WORDS = {
    "MSCI", "NSE", "BSE", "INDIA", "USD", "INR", "ETF", "FII", "DII",
    "IPO", "GDP", "RBI", "CPI", "SEBI", "AMFI", "NIFTY", "SENSEX",
    "GLOBAL", "INDEX", "LARGE", "MID", "SMALL", "CAP", "STANDARD",
    "YES", "ALL", "NEW", "OLD", "TOP", "OUT", "AND", "THE", "FOR",
    "AUM", "CMP", "INR", "LTP", "PE", "CE", "QIR", "SAIR",
}


# ── HTML → plain text ─────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag in ("p", "li", "td", "th", "h1", "h2", "h3", "br", "div", "tr"):
            self.parts.append(" | ")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return " ".join("".join(p.parts).split())


# ── MSCI official page scrape ────────────────────────────────────────────────

MSCI_REVIEW_PAGE = "https://www.msci.com/indexes/quarterly-index-review"
MSCI_PDF_BASE    = "https://www.msci.com"

def fetch_msci_official() -> dict | None:
    """
    Scrape the MSCI quarterly-index-review page.
    Returns dict with keys: pdf_url, quarter_label, announce_date_str, execution_date_str
    or None if the page is unreachable.
    """
    try:
        r = requests.get(MSCI_REVIEW_PAGE,
                         headers={**HEADERS, "Referer": "https://www.msci.com/indexes"},
                         timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [msci.com] Could not reach quarterly-index-review page: {e}")
        return None

    html = r.text

    # Find the PDF download link
    pdf_match = re.search(
        r'href=["\'](/downloads/[^"\']*Index[^"\']*\.pdf)["\']',
        html, re.IGNORECASE
    )
    if not pdf_match:
        print("  [msci.com] No PDF link found on quarterly-index-review page.")
        return None

    pdf_path = pdf_match.group(1)
    pdf_url  = MSCI_PDF_BASE + pdf_path

    # Extract quarter label from the filename, e.g. "MAy 2026" → "May 2026"
    fname_match = re.search(r'Factsheet_([A-Za-z]+\s+\d{4})', pdf_path)
    quarter_raw = fname_match.group(1).title() if fname_match else ""

    print(f"  [msci.com] Found PDF: {pdf_path.split('/')[-1]}")

    # Download and parse the PDF for dates
    info = {"pdf_url": pdf_url, "quarter_label": quarter_raw,
            "announce_date_str": "", "execution_date_str": ""}

    try:
        import io
        import pypdf
        pdf_r = requests.get(pdf_url,
                             headers={**HEADERS, "Referer": MSCI_REVIEW_PAGE},
                             timeout=20)
        pdf_r.raise_for_status()
        reader = pypdf.PdfReader(io.BytesIO(pdf_r.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # "On May 12, 2026, we communicated..."
        ann_m = re.search(r'On\s+(\w+ \d{1,2},\s*\d{4}),?\s+we communicated', text)
        if ann_m:
            info["announce_date_str"] = ann_m.group(1)

        # "implemented ... as of the close of May 29, 2026"
        exec_m = re.search(r'close of\s+(\w+ \d{1,2},\s*\d{4})', text)
        if exec_m:
            info["execution_date_str"] = exec_m.group(1)

        # global totals for display
        acwi_add = re.search(r'MSCI ACWI IMI\s*(\d+) Additions', text)
        acwi_del = re.search(r'MSCI ACWI IMI\s*\d+ Additions\s*(\d+) Deletions', text)
        if acwi_add:
            info["acwi_additions"] = acwi_add.group(1)
        if acwi_del:
            info["acwi_deletions"] = acwi_del.group(1)

    except ImportError:
        print("  [msci.com] pypdf not installed — skipping PDF parse (pip install pypdf)")
    except Exception as e:
        print(f"  [msci.com] PDF parse error: {e}")

    return info


# ── Article discovery ────────────────────────────────────────────────────────

def _find_article_urls(quarter: str, max_results: int = 5) -> list[str]:
    """
    Try to auto-discover article URLs for the MSCI announcement.
    Uses businessupturn.com and zeebiz.com site search (return real hrefs).
    Returns whatever it finds — caller falls back to asking the user.
    """
    month, year = quarter.split()[0], quarter.split()[-1]
    found_urls: list[str] = []
    seen: set[str] = set()

    search_targets = [
        ("businessupturn", f"https://www.businessupturn.com/?s=MSCI+{month}+{year}",
         r'https?://www\.businessupturn\.com/[^"\'>\s]+'),
        ("zeebiz",         f"https://www.zeebiz.com/?s=MSCI+{month}+{year}",
         r'https?://www\.zeebiz\.com/[^"\'>\s]+'),
        ("univest",        f"https://univest.in/blogs/category/msci?s={month}+{year}",
         r'https?://univest\.in/blogs/[^"\'>\s]+'),
    ]

    for name, search_url, link_pattern in search_targets:
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            links = re.findall(link_pattern, resp.text)
            for link in links:
                link = link.rstrip('"\'>')
                low  = link.lower()
                if ("msci" in low or "rebalanc" in low) and year in link:
                    if link not in seen:
                        seen.add(link)
                        found_urls.append(link)
            if found_urls:
                print(f"  Found via {name}: {found_urls[-1][:90]}")
        except Exception:
            pass

        if len(found_urls) >= max_results:
            break

    return list(dict.fromkeys(found_urls))[:max_results]


def _fetch_text(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        return _html_to_text(resp.text)
    except Exception as e:
        print(f"  [fetch] {url[:70]}… failed: {e}")
        return ""


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_flow(text: str) -> tuple[str | None, str | None]:
    """Return (direction, amount_str) e.g. ('OUTFLOW', '800M–1B')."""
    low = text.lower()

    # Find net India flow direction by proximity of outflow/inflow to "india"
    direction = None
    india_idx = low.find("india")
    search_area = text[max(0, india_idx - 300):india_idx + 600] if india_idx >= 0 else text[:1000]
    low_area = search_area.lower()

    out_idx = low_area.find("outflow")
    in_idx  = low_area.find("inflow")
    if out_idx >= 0 and in_idx >= 0:
        # Both present — check which is for India net (look for "net" nearby)
        net_idx = low_area.find("net")
        if net_idx >= 0:
            direction = "OUTFLOW" if abs(net_idx - out_idx) < abs(net_idx - in_idx) else "INFLOW"
        else:
            direction = "OUTFLOW" if out_idx < in_idx else "INFLOW"
    elif out_idx >= 0:
        direction = "OUTFLOW"
    elif in_idx >= 0:
        direction = "INFLOW"

    if direction is None:
        # Fallback: "passive outflows" anywhere
        if "passive outflow" in low:
            direction = "OUTFLOW"
        elif "passive inflow" in low:
            direction = "INFLOW"

    # Amount — find total flow figures
    amount = None
    patterns = [
        # "$800 million to $1 billion" / "$800M to $1B"
        r'\$\s*(\d+(?:\.\d+)?)\s*(?:million|mn|M)\s*(?:to|-|–|to)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|B)',
        r'\$\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|B)\s*(?:to|-|–)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|B)',
        r'\$\s*(\d+(?:\.\d+)?)\s*(?:million|mn|M)\s*(?:to|-|–)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:million|mn|M)',
        # "exceed $1.6 billion" / "$1.6 billion"
        r'(?:exceed|over|around|approximately|~)\s*\$\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|B)',
        r'\$\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|B)',
        r'\$\s*(\d+(?:\.\d+)?)\s*(?:million|mn|M)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            g = m.groups()
            full = m.group(0).lower()
            if len(g) == 2:
                unit = "B" if "billion" in full else "M"
                amount = f"{g[0]}{unit}–{g[1]}{unit}"
            else:
                unit = "B" if "billion" in full else "M"
                amount = f"{g[0]}{unit}"
            break

    return direction, amount


def _normalise_flow(raw: str) -> str:
    """'  +$491 million ' → '$491M'"""
    s = re.sub(r'\s+', '', raw).lstrip('+')
    s = re.sub(r'(?i)million', 'M', s)
    s = re.sub(r'(?i)(billion|bn)', 'B', s)
    return s


def _extract_stock_table(text: str) -> tuple[list[dict], list[dict]]:
    """
    Two-pass extraction:
      Pass 1 — structured table rows (NSE symbols are ALL_CAPS, no IGNORECASE):
        "Federal Bank | FEDERALBNK | Inclusion | +$491 million"
      Pass 2 — inline text patterns:
        "RVNL Exclusion $135 million"

    NSE symbols must be 2-15 ALL_CAPS alphanumeric chars to distinguish from prose words.
    """
    removed, added = [], []

    # ── Pass 1: table rows with pipe-separated cells ──────────────────────────
    # Strict: NSE symbol must be ALL_CAPS (no re.IGNORECASE here)
    row_pat = re.compile(
        r'([A-Z][A-Z0-9]{1,14})'           # NSE symbol: strictly uppercase
        r'\s*\|[^|]{0,60}\|'               # skip one cell (or zero cells)
        r'\s*(Inclusion|Exclusion)'         # event type (case-insensitive via group)
        r'[^|]*\|[^|]*'
        r'([+\-~]?\s*\$\s*[\d,.]+\s*(?:million|bn|billion|mn|M|B))',
        re.IGNORECASE,  # only for Inclusion/Exclusion keyword, symbol already strictly uppercase
    )
    # Note: re.IGNORECASE here applies to all groups; fix by splitting into two patterns
    # Use a strict symbol-only pattern without IGNORECASE for symbol group:
    row_pat_strict = re.compile(
        r'\b([A-Z][A-Z0-9]{1,14})\b'       # strictly uppercase NSE symbol
        r'(?:\s*\|\s*[^|]{0,60})?\s*\|\s*'  # optional extra cell + pipe
        r'(Inclusion|Exclusion)'
        r'[^|]*\|\s*'
        r'([+\-~]?\s*\$\s*[\d,.]+\s*(?:million|billion|bn|M|B))',
    )
    for m in row_pat_strict.finditer(text):
        sym       = m.group(1)
        event_raw = m.group(2).lower()
        flow_raw  = m.group(3)
        if sym in FLOW_EXCLUDE_WORDS or len(sym) < 2:
            continue
        entry = {"symbol": sym, "flow_usd": _normalise_flow(flow_raw)}
        if "inclusion" in event_raw:
            if not any(e["symbol"] == sym for e in added):
                added.append(entry)
        else:
            if not any(e["symbol"] == sym for e in removed):
                removed.append(entry)

    if removed or added:
        return removed, added

    # ── Pass 2: inline patterns without pipe structure ────────────────────────
    inline_pat = re.compile(
        r'\b([A-Z][A-Z0-9]{1,14})\b'       # strictly uppercase symbol
        r'(?:\s+[A-Za-z ]{0,30})?\s+'
        r'(Inclusion|Exclusion)'
        r'[^\$]*'
        r'([+\-~]?\s*\$\s*[\d,.]+\s*(?:million|billion|bn|M|B))',
    )
    for m in inline_pat.finditer(text):
        sym       = m.group(1)
        event_raw = m.group(2).lower()
        flow_raw  = m.group(3)
        if sym in FLOW_EXCLUDE_WORDS or len(sym) < 2:
            continue
        entry = {"symbol": sym, "flow_usd": _normalise_flow(flow_raw)}
        if "inclusion" in event_raw:
            if not any(e["symbol"] == sym for e in added):
                added.append(entry)
        else:
            if not any(e["symbol"] == sym for e in removed):
                removed.append(entry)

    return removed, added


def _extract_stocks_contextual(text: str, context_keywords: list[str]) -> list[str]:
    """Fallback: find ALL-CAPS stock symbols near context keywords."""
    candidates = []
    low = text.lower()
    for kw in context_keywords:
        idx = 0
        while True:
            idx = low.find(kw, idx)
            if idx < 0:
                break
            window = text[max(0, idx - 50):idx + 350]
            syms = re.findall(r'\b([A-Z][A-Z0-9]{2,14})\b', window)
            for s in syms:
                if s not in FLOW_EXCLUDE_WORDS and s not in candidates:
                    candidates.append(s)
            idx += len(kw)
    return candidates


def _validate_with_kite(symbols: list[str]) -> list[str]:
    """Check symbols against Kite NSE instruments. Returns confirmed ones."""
    try:
        token_path = os.path.join(os.path.dirname(__file__), "access_token.txt")
        with open(token_path) as f:
            access_token = f.read().strip()
        from kiteconnect import KiteConnect
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.getenv("KITE_API_KEY", "")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        valid = []
        for sym in symbols:
            try:
                q = kite.quote(f"NSE:{sym}")
                if q:
                    valid.append(sym)
                time.sleep(0.08)
            except Exception:
                pass
        return valid
    except Exception as e:
        print(f"  [kite] Validation unavailable ({e}) — skipping")
        return symbols


# ── Interactive confirmation ──────────────────────────────────────────────────

def _ask(label: str, current: str) -> str:
    print(f"\n  {label}: {current or '(not found)'}")
    resp = input("  Accept? [Y/n/edit] ").strip().lower()
    if resp in ("", "y", "yes"):
        return current or ""
    return input(f"  Enter {label}: ").strip()


def _ask_stocks(label: str, stocks: list[dict]) -> list[dict]:
    """Confirm/edit a list of {symbol, flow_usd} dicts."""
    if stocks:
        print(f"\n  {label}:")
        for s in stocks:
            print(f"    {s['symbol']:<16} flow ~{s['flow_usd']}")
    else:
        print(f"\n  {label}: (none found)")

    resp = input("  Accept? [Y/n/edit] ").strip().lower()
    if resp in ("", "y", "yes"):
        return stocks

    print("  Enter each as 'SYMBOL:FLOW' (e.g. RVNL:135M), one per line. Empty line to finish.")
    result = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        if ":" in line:
            sym, flow = line.split(":", 1)
            result.append({"symbol": sym.strip().upper(), "flow_usd": flow.strip()})
        else:
            result.append({"symbol": line.strip().upper(), "flow_usd": "TBD"})
    return result


# ── Patch econ_calendar.py ────────────────────────────────────────────────────

def _patch_econ_calendar(quarter: str, flow: str, flow_usd: str,
                          removed: list[dict], added: list[dict]):
    path = os.path.join(os.path.dirname(__file__), "econ_calendar.py")
    with open(path) as f:
        src = f.read()

    q_pattern = re.compile(
        r'(\{[^{}]*"quarter"\s*:\s*"' + re.escape(quarter) + r'"[^{}]*\})',
        re.DOTALL,
    )
    m = q_pattern.search(src)
    if not m:
        print(f"\n  Quarter '{quarter}' not found in econ_calendar.py — edit manually.")
        return

    old_block = m.group(1)
    new_block  = old_block

    new_block = re.sub(r'"india_flow"\s*:\s*None',    f'"india_flow":     "{flow}"',    new_block)
    new_block = re.sub(r'"india_flow_usd"\s*:\s*None', f'"india_flow_usd": "{flow_usd}"', new_block)

    def _fmt_list(items: list[dict]) -> str:
        if not items:
            return "[]"
        lines = "[\n"
        for item in items:
            lines += f'            {{"symbol": "{item["symbol"]}", "flow_usd": "{item["flow_usd"]}"}},\n'
        lines += "        ]"
        return lines

    new_block = re.sub(r'"removed"\s*:\s*\[\s*\]', f'"removed": {_fmt_list(removed)}', new_block)
    new_block = re.sub(r'"added"\s*:\s*\[\s*\]',   f'"added":   {_fmt_list(added)}',   new_block)

    new_src = src[:m.start(1)] + new_block + src[m.end(1):]
    with open(path, "w") as f:
        f.write(new_src)
    print(f"\n  ✅ econ_calendar.py updated for {quarter}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    given_url = sys.argv[2].strip() if len(sys.argv) >= 3 else None

    # ── Always fetch the MSCI official page first ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MSCI Calendar Updater")
    print(f"{'='*60}")
    print(f"\n  Checking https://www.msci.com/indexes/quarterly-index-review …")
    official = fetch_msci_official()

    if official:
        q_label  = official.get("quarter_label", "")
        ann_date = official.get("announce_date_str", "")
        exe_date = official.get("execution_date_str", "")
        additions = official.get("acwi_additions", "?")
        deletions = official.get("acwi_deletions", "?")
        print(f"  Latest MSCI quarter : {q_label}")
        if ann_date:
            print(f"  Announced           : {ann_date}")
        if exe_date:
            print(f"  Execution date      : {exe_date}")
        print(f"  ACWI IMI globally   : {additions} additions, {deletions} deletions")
        print(f"  (India-specific stock list is NOT in the official PDF — need financial site)")
    else:
        print("  Could not reach msci.com — continuing without official data.")
        q_label = ""

    # ── Determine quarter to update ───────────────────────────────────────────
    if len(sys.argv) < 2:
        # No args: just show what's current and exit
        if q_label:
            print(f"\n  Run:  python update_msci_calendar.py \"{q_label}\" <india-article-url>")
            print(f"  e.g.: python update_msci_calendar.py \"{q_label}\" "
                  f"https://univest.in/blogs/msci-rebalancing-{q_label.lower().replace(' ', '-')}")
        else:
            print('\n  Usage: python update_msci_calendar.py "Aug 2026" [url]')
        sys.exit(0)

    quarter = sys.argv[1].strip()

    print(f"\n{'─'*60}")
    print(f"  Updating quarter: {quarter}")
    print(f"{'─'*60}")

    # ── Resolve which URLs to fetch ───────────────────────────────────────────
    if given_url:
        print(f"\n  Using provided URL: {given_url[:90]}")
        urls = [given_url]
    else:
        print(f"\n  Searching known financial sites for MSCI {quarter} announcement…")
        urls = _find_article_urls(quarter, max_results=5)
        if urls:
            print(f"  Found {len(urls)} article(s):")
            for u in urls:
                print(f"    {u[:90]}")
        else:
            month, year = quarter.split()[0], quarter.split()[-1]
            print(f"\n  Auto-discovery found nothing.")
            print(f"  Tip: Google  →  site:univest.in MSCI rebalancing {month} {year}")
            print(f"       or       univest.in/blogs/msci-rebalancing-{month.lower()}-{year}")
            manual = input("\n  Paste article URL (or press Enter to skip): ").strip()
            urls = [manual] if manual else []

    # ── Fetch and extract ─────────────────────────────────────────────────────
    combined_text = ""
    for url in urls[:4]:
        print(f"\n  Fetching {url[:70]}…")
        text = _fetch_text(url)
        if text:
            combined_text += " " + text
            print(f"  Got {len(text):,} chars")

    if not combined_text.strip():
        print("  Could not fetch any articles — switching to manual input.")

    flow_dir, flow_amt = _extract_flow(combined_text)
    print(f"\n  Extracted flow direction : {flow_dir or 'not found'}")
    print(f"  Extracted flow amount    : {flow_amt or 'not found'}")

    # Try structured table extraction first, fall back to contextual
    removed_stocks, added_stocks = _extract_stock_table(combined_text)

    if not removed_stocks:
        print("  Table extraction found nothing — trying contextual extraction…")
        raw_removed = _extract_stocks_contextual(
            combined_text, ["removed", "excluded", "exit", "deletion", "exclusion"])
        raw_added   = _extract_stocks_contextual(
            combined_text, ["added", "included", "inclusion", "addition", "enters", "join"])
        print(f"  Validating {len(raw_removed + raw_added)} candidates against Kite…")
        valid_removed = _validate_with_kite(raw_removed[:20])
        valid_added   = _validate_with_kite(raw_added[:20])
        removed_stocks = [{"symbol": s, "flow_usd": "TBD"} for s in valid_removed]
        added_stocks   = [{"symbol": s, "flow_usd": "TBD"} for s in valid_added]
    else:
        print(f"  Table extraction found {len(removed_stocks)} removed, {len(added_stocks)} added")

    # ── User confirmation ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Review extracted data (Enter to accept, or type to edit)")
    print(f"{'─'*60}")

    flow_dir_final = _ask("Flow direction (INFLOW/OUTFLOW)", flow_dir or "")
    if not flow_dir_final:
        flow_dir_final = input("  Enter manually (INFLOW/OUTFLOW): ").strip().upper()

    flow_amt_final = _ask("Flow amount (e.g. 800M–1B)", flow_amt or "")
    if not flow_amt_final:
        flow_amt_final = input("  Enter manually (e.g. 800M–1B): ").strip()

    removed_final = _ask_stocks("REMOVED stocks → SHORT these (Leg 1)", removed_stocks)
    added_final   = _ask_stocks("ADDED stocks   → LONG these  (Leg 1)", added_stocks)

    # ── Summary + write ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  MSCI {quarter} — writing to econ_calendar.py:")
    print(f"{'─'*60}")
    arrow = "📉" if flow_dir_final == "OUTFLOW" else "📈"
    print(f"  {arrow} India {flow_dir_final}  ~${flow_amt_final}")
    print(f"  SHORT (removed):")
    for s in removed_final:
        print(f"    {s['symbol']:<16}  ~{s['flow_usd']}")
    print(f"  LONG  (added):")
    for s in added_final:
        print(f"    {s['symbol']:<16}  ~{s['flow_usd']}")
    print(f"{'─'*60}")

    confirm = input("\n  Write? [Y/n] ").strip().lower()
    if confirm in ("", "y", "yes"):
        _patch_econ_calendar(
            quarter, flow_dir_final, flow_amt_final,
            removed_final, added_final,
        )
        print("  Done. Restart ml_runner to pick up changes.\n")
    else:
        print("  Aborted — no changes written.\n")


if __name__ == "__main__":
    main()
