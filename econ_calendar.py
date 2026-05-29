"""
econ_calendar.py — Economic calendar: fetch once at startup, warn at T-5.

Usage:
    # At startup
    from econ_calendar import load_and_print
    load_and_print()

    # Each minute in main loop (pass a persistent set across calls)
    from econ_calendar import check_upcoming
    _warned = set()
    check_upcoming(_warned)

    # In signal gate / position monitor
    from econ_calendar import rate_decision_imminent, msci_close_block
    if rate_decision_imminent():
        ...  # block entry or exit position
    if msci_close_block():
        ...  # block new entries after 14:30 on MSCI execution days
"""

import requests
from datetime import datetime, date, timezone, timedelta

IST             = timezone(timedelta(hours=5, minutes=30))
WARN_MINS       = 5    # warn and block this many minutes before the event
POST_EVENT_MINS = 10   # stay blocked this many minutes after the event

# ── MSCI Quarterly Rebalancing Calendar ──────────────────────────────────────
#
# Playbook (3 legs, happens 4× per year):
#   Leg 1 — Announcement day to T-2: SHORT removed stocks, LONG added stocks
#   Leg 2 — Execution day morning:   Buy index PUT (Sensex/Nifty), ride close flush
#   Leg 3 — Post-execution:          Optional long on removed stocks (overhang cleared)
#
# Key insight: on execution day, individual stocks are already priced in (front-run
# over 17 days). The index PUT is the cleanest trade — $800M+ passive selling hits
# as a single mechanical close-time event (15:00–15:30 IST).
#
# Update each quarter when MSCI announces (~2nd week of Feb/May/Aug/Nov).
# Set india_flow=None and leave removed/added empty for future quarters until announced.

MSCI_EVENTS = [
    {
        "quarter":        "Feb 2026",
        "announce_date":  date(2026, 2, 10),
        "execution_date": date(2026, 2, 27),
        "effective_date": date(2026, 3, 2),
        "india_flow":     None,   # update when known
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "May 2026",
        "announce_date":  date(2026, 5, 12),
        "execution_date": date(2026, 5, 29),
        "effective_date": date(2026, 6, 1),
        "india_flow":     "OUTFLOW",
        "india_flow_usd": "800M–1B",
        # Validated: removed stocks rallied on exec day (front-run done);
        # MCX fell -6.5% despite being added (profit-taking + broad dump).
        # Sensex -1600pts, 600pt candle at 15:00.
        "removed": [
            {"symbol": "HYUNDAI",   "flow_usd": "281M"},
            {"symbol": "JUBLFOOD",  "flow_usd": "161M"},
            {"symbol": "KALYANKJIL","flow_usd": "137M"},
            {"symbol": "RVNL",      "flow_usd": "136M"},
        ],
        "added": [
            {"symbol": "FEDERALBNK","flow_usd": "TBD"},
            {"symbol": "MCX",       "flow_usd": "TBD"},
            {"symbol": "NATIONALUM","flow_usd": "TBD"},
            {"symbol": "INDIANB",   "flow_usd": "TBD"},
        ],
    },
    {
        "quarter":        "Aug 2026",
        "announce_date":  date(2026, 8, 12),   # approximate — confirm when published
        "execution_date": date(2026, 8, 31),
        "effective_date": date(2026, 9, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "Nov 2026",
        "announce_date":  date(2026, 11, 11),  # approximate — confirm when published
        "execution_date": date(2026, 11, 30),
        "effective_date": date(2026, 12, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "Feb 2027",
        "announce_date":  date(2027, 2, 10),
        "execution_date": date(2027, 2, 26),
        "effective_date": date(2027, 3, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "May 2027",
        "announce_date":  date(2027, 5, 12),
        "execution_date": date(2027, 5, 30),
        "effective_date": date(2027, 6, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "Aug 2027",
        "announce_date":  date(2027, 8, 11),
        "execution_date": date(2027, 8, 29),
        "effective_date": date(2027, 9, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
    {
        "quarter":        "Nov 2027",
        "announce_date":  date(2027, 11, 10),
        "execution_date": date(2027, 11, 28),
        "effective_date": date(2027, 12, 1),
        "india_flow":     None,
        "india_flow_usd": None,
        "removed":        [],
        "added":          [],
    },
]

# Quick-lookup sets derived from MSCI_EVENTS
_MSCI_EXECUTION_DATES  = {e["execution_date"]  for e in MSCI_EVENTS}
_MSCI_ANNOUNCE_DATES   = {e["announce_date"]   for e in MSCI_EVENTS}

MSCI_BLOCK_FROM_HOUR = 14
MSCI_BLOCK_FROM_MIN  = 30

# ── ForexFactory calendar config ─────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
_COUNTRIES   = {"INR", "USD", "EUR", "CNY", "JPY", "GBP"}
_SHOW_IMPACT = {"HIGH", "MEDIUM"}
_RATE_KEYWORDS = ("rate decision", "interest rate", "repo rate", "rbi rate", "monetary policy")

_events: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_rate_decision(title: str) -> bool:
    return any(kw in title.lower() for kw in _RATE_KEYWORDS)


def _minutes_until(time_str: str) -> float | None:
    try:
        now = datetime.now(IST)
        event_dt = now.replace(
            hour=int(time_str[:2]), minute=int(time_str[3:5]),
            second=0, microsecond=0,
        )
        return (event_dt - now).total_seconds() / 60
    except Exception:
        return None


def _get_msci_event(d: date) -> dict | None:
    for ev in MSCI_EVENTS:
        if ev["announce_date"] == d or ev["execution_date"] == d:
            return ev
    return None


def _days_until(d: date) -> int:
    return (d - datetime.now(IST).date()).days


def _print_msci_section():
    today = datetime.now(IST).date()

    # Check if today is an announcement or execution day
    ev = _get_msci_event(today)

    if ev and ev["announce_date"] == today:
        print(f"\n  {'!'*56}")
        print(f"  🔔 MSCI {ev['quarter'].upper()} — ANNOUNCEMENT DAY")
        print(f"  {'!'*56}")
        flow = ev.get("india_flow")
        amt  = ev.get("india_flow_usd")
        if flow:
            arrow = "📉" if flow == "OUTFLOW" else "📈"
            print(f"  {arrow} India NET {flow}  ~${amt}")
            print(f"  Execution: {ev['execution_date'].strftime('%d %b %Y')} "
                  f"(in {_days_until(ev['execution_date'])} days) — buy index PUT then")
            removed = ev.get("removed", [])
            added   = ev.get("added",   [])
            if removed:
                syms = "  ".join(s["symbol"] for s in removed)
                flows = "  ".join(f"${s['flow_usd']}" for s in removed)
                print(f"  SHORT (removed): {syms}")
                print(f"                   {flows}")
            if added:
                syms = "  ".join(s["symbol"] for s in added)
                print(f"  LONG  (added):   {syms}")
        else:
            print(f"  Flow details TBD — check MSCI press release")
            print(f"  Execution: {ev['execution_date'].strftime('%d %b %Y')} "
                  f"(in {_days_until(ev['execution_date'])} days)")
        print(f"  {'─'*56}")
        print(f"  📋 UPDATE CALENDAR:")
        print(f"     python update_msci_calendar.py \"{ev['quarter']}\" <univest-article-url>")
        print(f"     (run with no args first to auto-detect from msci.com)")
        print(f"  {'!'*56}\n")
        return

    if ev and ev["execution_date"] == today:
        print(f"\n  {'!'*56}")
        print(f"  ⚠️  MSCI {ev['quarter'].upper()} — EXECUTION DAY")
        flow = ev.get("india_flow")
        amt  = ev.get("india_flow_usd")
        if flow:
            print(f"  India NET {flow} ~${amt} hits at close (15:00–15:30 IST)")
        print(f"  New entries BLOCKED after "
              f"{MSCI_BLOCK_FROM_HOUR:02d}:{MSCI_BLOCK_FROM_MIN:02d} IST")
        print(f"  {'!'*56}\n")
        return

    # Show countdown to next upcoming MSCI event (announce or execute)
    upcoming = []
    for e in MSCI_EVENTS:
        days_to_ann  = _days_until(e["announce_date"])
        days_to_exec = _days_until(e["execution_date"])
        if days_to_ann >= 0:
            upcoming.append((days_to_ann, "announce", e))
        elif days_to_exec >= 0:
            upcoming.append((days_to_exec, "execute", e))
    if not upcoming:
        return
    upcoming.sort()
    days, kind, e = upcoming[0]
    if days <= 30:
        if kind == "announce":
            print(f"  📅 MSCI {e['quarter']} announcement in {days} days "
                  f"({e['announce_date'].strftime('%d %b')})  "
                  f"→ execution {e['execution_date'].strftime('%d %b')}")
        else:
            flow = e.get("india_flow")
            amt  = e.get("india_flow_usd")
            flow_str = f"  India {flow} ~${amt}" if flow else ""
            print(f"  ⚠️  MSCI {e['quarter']} execution in {days} days "
                  f"({e['execution_date'].strftime('%d %b')}) — "
                  f"buy index PUT today{flow_str}")


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_print():
    """Fetch today's HIGH/MEDIUM events, cache in memory, print to terminal."""
    global _events
    _events = []

    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers=_HEADERS, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [calendar] Could not fetch events: {e}")
        return

    today = datetime.now(IST).strftime("%Y-%m-%d")

    for e in data:
        if (e.get("date") or "")[:10] != today:
            continue
        country = (e.get("country") or "").upper()
        if country not in _COUNTRIES:
            continue
        impact = (e.get("impact") or "low").upper()
        if impact not in _SHOW_IMPACT:
            continue
        try:
            utc_dt   = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
            time_str = utc_dt.astimezone(IST).strftime("%H:%M")
        except Exception:
            time_str = "--:--"
        title = e.get("title", "")
        _events.append({
            "time":          time_str,
            "country":       country,
            "impact":        impact,
            "event":         title,
            "forecast":      e.get("forecast") or "",
            "previous":      e.get("previous") or "",
            "rate_decision": country == "INR" and _is_rate_decision(title),
        })

    _events.sort(key=lambda x: x["time"])

    print(f"\n  {'─'*56}")
    print(f"  ECONOMIC EVENTS TODAY — {today}")
    print(f"  {'─'*56}")
    if not _events:
        print("  No high/medium impact events today.")
    for ev in _events:
        flag = "🔴" if ev["impact"] == "HIGH" else "🟡"
        tag  = " [RATE DECISION]" if ev["rate_decision"] else ""
        fc   = f"  fcst={ev['forecast']}" if ev["forecast"] else ""
        pv   = f"  prev={ev['previous']}" if ev["previous"] else ""
        print(f"  {flag} {ev['time']}  [{ev['country']}]  {ev['event']}{tag}{fc}{pv}")
    print(f"  {'─'*56}")

    _print_msci_section()


def check_upcoming(fired_set: set) -> list[dict]:
    """
    Call once per minute. Prints a warning for any cached event within WARN_MINS.
    fired_set — persistent set owned by caller to avoid duplicate warnings.
    """
    imminent = []
    for ev in _events:
        mins = _minutes_until(ev["time"])
        if mins is None:
            continue
        if 0 <= mins <= WARN_MINS:
            key = f"{ev['time']}_{ev['event']}"
            if key not in fired_set:
                fired_set.add(key)
                flag = "🔴" if ev["impact"] == "HIGH" else "🟡"
                tag  = (" ⚠️  RATE DECISION — runner will exit position & pause entries"
                        if ev["rate_decision"] else "")
                print(f"\n  {'!'*56}")
                print(f"  {flag} EVENT IN ~{int(mins)}m: [{ev['country']}] {ev['event']}{tag}")
                print(f"  {'!'*56}\n")
            imminent.append(ev)
    return imminent


def rate_decision_imminent() -> bool:
    """True if an INR rate decision is within WARN_MINS ahead or POST_EVENT_MINS after."""
    for ev in _events:
        if not ev["rate_decision"]:
            continue
        mins = _minutes_until(ev["time"])
        if mins is not None and -POST_EVENT_MINS <= mins <= WARN_MINS:
            return True
    return False


def msci_close_block() -> bool:
    """
    True if today is an MSCI execution date AND time >= 14:30 IST.
    Blocks new entries during the passive close-window flow.
    """
    now = datetime.now(IST)
    if now.date() not in _MSCI_EXECUTION_DATES:
        return False
    return (now.hour, now.minute) >= (MSCI_BLOCK_FROM_HOUR, MSCI_BLOCK_FROM_MIN)


def is_msci_rebalance_day() -> bool:
    """True if today is an MSCI execution date (regardless of time)."""
    return datetime.now(IST).date() in _MSCI_EXECUTION_DATES


def is_msci_announce_day() -> bool:
    """True if today is an MSCI announcement date."""
    return datetime.now(IST).date() in _MSCI_ANNOUNCE_DATES
