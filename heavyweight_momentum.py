"""
heavyweight_momentum.py — Track top NIFTY heavyweights and their ROC.
Used by HTL to detect move exhaustion or confirm continuation.
"""

from collections import deque
from datetime import datetime

from kite_interface import get_kite_client
from config import (HW_ROC_WINDOW, HW_ROC_WINDOW_FAST,
                    HW_STALL_WINDOW, HW_STALL_RATIO,
                    HW_STRONG_THRESHOLD, HW_WEAK_THRESHOLD)

# ─────────────────────────────────────────────────────────────────────
# Top 10 NIFTY heavyweights by approximate free-float weight (Mar 2026)
# Update periodically when NSE rebalances (every 6 months).
# ─────────────────────────────────────────────────────────────────────
NIFTY_HEAVYWEIGHTS = [
    {"symbol": "NSE:HDFCBANK",  "name": "HDFC Bank",  "weight": 13.1},
    {"symbol": "NSE:RELIANCE",  "name": "Reliance",   "weight": 9.2},
    {"symbol": "NSE:ICICIBANK", "name": "ICICI Bank",  "weight": 8.5},
    {"symbol": "NSE:INFY",      "name": "Infosys",     "weight": 6.0},
    {"symbol": "NSE:TCS",       "name": "TCS",          "weight": 4.5},
    {"symbol": "NSE:BHARTIARTL","name": "Bharti Airtel","weight": 4.2},
    {"symbol": "NSE:ITC",       "name": "ITC",          "weight": 4.0},
    {"symbol": "NSE:LT",        "name": "L&T",          "weight": 3.8},
    {"symbol": "NSE:SBIN",      "name": "SBI",          "weight": 3.2},
    {"symbol": "NSE:KOTAKBANK", "name": "Kotak Bank",   "weight": 3.0},
]

TOTAL_WEIGHT = sum(s["weight"] for s in NIFTY_HEAVYWEIGHTS)

kite = get_kite_client()


def fetch_heavyweight_prices():
    """Fetch LTP for all heavyweights in a single API call."""
    symbols = [s["symbol"] for s in NIFTY_HEAVYWEIGHTS]
    try:
        quotes = kite.quote(symbols)
    except Exception:
        return None

    prices = {}
    for hw in NIFTY_HEAVYWEIGHTS:
        sym = hw["symbol"]
        if sym in quotes:
            prices[sym] = quotes[sym]["last_price"]
    return prices if len(prices) == len(NIFTY_HEAVYWEIGHTS) else None


def update_heavyweight_history(hw_history, prices):
    """Append current prices snapshot to history deque."""
    hw_history.append({"time": datetime.now(), "prices": prices})


HW_CACHE_FILE = "data/hw_history.json"


def clear_hw_history():
    """Delete cached heavyweight history (end-of-day cleanup)."""
    import os
    try:
        os.remove(HW_CACHE_FILE)
    except FileNotFoundError:
        pass


def save_hw_history(hw_history):
    """Persist heavyweight history to disk."""
    import json
    records = [
        {"time": entry["time"].isoformat(), "prices": entry["prices"]}
        for entry in hw_history
    ]
    try:
        with open(HW_CACHE_FILE, "w") as f:
            json.dump(records, f)
    except Exception:
        pass


def restore_hw_history(hw_history):
    """Reload heavyweight history from disk on startup."""
    import json, os
    if not os.path.exists(HW_CACHE_FILE):
        return
    try:
        with open(HW_CACHE_FILE, "r") as f:
            records = json.load(f)
        cutoff = datetime.now() - __import__("datetime").timedelta(minutes=60)
        for rec in records:
            t = datetime.fromisoformat(rec["time"])
            if t >= cutoff:
                hw_history.append({"time": t, "prices": rec["prices"]})
        if hw_history:
            print(f"✅ HW history restored ({len(hw_history)} candles)")
    except Exception as e:
        print(f"⚠ HW history restore failed: {e}")


def compute_heavyweight_roc(hw_history, window=None):
    """
    Compute weighted ROC for heavyweights over `window` candles.

    Returns dict:
        weighted_roc    : float — weight-adjusted ROC (%)
        movers          : list  — top movers with individual ROC
        direction       : str   — "BULLISH" / "BEARISH" / "FLAT"
        strength        : str   — "STRONG" / "MODERATE" / "WEAK"
        aligned_count   : int   — how many heavyweights move in same direction
        aligned_weight  : float — total weight of aligned heavyweights
    """
    if window is None:
        window = HW_ROC_WINDOW

    if len(hw_history) < window + 1:
        return None

    current = hw_history[-1]["prices"]
    past = hw_history[-(window + 1)]["prices"]

    weighted_roc = 0.0
    movers = []
    bullish_weight = 0.0
    bearish_weight = 0.0

    for hw in NIFTY_HEAVYWEIGHTS:
        sym = hw["symbol"]
        if sym not in current or sym not in past or past[sym] == 0:
            continue

        roc = ((current[sym] - past[sym]) / past[sym]) * 100
        norm_weight = hw["weight"] / TOTAL_WEIGHT
        weighted_roc += roc * norm_weight

        if roc > 0.05:
            bullish_weight += hw["weight"]
        elif roc < -0.05:
            bearish_weight += hw["weight"]

        movers.append({
            "name": hw["name"],
            "symbol": sym,
            "weight": hw["weight"],
            "roc": roc,
        })

    movers.sort(key=lambda m: abs(m["roc"]), reverse=True)

    if weighted_roc > 0.05:
        direction = "BULLISH"
        aligned_weight = bullish_weight
    elif weighted_roc < -0.05:
        direction = "BEARISH"
        aligned_weight = bearish_weight
    else:
        direction = "FLAT"
        aligned_weight = max(bullish_weight, bearish_weight)

    aligned_count = sum(
        1 for m in movers
        if (direction == "BULLISH" and m["roc"] > 0.05)
        or (direction == "BEARISH" and m["roc"] < -0.05)
    )

    abs_roc = abs(weighted_roc)
    if abs_roc >= HW_STRONG_THRESHOLD:
        strength = "STRONG"
    elif abs_roc >= HW_WEAK_THRESHOLD:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    return {
        "weighted_roc": weighted_roc,
        "direction": direction,
        "strength": strength,
        "aligned_count": aligned_count,
        "aligned_weight": aligned_weight,
        "movers": movers[:5],   # top 5 movers for display
    }


def detect_hw_stall(hw_history):
    """
    Detect if heavyweights have stalled after a move.

    Compares a very short ROC (2 min) against the broad ROC (15 min).
    If broad ROC says "big move happened" but short ROC is near-zero,
    the heavyweights have flatlined — move exhaustion is imminent.

    Returns:
        dict with stalled (bool), broad_roc, short_roc, ratio
        None if insufficient data
    """
    broad_window = HW_ROC_WINDOW
    short_window = HW_STALL_WINDOW

    if len(hw_history) < broad_window + 1:
        return None

    # Broad ROC — the move that happened
    broad_roc = _weighted_roc(hw_history, broad_window)

    # Short ROC — what's happening right now
    if len(hw_history) < short_window + 1:
        return None
    short_roc = _weighted_roc(hw_history, short_window)

    abs_broad = abs(broad_roc)
    abs_short = abs(short_roc)

    # Only flag stall if there was a meaningful broad move
    if abs_broad < HW_WEAK_THRESHOLD:
        return {"stalled": False, "broad_roc": broad_roc,
                "short_roc": short_roc, "ratio": 1.0}

    ratio = abs_short / abs_broad if abs_broad > 0 else 1.0
    stalled = ratio < HW_STALL_RATIO

    return {
        "stalled": stalled,
        "broad_roc": broad_roc,
        "short_roc": short_roc,
        "ratio": ratio,
    }


def _weighted_roc(hw_history, window):
    """Helper: compute weighted ROC over a given window."""
    current = hw_history[-1]["prices"]
    past = hw_history[-(window + 1)]["prices"]
    w_roc = 0.0
    for hw in NIFTY_HEAVYWEIGHTS:
        sym = hw["symbol"]
        if sym in current and sym in past and past[sym] != 0:
            roc = ((current[sym] - past[sym]) / past[sym]) * 100
            w_roc += roc * (hw["weight"] / TOTAL_WEIGHT)
    return w_roc


def compute_roc_trend(hw_history, lookback=3, window=None):
    """
    Check if heavyweight ROC is accelerating or decelerating.
    Compares the last `lookback` ROC readings.

    Returns:
        "ACCELERATING" — ROC magnitude increasing (move has legs)
        "DECELERATING" — ROC magnitude decreasing (exhaustion)
        "STEADY"       — no clear trend
        None           — insufficient data
    """
    if window is None:
        window = HW_ROC_WINDOW

    needed = window + lookback + 1
    if len(hw_history) < needed:
        return None

    rocs = []
    for i in range(lookback):
        idx = -(1 + i)
        past_idx = idx - window
        if abs(past_idx) > len(hw_history):
            return None
        current = hw_history[idx]["prices"]
        past = hw_history[past_idx]["prices"]

        w_roc = 0.0
        for hw in NIFTY_HEAVYWEIGHTS:
            sym = hw["symbol"]
            if sym in current and sym in past and past[sym] != 0:
                roc = ((current[sym] - past[sym]) / past[sym]) * 100
                w_roc += roc * (hw["weight"] / TOTAL_WEIGHT)
        rocs.append(w_roc)

    # rocs[0] = most recent, rocs[-1] = oldest
    if len(rocs) < 2:
        return None

    # Check if magnitude is consistently increasing or decreasing
    magnitudes = [abs(r) for r in rocs]
    increasing = all(magnitudes[i] >= magnitudes[i + 1] for i in range(len(magnitudes) - 1))
    decreasing = all(magnitudes[i] <= magnitudes[i + 1] for i in range(len(magnitudes) - 1))

    if increasing and magnitudes[0] > magnitudes[-1] * 1.1:
        return "ACCELERATING"
    elif decreasing and magnitudes[-1] > magnitudes[0] * 1.1:
        return "DECELERATING"
    return "STEADY"
