"""
price_tracker.py — Background option LTP monitor.

Polls the traded option's last price every 10 seconds in a separate thread.
Computes a three-phase stop system and exposes state for HTL fusion.

Phases:
  1. HARD STOP  — option drops PT_HARD_STOP_PCT from entry → EXIT
  2. BREAKEVEN  — option gains PT_BREAKEVEN_TRIGGER from entry → stop at entry
  3. TRAILING   — 40% of peak gain retraced → EXIT (tightens to 25% if
                  peak is stale for PT_STALE_PEAK_SEC)
"""

import threading
import time
from datetime import datetime

from config import (
    PT_POLL_INTERVAL, PT_HARD_STOP_PCT, PT_BREAKEVEN_TRIGGER,
    PT_TRAIL_DRAWDOWN, PT_STALE_PEAK_SEC, PT_STALE_DRAWDOWN,
)


class PriceTracker:
    """Tracks option LTP in a background thread and computes stop phases."""

    def __init__(self, kite, symbol, entry_price, notify_fn=None):
        """
        Parameters
        ----------
        kite        : KiteConnect client
        symbol      : full exchange:tradingsymbol (e.g. "NFO:NIFTY2640322850CE")
        entry_price : float — price at which the option was bought
        notify_fn   : callable(str) — sends telegram/console alerts
        """
        self.kite = kite
        self.symbol = symbol
        self.entry_price = float(entry_price)
        self.notify = notify_fn or (lambda msg: None)

        # Live state — read by HTL on each tick
        self.current_ltp = entry_price
        self.peak_price = entry_price
        self.peak_time = datetime.now()
        self.phase = "HARD_STOP"       # HARD_STOP → BREAKEVEN → TRAILING
        self.drawdown_pct = 0.0        # % of gain lost from peak
        self.gain_pct = 0.0            # current gain as % of entry
        self.peak_stale = False        # True if no new peak for PT_STALE_PEAK_SEC
        self.verdict = None            # None = no override, "EXIT" = price stop hit

        # Internal
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Launch the background polling thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="price-tracker"
        )
        self._thread.start()

    def stop(self):
        """Signal the thread to stop."""
        self._stop.set()

    def snapshot(self):
        """Return a dict of current state for HTL consumption."""
        with self._lock:
            return {
                "ltp": self.current_ltp,
                "entry": self.entry_price,
                "peak": self.peak_price,
                "phase": self.phase,
                "gain_pct": self.gain_pct,
                "drawdown_pct": self.drawdown_pct,
                "peak_stale": self.peak_stale,
                "verdict": self.verdict,
            }

    # ── internal ────────────────────────────────────────────────────────

    def _fetch_ltp(self):
        try:
            data = self.kite.ltp(self.symbol)
            return data[self.symbol]["last_price"]
        except Exception:
            return None

    def _poll_loop(self):
        while not self._stop.is_set():
            ltp = self._fetch_ltp()
            if ltp is not None:
                self._update(ltp)
            self._stop.wait(PT_POLL_INTERVAL)

    def _update(self, ltp):
        with self._lock:
            self.current_ltp = ltp
            self.gain_pct = (ltp - self.entry_price) / self.entry_price

            # ── Update peak ──────────────────────────────────────────
            if ltp > self.peak_price:
                self.peak_price = ltp
                self.peak_time = datetime.now()
                self.peak_stale = False
            else:
                elapsed = (datetime.now() - self.peak_time).total_seconds()
                self.peak_stale = elapsed >= PT_STALE_PEAK_SEC

            # ── Phase transitions ────────────────────────────────────
            gain_from_entry = ltp - self.entry_price
            peak_gain = self.peak_price - self.entry_price

            if self.phase == "HARD_STOP":
                if self.gain_pct >= PT_BREAKEVEN_TRIGGER:
                    self.phase = "BREAKEVEN"
                    self.notify(
                        f"📊 Price tracker: +{self.gain_pct:.0%} from entry "
                        f"(₹{self.entry_price:.0f}→₹{ltp:.0f}) — "
                        f"stop moved to BREAKEVEN"
                    )

            elif self.phase == "BREAKEVEN":
                # Stay in breakeven until peak gain establishes
                # (trailing only meaningful once there's a peak to trail)
                if peak_gain > 0:
                    self.phase = "TRAILING"

            # ── Compute drawdown % of gain ───────────────────────────
            if peak_gain > 0:
                drawdown_from_peak = self.peak_price - ltp
                self.drawdown_pct = drawdown_from_peak / peak_gain
            else:
                self.drawdown_pct = 0.0

            # ── Check stops ──────────────────────────────────────────
            if self.verdict == "EXIT":
                return  # already triggered, don't re-alert

            # Phase 1: hard stop
            if self.gain_pct <= -PT_HARD_STOP_PCT:
                self.verdict = "EXIT"
                self.notify(
                    f"🚨 PRICE STOP: option down {self.gain_pct:.0%} "
                    f"(₹{self.entry_price:.0f}→₹{ltp:.0f}) — hard stop hit"
                )
                return

            # Phase 2: breakeven stop (only after breakeven was triggered)
            if self.phase in ("BREAKEVEN", "TRAILING") and ltp < self.entry_price:
                self.verdict = "EXIT"
                self.notify(
                    f"🚨 BREAKEVEN STOP: option fell back below entry "
                    f"(₹{self.entry_price:.0f}→₹{ltp:.0f}) — closing"
                )
                return

            # Phase 3: trailing drawdown of gain
            if self.phase == "TRAILING" and peak_gain > 0:
                threshold = PT_STALE_DRAWDOWN if self.peak_stale else PT_TRAIL_DRAWDOWN
                if self.drawdown_pct >= threshold:
                    label = "STALE TRAIL" if self.peak_stale else "TRAIL"
                    self.notify(
                        f"🚨 {label} STOP: gave back {self.drawdown_pct:.0%} of "
                        f"₹{peak_gain:.0f} gain "
                        f"(peak ₹{self.peak_price:.0f}→₹{ltp:.0f}) — closing"
                    )
                    self.verdict = "EXIT"
                    return


def resolve_option_symbol(instruments, strike, option_type, expiry,
                          name="NIFTY", exchange="NFO"):
    """
    Find the full exchange:tradingsymbol for a given strike/type/expiry.
    Returns e.g. "NFO:NIFTY2640322850CE" or None.
    """
    opt_type = option_type.upper()
    if opt_type in ("CALL", "CE"):
        opt_type = "CE"
    elif opt_type in ("PUT", "PE"):
        opt_type = "PE"

    for ins in instruments:
        if (ins["name"] == name
                and ins["expiry"] == expiry
                and ins["strike"] == strike
                and ins["instrument_type"] == opt_type):
            return f"{exchange}:{ins['tradingsymbol']}"
    return None
