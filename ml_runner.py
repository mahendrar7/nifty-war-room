"""
ml_runner.py — Live ML trading runner.

Reads the war room's 1-min CSV, resamples to 5-min candles, runs the ML model,
places trades via Kite, and manages two-lot runner exits with Telegram alerts.

Usage:
    python ml_runner.py --instrument sensex --lots 6
    python ml_runner.py --instrument nifty --lots 4
    python ml_runner.py --instrument sensex --lots 6 --mode paper
"""

import os
import sys
import csv
import json
import time
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, date, timedelta
from scipy.stats import norm as scipy_norm

from kite_interface import get_kite_client
from notifier import send_telegram_message, send_runner_alert
from ml_engine import (
    MLEngine, Resampler, build_targets, add_lag_features,
    PROBA_THRESHOLD, FORWARD_CANDLES, FeedbackLedger,
)
from gamma_engine import _bs_d1, implied_vol
from config import INSTRUMENT_PROFILES, RISK_FREE_RATE


# =============================================================================
# INSTRUMENT CONFIGS (from calibrated backtest results)
# =============================================================================

RUNNER_CONFIGS = {
    "sensex": {
        "sl_pts":                   15,
        "tp_pts":                   25,
        "runner_ext":               2.16,
        "retracement_pct":          0.07,
        "lot2_floor_pts":           10,
        "sl_circuit_breaker":       2,
        "circuit_pause_candles":    2,
        "counter_trend_pts_limit":  300,
        "min_conf_short":           0.55,
        "min_conf_long":            0.75,
        "lot_size":                 20,
        "default_lots":             6,
        "hold_candles":             9,
        "target_delta":             0.40,
        "candle_minutes":           5,
        "slippage_pts":             2,
        "trail_update_pts":         5,
        "sl_adjust_trigger_pts":    15,
        "sl_adjust_move_pts":       10,
        "spot_trigger_pts":         20,    # wait for spot to move 20pts in signal direction before entering
        "spot_trigger_trend_limit": 200,  # skip trigger when abs(trend_pts) >= 200 — market already trending
        "skip_expiry_day":          True,  # block all entries on weekly expiry day
        "session_disp_long_block_pts": 200, # block LONG when spot is this far below session high AND below session open
        "daily_loss_limit":            15000, # stop new entries for the day once closed PnL hits -Rs 15,000
        "paper_starting_margin":       60000, # simulated starting margin for paper mode
    },
    "nifty": {
        "sl_pts":                   8,
        "tp_pts":                   10,
        "runner_ext":               2.2,
        "retracement_pct":          0.07,
        "lot2_floor_pts":           3,
        "sl_circuit_breaker":       2,
        "circuit_pause_candles":    2,
        "counter_trend_pts_limit":  75,
        "min_conf_short":           0.55,
        "min_conf_long":            0.55,
        "lot_size":                 65,
        "default_lots":             4,
        "hold_candles":             9,
        "target_delta":             0.30,
        "candle_minutes":           5,
        "slippage_pts":             1,
        "trail_update_pts":         2,
        "sl_adjust_trigger_pts":    15,
        "sl_adjust_move_pts":       10,
        "spot_trigger_pts":         None,  # None = enter immediately; set to e.g. 20 to wait for spot confirmation
        "skip_expiry_day":          False, # block all entries on weekly expiry day
        "session_disp_long_block_pts": None, # disabled for NIFTY
        "daily_loss_limit":            None, # disabled for NIFTY
        "paper_starting_margin":       None, # disabled for NIFTY
    },
}


# =============================================================================
# TRADE STATE MACHINE
# =============================================================================

class TradeState(Enum):
    IDLE = "IDLE"
    AWAITING_TRIGGER = "AWAITING_TRIGGER"
    POSITION_OPEN = "POSITION_OPEN"
    LOT1_EXITED = "LOT1_EXITED"
    COOLDOWN = "COOLDOWN"


# =============================================================================
# BLACK-SCHOLES DELTA
# =============================================================================

def bs_delta(S, K, T, r, sigma, option_type="CE"):
    if sigma <= 0 or T <= 0:
        return 0.5 if option_type == "CE" else -0.5
    d1 = _bs_d1(S, K, T, r, sigma)
    if option_type == "CE":
        return scipy_norm.cdf(d1)
    return scipy_norm.cdf(d1) - 1.0


# =============================================================================
# KITE ORDER WRAPPER WITH RETRY
# =============================================================================

MAX_RETRIES = 5

_MARGIN_ERRORS = ("insufficient", "margin", "funds", "credit")

def place_order_safe(kite, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **kwargs)
            return order_id
        except Exception as e:
            err = str(e)
            if any(kw in err.lower() for kw in _MARGIN_ERRORS):
                print(f"  ORDER FAILED (margin): {err}")
                send_telegram_message(f"🚨 MARGIN INSUFFICIENT: {err}\nParams: {kwargs}")
                return None
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Order attempt {attempt+1} failed: {err} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  ORDER FAILED after {MAX_RETRIES} attempts: {err}")
                send_telegram_message(f"⚠️ ORDER FAILED: {err}\nParams: {kwargs}")
                return None


def cancel_order_safe(kite, order_id):
    for attempt in range(3):
        try:
            kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
            return True
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"  Cancel failed for order {order_id}: {e}")
                return False


def get_ltp_safe(kite, symbol, retries=3):
    for attempt in range(retries):
        try:
            quote = kite.quote(symbol)
            return quote[symbol]["last_price"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None


# =============================================================================
# ML RUNNER
# =============================================================================

class MLRunner:

    def __init__(self, instrument, num_lots, kite, paper=False):
        self.instrument = instrument.lower()
        self.cfg = RUNNER_CONFIGS[self.instrument]
        self.profile = INSTRUMENT_PROFILES[instrument.upper()]
        self.num_lots = num_lots
        self.kite = kite
        self.paper = paper

        self.state = TradeState.IDLE
        self.trade = {}
        self.cooldown_remaining = 0
        self.last_candle_count = 0
        self.consecutive_sl = 0
        self.session_pnl = 0.0
        self.available_margin = 0.0
        self.day_suspended = False
        self._warned_events = set()
        self.pending_signal = None   # set when AWAITING_TRIGGER
        self.trigger_expiry = None

        # Paper trading
        self._paper_order_counter = 0
        mode = "paper" if paper else "live"
        self._trade_log_file = f"data/ml_{mode}_log_{self.instrument}.csv"

        # Session trade log for EOD summary
        self.completed_trades = []

        # ML engine — configured for 5-min candles
        self.engine = MLEngine(self.instrument)
        self.engine.model_xgb = f"data/ml_runner_{self.instrument}.ubj"
        self.engine.model_file = f"data/ml_runner_{self.instrument}.joblib"
        self.engine.dataset_file = f"data/ml_runner_dataset_{self.instrument}.csv"
        self.engine.threshold_file = f"data/ml_runner_threshold_{self.instrument}.txt"

        self.resampler = Resampler(candle_minutes=self.cfg["candle_minutes"])

        # Feedback ledger — records SL/TP outcomes for EOD-weighted retraining
        self._feedback = FeedbackLedger(
            path=f"data/ml_feedback_{self.instrument}.csv"
        )
        self._pending_signal_ts = None   # candle_ts of the signal that opened current trade

        # State file for crash recovery
        self.state_file = f"data/ml_runner_state_{self.instrument}.json"

        # Instrument identifiers for Kite
        self.spot_symbol = self.profile["spot_symbol"]
        self.exchange = self.profile["exchange"]
        self.inst_name = self.profile["name"]
        self.strike_step = self.profile["strike_step"]

        self._instruments_cache = None
        self._instruments_date = None

    # ── Paper / Live Helpers ──────────────────────────────────────────────────

    def _place_order(self, **kwargs):
        if self.paper:
            self._paper_order_counter += 1
            oid = f"PAPER-{self._paper_order_counter}"
            action = kwargs.get("transaction_type", "?")
            qty = kwargs.get("quantity", "?")
            sym = kwargs.get("tradingsymbol", "?")
            otype = kwargs.get("order_type", "?")
            trigger = kwargs.get("trigger_price", "")
            print(f"  [PAPER] Order {oid}: {action} {qty}x {sym} {otype}"
                  f"{f' trigger={trigger}' if trigger else ''}")
            return oid
        return place_order_safe(self.kite, **kwargs)

    def _cancel_order(self, order_id):
        if self.paper:
            print(f"  [PAPER] Cancelled order {order_id}")
            return True
        return cancel_order_safe(self.kite, order_id)

    def _modify_sl_trigger(self, order_id, new_trigger):
        if self.paper:
            print(f"  [PAPER] Modified order {order_id} trigger → {new_trigger}")
            return True
        try:
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id,
                trigger_price=new_trigger,
            )
            return True
        except Exception as e:
            print(f"  ⚠ modify_order failed: {e}")
            return False

    def _send_alert(self, msg):
        prefix = "📝 [PAPER] " if self.paper else ""
        send_runner_alert(prefix + msg)

    def _log_trade(self, action, price, qty, pnl=None, notes=""):
        t = self.trade
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "symbol": t.get("trading_sym", ""),
            "direction": t.get("direction", ""),
            "strike": t.get("strike", ""),
            "qty": qty,
            "price": price,
            "entry_price": t.get("entry_price", ""),
            "sl": t.get("sl_trigger", ""),
            "tp": t.get("tp_price", ""),
            "pnl": pnl if pnl is not None else "",
            "notes": notes,
        }
        write_header = not os.path.exists(self._trade_log_file)
        with open(self._trade_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── Startup ────────────────────────────────────────────────────────────────

    def startup(self):
        mode_label = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  ML RUNNER — {self.instrument.upper()} [{mode_label}]")
        print(f"  Lots: {self.num_lots} ({self.num_lots * self.cfg['lot_size']} qty)")
        print(f"  SL: {self.cfg['sl_pts']}pts | TP: {self.cfg['tp_pts']}pts | "
              f"TP2: {round(self.cfg['tp_pts'] * self.cfg['runner_ext'], 1)}pts | "
              f"Slippage: {self.cfg.get('slippage_pts', 0)}pts (paper SL exits)")
        trail_desc = (f"{int(self.cfg['retracement_pct']*100)}% retracement"
                      if self.cfg['retracement_pct'] else "TP_TRAIL")
        print(f"  Lot2 floor: {trail_desc}")
        print(f"{'='*60}\n")

        # Load or train model (done early so it's ready by 9:15)
        if os.path.exists(self.engine.model_xgb) and os.path.exists(self.engine.model_file):
            try:
                self.engine.load()
                print(f"  5-min model loaded for {self.instrument}")
            except Exception as e:
                print(f"  Model load failed ({e}) — retraining...")
                self._retrain_model()
        else:
            print(f"  No 5-min model found — training from scratch...")
            self._retrain_model()

        # Restore state from disk if we crashed mid-trade
        self._load_state()
        self._init_margin()

        # Wait until 9:15 before alerting and entering the main loop
        self._wait_until_market_open()

        from econ_calendar import load_and_print
        load_and_print()

        self._send_alert(
            f"🤖 ML Runner started — {self.instrument.upper()} [{mode_label}]\n"
            f"Lots: {self.num_lots} | SL: {self.cfg['sl_pts']} | "
            f"TP: {self.cfg['tp_pts']} | TP2: {round(self.cfg['tp_pts'] * self.cfg['runner_ext'],1)}"
        )

    # ── Model Training ─────────────────────────────────────────────────────────

    def _retrain_model(self):
        print("  Training 5-min model from archived data...")
        csv_glob = f"data/options_log_1min_{self.instrument}_????????.csv"
        files = sorted(glob.glob(csv_glob))
        if not files:
            print(f"  ERROR: No archived CSVs found: {csv_glob}")
            sys.exit(1)

        files = files[-30:]  # last 30 days
        live_csv = self._today_csv()
        if os.path.exists(live_csv):
            files.append(live_csv)

        print(f"  Using {len(files)} files for training")

        old_freq = self.resampler.freq
        raw = self.resampler.load_csv(files)
        candles = self.resampler.resample_v2(raw)
        print(f"  5-min candles: {len(candles)}")

        candles, x_pts = build_targets(candles, instrument=self.instrument)
        candles = add_lag_features(candles)
        self.engine.x_points = x_pts
        self.engine.dataset = candles

        # Apply feedback weights if ledger has any outcomes
        fb_weights = None
        if not self._feedback._df.empty:
            ts_list   = candles.index.tolist()
            date_list = (candles["trading_date"].tolist()
                         if "trading_date" in candles.columns
                         else [""] * len(ts_list))
            fb_weights = self._feedback.get_sample_weights(ts_list, date_list)
            n_naughty = int((fb_weights >= 4.0).sum())
            n_nice    = int((fb_weights == 3.0).sum())
            print(f"  📊 Feedback weights: {n_naughty} NAUGHTY × {n_nice} NICE "
                  f"(from {len(self._feedback._df)} recorded outcomes)")

        self.engine.train(feedback_weights=fb_weights)
        print(f"  ✅ 5-min model trained and saved")

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def run(self):
        self.startup()

        while True:
            now = datetime.now()

            # Market hours check (9:15 - 15:30)
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                self._wait_until_next_minute()
                continue

            if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                self._on_market_close()
                break

            try:
                from econ_calendar import check_upcoming, rate_decision_imminent
                check_upcoming(self._warned_events)

                # Rate decision at T-5: exit position and block entries
                if rate_decision_imminent():
                    if self.state in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
                        print("  ⚠️  RBI rate decision imminent — exiting position")
                        self._send_alert("⚠️ RBI rate decision in <5min — exiting position, pausing entries")
                        ltp = get_ltp_safe(self.kite, self.trade["full_symbol"])
                        if self.state == TradeState.POSITION_OPEN:
                            self._exit_all_market(ltp or 0, "RBI_EVENT")
                        else:
                            self._exit_lot2_market(ltp or 0, "RBI_EVENT")
                    self._wait_until_next_minute()
                    continue

                # 1. If in position, monitor exits at 5-second intervals
                if self.state in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
                    self._monitor_position_fast()
                    continue

                # 2a. Waiting for spot confirmation trigger
                if self.state == TradeState.AWAITING_TRIGGER:
                    self._check_trigger()

                # 2b. Check for new ML signal at 5-min boundaries
                elif self.state in (TradeState.IDLE, TradeState.COOLDOWN):
                    self._check_signal()

            except Exception as e:
                print(f"  ERROR in main loop: {e}")
                traceback.print_exc()

            self._wait_until_next_minute()

    # ── Signal Detection ───────────────────────────────────────────────────────

    def _today_csv(self):
        date_str = datetime.now().strftime("%d%m%Y")
        return f"data/options_log_1min_{self.instrument}_{date_str}.csv"

    def _live_csv(self):
        return self.profile["csv_file"]

    def _next_candle_time(self):
        now = datetime.now()
        mins = now.minute
        next_min = ((mins // self.cfg["candle_minutes"]) + 1) * self.cfg["candle_minutes"]
        if next_min >= 60:
            return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return now.replace(minute=next_min, second=0, microsecond=0)

    def _check_signal(self):
        from econ_calendar import rate_decision_imminent, msci_close_block
        if rate_decision_imminent():
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Entries blocked — RBI rate decision imminent")
            return
        if msci_close_block():
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Entries blocked — MSCI rebalancing close window")
            return

        if self.day_suspended:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Day suspended — insufficient margin for minimum lots")
            return

        daily_limit = self.cfg.get("daily_loss_limit")
        if daily_limit and self.session_pnl <= -daily_limit:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Daily loss limit reached "
                  f"(session PnL Rs {self.session_pnl:,.0f}) — no new entries today")
            return

        csv_file = self._live_csv()
        if not os.path.exists(csv_file):
            print(f"  ⚠ CSV not found: {csv_file}")
            return

        try:
            raw = self.resampler.load_csv(csv_file)
            candles = self.resampler.resample_v2(raw)
        except Exception as e:
            print(f"  ⚠ Failed to load/resample {csv_file}: {e}")
            return

        # Exclude the current live candle — its window hasn't closed yet
        now = datetime.now()
        closed_until = pd.Timestamp(now).floor(f"{self.cfg['candle_minutes']}min")
        candles = candles[candles.index < closed_until]

        min_candles = 4  # lags=3 → need lags+1 rows to survive dropna
        if len(candles) < min_candles:
            print(f"  ⚠ Not enough candles yet ({len(candles)}/{min_candles})")
            return

        # Only act on a NEW candle

        if len(candles) == self.last_candle_count:
            next_candle = self._next_candle_time()
            print(f"  [{now.strftime('%H:%M:%S')}] Waiting — {len(candles)} candles, "
                  f"next check at {next_candle.strftime('%H:%M')}")
            return
        self.last_candle_count = len(candles)

        # Cooldown check
        if self.state == TradeState.COOLDOWN:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining > 0:
                return
            self.state = TradeState.IDLE
            print(f"  Cooldown ended, ready for new signals")

        # Add lag features and predict on latest candle
        candles = add_lag_features(candles)
        if candles.empty:
            print(f"  ⚠ No candles after lag feature computation")
            return
        latest = candles.iloc[-1].to_dict()
        result = self.engine.predict_latest(latest)

        now = datetime.now()
        direction = "LONG" if result["signal"] == 1 else ("SHORT" if result["signal"] == -1 else "FLAT")
        confidence = result["confidence"]
        print(f"  [{now.strftime('%H:%M:%S')}] Candle {len(candles)} → {direction} conf={confidence:.2f}")

        if result["signal"] == 0:
            return

        # Hybrid confidence filter
        min_conf = (self.cfg["min_conf_long"] if direction == "LONG"
                    else self.cfg["min_conf_short"])
        if confidence < min_conf:
            print(f"  Signal {direction} conf={confidence:.2f} below threshold {min_conf}")
            return

        # Don't enter after 15:00 — not enough hold time
        now = datetime.now()
        if now.hour >= 15:
            print(f"  Signal {direction} skipped — too close to market close")
            return

        # Strong-trend counter-trend gate: block signals opposing bias when trend > 300pts
        trend_pts   = latest.get("trend_pts", 0)
        bias_enc    = latest.get("bias_encoded", 0)
        ct_threshold = self.cfg.get("counter_trend_pts_limit", 300)
        if trend_pts > ct_threshold and bias_enc != 0:
            is_counter = (direction == "SHORT" and bias_enc == 1) or \
                         (direction == "LONG"  and bias_enc == -1)
            if is_counter:
                print(f"  Signal {direction} blocked — counter-trend in {trend_pts:.0f}pt trend")
                return

        # Session displacement gate: block LONGs when session is grinding lower.
        # Fires only when spot is BOTH (a) >N pts below session high AND (b) below session open.
        # Condition (b) makes it regime-aware: pullback longs in a green session are not blocked.
        long_block_pts = self.cfg.get("session_disp_long_block_pts")
        if long_block_pts and direction == "LONG" and "spot_close" in candles.columns:
            spot_series    = candles["spot_close"]
            session_open_s = float(spot_series.iloc[0])
            session_high_s = float(spot_series.max())
            current_spot   = float(spot_series.iloc[-1])
            drop_from_high = session_high_s - current_spot
            if drop_from_high > long_block_pts and current_spot < session_open_s:
                print(f"  Signal LONG blocked — {drop_from_high:.0f}pts below session high, "
                      f"{session_open_s - current_spot:.0f}pts below session open")
                return

        print(f"\n  📡 ML SIGNAL: {direction} conf={confidence:.2f}")

        if self.cfg.get("skip_expiry_day"):
            expiry = self._get_nearest_expiry()
            if expiry == date.today():
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Entry blocked — expiry day ({expiry}), signal logged only")
                return

        # Track signal candle timestamp for feedback recording at trade close
        self._pending_signal_ts = candles.index[-1]
        self._pending_signal_dir = result["signal"]
        self._pending_confidence = confidence

        trigger_pts = self.cfg.get("spot_trigger_pts")
        trigger_trend_limit = self.cfg.get("spot_trigger_trend_limit")
        # Skip trigger when market is already trending hard — chasing a running move is worse
        use_trigger = (trigger_pts is not None and
                       (trigger_trend_limit is None or abs(trend_pts) < trigger_trend_limit))
        if use_trigger:
            spot = self._get_spot()
            if spot is None:
                print("  Cannot get spot — entering immediately")
                self._enter_trade(direction, confidence)
                return
            window_mins = 2 * self.cfg["candle_minutes"]   # FORWARD_CANDLES × 5
            self.pending_signal  = {"direction": direction, "confidence": confidence, "spot": spot}
            self.trigger_expiry  = datetime.now() + timedelta(minutes=window_mins)
            self.state           = TradeState.AWAITING_TRIGGER
            print(f"  ⏳ Awaiting spot trigger: {direction} needs spot to move "
                  f"{trigger_pts}pts | window={window_mins}min | spot={spot} "
                  f"(trend_pts={trend_pts:.0f} < limit={trigger_trend_limit})")
            self._send_alert(f"⏳ Signal {direction} conf={confidence:.2f} | "
                             f"Waiting for spot +{trigger_pts}pts @ {spot}")
        else:
            if trigger_pts and trigger_trend_limit and abs(trend_pts) >= trigger_trend_limit:
                print(f"  ⚡ Entering immediately — trending {trend_pts:.0f}pts (skip trigger)")
            self._enter_trade(direction, confidence)

    # ── Spot Trigger Check ─────────────────────────────────────────────────────

    def _check_trigger(self):
        now     = datetime.now()
        pending = self.pending_signal
        trigger_pts = self.cfg.get("spot_trigger_pts", 20)

        if now >= self.trigger_expiry:
            print(f"  [{now.strftime('%H:%M:%S')}] ⌛ Trigger window expired — signal abandoned")
            self._send_alert(f"⌛ Trigger expired — {pending['direction']} signal abandoned")
            # Record as wrong: model was confident but spot never confirmed direction
            if self._pending_signal_ts is not None:
                try:
                    self._feedback.record_trade_outcome(
                        candle_ts=self._pending_signal_ts,
                        signal=self._pending_signal_dir,
                        confidence=self._pending_confidence,
                        exit_type="TRIGGER_EXPIRED",
                        outcome="wrong",
                        trading_date=str(self._pending_signal_ts.date()),
                        entry_ltp=None,
                    )
                except Exception as e:
                    print(f"  ⚠ Feedback record (abandoned) failed: {e}")
                self._pending_signal_ts = None
            self.state          = TradeState.IDLE
            self.pending_signal = None
            return

        spot = self._get_spot()
        if spot is None:
            return

        signal_spot = pending["spot"]
        direction   = pending["direction"]
        move = (spot - signal_spot) * (1 if direction == "LONG" else -1)
        remaining   = int((self.trigger_expiry - now).total_seconds() / 60)
        print(f"  [{now.strftime('%H:%M:%S')}] ⏳ {direction} trigger: "
              f"spot moved {move:+.0f}/{trigger_pts}pts | {remaining}min left")

        if move >= trigger_pts:
            print(f"  ✅ Spot trigger fired! {direction} +{move:.0f}pts")
            self._send_alert(f"✅ Trigger fired! {direction} spot +{move:.0f}pts — entering")
            self.state          = TradeState.IDLE
            self.pending_signal = None
            self._enter_trade(direction, pending["confidence"])

    # ── Trade Entry ────────────────────────────────────────────────────────────

    def _enter_trade(self, direction, confidence):
        spot = self._get_spot()
        if spot is None:
            print("  Cannot get spot price — skipping entry")
            return

        option_type = "CE" if direction == "LONG" else "PE"
        expiry = self._get_nearest_expiry()
        if expiry is None:
            print("  Cannot get expiry — skipping entry")
            return

        # Select 0.4 delta strike — returns LTP from the bulk quote, no extra fetch needed
        strike, trading_sym, option_ltp = self._select_strike(spot, option_type, expiry)
        if strike is None:
            print("  Cannot find target delta strike — skipping entry")
            return

        full_symbol = f"{self.exchange}:{trading_sym}"

        # Lot-stepping: try configured lots, step down by 2 if margin is insufficient.
        # Start from the nearest even number at or below configured lots so the
        # sequence always lands on even values and reaches 2 regardless of config.
        lot_size    = self.cfg["lot_size"]
        start_lots  = self.num_lots if self.num_lots % 2 == 0 else self.num_lots - 1
        actual_lots = None
        for try_lots in range(start_lots, 1, -2):
            # Cost = premium for all lots + brokerage (entry order) + taxes
            cost = option_ltp * lot_size * try_lots + 20 + 12 * try_lots
            if self.available_margin >= cost:
                actual_lots = try_lots
                break

        if actual_lots is None:
            min_cost = option_ltp * lot_size * 2 + 20 + 24
            print(f"  Margin too low for 2 lots — need Rs {min_cost:,.0f}, "
                  f"have Rs {self.available_margin:,.0f} — suspending for the day")
            self._send_alert(
                f"🚨 Margin insufficient for 2 lots (have Rs {self.available_margin:,.0f}, "
                f"need Rs {min_cost:,.0f}) — no more entries today"
            )
            self.day_suspended = True
            self._save_state()
            return

        if actual_lots != self.num_lots:
            print(f"  ⚠ Margin reduced: {actual_lots} lots (configured {self.num_lots}) "
                  f"— margin Rs {self.available_margin:,.0f}")
            self._send_alert(
                f"⚠️ Low margin — entering {actual_lots} lots (configured: {self.num_lots})\n"
                f"Available: Rs {self.available_margin:,.0f}"
            )

        qty_per_side = lot_size * max(1, actual_lots // 2)
        total_qty    = qty_per_side * 2

        # Place ENTRY order
        entry_order_id = self._place_order(
            exchange=self.exchange,
            tradingsymbol=trading_sym,
            transaction_type=self.kite.TRANSACTION_TYPE_BUY,
            quantity=total_qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
        )
        if entry_order_id is None:
            self._send_alert(f"❌ Entry order FAILED for {trading_sym} — suspending for the day")
            # Drive session_pnl past the daily limit so _check_signal blocks further entries
            daily_limit = self.cfg.get("daily_loss_limit", 0)
            self.session_pnl = -(daily_limit + 1) if daily_limit else self.session_pnl
            self._save_state()
            return

        # Poll for fill price — exits as soon as Kite confirms, hard cap at 1s
        entry_price = None
        if not self.paper:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                entry_price = self._get_fill_price(entry_order_id)
                if entry_price:
                    break
                time.sleep(0.1)
        if not entry_price:
            entry_price = option_ltp  # fallback to pre-trade LTP from bulk quote

        # Compute SL trigger price
        sl_trigger = round(entry_price - self.cfg["sl_pts"], 2)
        sl_trigger = max(sl_trigger, 0.05)

        # Place SL-M order IMMEDIATELY
        sl_order_id = self._place_order(
            exchange=self.exchange,
            tradingsymbol=trading_sym,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=total_qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_SLM,
            trigger_price=sl_trigger,
        )

        if sl_order_id is None:
            # CRITICAL: SL could not be placed — exit immediately
            self._send_alert(
                f"🚨 SL ORDER FAILED — exiting position immediately!\n"
                f"{trading_sym} qty={total_qty}"
            )
            self._place_order(
                exchange=self.exchange,
                tradingsymbol=trading_sym,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=total_qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
            )
            return

        # Record trade state
        tp_price = round(entry_price + self.cfg["tp_pts"], 2)
        tp2_price = round(entry_price + self.cfg["tp_pts"] * self.cfg["runner_ext"], 2)

        self.trade = {
            "direction":        direction,
            "confidence":       confidence,
            "option_type":      option_type,
            "strike":           strike,
            "trading_sym":      trading_sym,
            "full_symbol":      full_symbol,
            "entry_price":      entry_price,
            "entry_spot":       spot,
            "entry_time":       datetime.now().isoformat(),
            "sl_trigger":       sl_trigger,
            "tp_price":         tp_price,
            "tp2_price":        tp2_price,
            "total_qty":        total_qty,
            "qty_per_side":     qty_per_side,
            "entry_order_id":   entry_order_id,
            "sl_order_id":      sl_order_id,
            "candles_held":     0,
            "lot2_peak_ltp":    0.0,
            "sl_adjusted":      False,
            "actual_lots":      actual_lots,
        }

        self.state = TradeState.POSITION_OPEN
        self._save_state()

        self._send_alert(
            f"📡 {direction} conf={confidence:.2f} | spot={spot}\n"
            f"✅ ENTRY: {total_qty} qty {trading_sym} @ {entry_price}\n"
            f"SL-M @ {sl_trigger} | TP1: {tp_price} | TP2: {tp2_price}"
        )
        print(f"  ✅ Entry: {trading_sym} @ {entry_price}, SL-M @ {sl_trigger}")

        self._log_trade("ENTRY", entry_price, total_qty,
                        notes=f"conf={confidence:.2f}")

    # ── Position Monitor ───────────────────────────────────────────────────────

    def _monitor_position_fast(self):
        """Poll position every 5 seconds until closed or next minute boundary."""
        next_min = datetime.now().replace(second=2, microsecond=0) + timedelta(minutes=1)
        while datetime.now() < next_min:
            if self.state not in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
                break
            self._monitor_position()
            if self.state not in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
                break
            time.sleep(5)

    def _monitor_position(self):
        t = self.trade
        ltp = get_ltp_safe(self.kite, t["full_symbol"])
        if ltp is None:
            print(f"  ⚠ Could not fetch LTP for {t['full_symbol']} — skipping monitor tick")
            return

        now = datetime.now()
        entry_time = datetime.fromisoformat(t["entry_time"])
        minutes_held = (now - entry_time).total_seconds() / 60
        candles_held = int(minutes_held // self.cfg["candle_minutes"])
        t["candles_held"] = candles_held

        if self.state == TradeState.POSITION_OPEN:
            self._monitor_both_lots(ltp, candles_held)
        elif self.state == TradeState.LOT1_EXITED:
            self._monitor_lot2(ltp, candles_held)

    def _monitor_both_lots(self, ltp, candles_held):
        t = self.trade

        # SL adjustment — once option gains trigger_pts from entry, move SL up
        if not t.get("sl_adjusted") and \
                ltp >= t["entry_price"] + self.cfg.get("sl_adjust_trigger_pts", 15):
            new_sl = round(t["sl_trigger"] + self.cfg.get("sl_adjust_move_pts", 10), 2)
            if self._modify_sl_trigger(t["sl_order_id"], new_sl):
                old_sl = t["sl_trigger"]
                t["sl_trigger"]  = new_sl
                t["sl_adjusted"] = True
                self._save_state()
                print(f"  SL adjusted: {old_sl} → {new_sl} (option up {self.cfg.get('sl_adjust_trigger_pts',15)}pts)")
                self._send_alert(
                    f"🔒 SL moved: {old_sl} → {new_sl} | {t['trading_sym']}"
                )

        # Check if SL-M was triggered (order status / paper LTP check)
        if self._is_sl_triggered(ltp):
            sl_exit = round(t["sl_trigger"] - self.cfg.get("slippage_pts", 0), 2) if self.paper else t["sl_trigger"]
            pnl = round((sl_exit - t["entry_price"]) * t["total_qty"])
            self._send_alert(
                f"🔴 SL HIT: {t['trading_sym']}\n"
                f"Entry: {t['entry_price']} → Exit: {sl_exit}\n"
                f"P&L: Rs {pnl:+,}"
            )
            self._log_trade("SL_HIT", sl_exit, t["total_qty"], pnl=pnl)
            self._close_trade(candles_held, pnl=pnl, exit_type="SL")
            return

        # Check TP hit (option LTP ≥ TP price)
        if ltp >= t["tp_price"]:
            print(f"  TP HIT for lot1! LTP={ltp} ≥ TP={t['tp_price']}")
            self._exit_lot1(ltp)
            return

        # TIME exit
        if candles_held >= self.cfg["hold_candles"]:
            print(f"  TIME EXIT: {candles_held} candles held")
            self._exit_all_market(ltp, "TIME")
            return

    def _monitor_lot2(self, ltp, candles_held):
        t = self.trade

        # Check if lot2 SL was triggered
        if self._is_sl_triggered(ltp):
            sl_price = t.get("lot2_sl_trigger", t["tp_price"])
            if self.paper:
                sl_price = round(sl_price - self.cfg.get("slippage_pts", 0), 2)
            pnl_lot1 = t.get("lot1_pnl", 0)
            pnl_lot2 = round((sl_price - t["entry_price"]) * t["qty_per_side"])
            total_pnl = pnl_lot1 + pnl_lot2
            self._send_alert(
                f"🔴 Lot2 SL/Trail triggered: {t['trading_sym']}\n"
                f"Lot1 P&L: Rs {pnl_lot1:+,} | Lot2 P&L: Rs {pnl_lot2:+,}\n"
                f"Total: Rs {total_pnl:+,}"
            )
            self._log_trade("LOT2_TRAIL", sl_price, t["qty_per_side"], pnl=total_pnl)
            self._close_trade(candles_held, pnl=total_pnl, exit_type="TRAIL")
            return

        # Update lot2 peak
        if ltp > t["lot2_peak_ltp"]:
            t["lot2_peak_ltp"] = ltp
            # Update trailing SL if retracement mode
            if self.cfg["retracement_pct"]:
                self._update_lot2_trail(ltp)

        # Check TP2 hit — only when no retracement trail is configured.
        # When trailing is active, TP2 is a minimum target reference only;
        # the trail handles the actual exit so we don't cap the runner early.
        if not self.cfg["retracement_pct"] and ltp >= t["tp2_price"]:
            print(f"  TP2 HIT! LTP={ltp} ≥ TP2={t['tp2_price']}")
            self._exit_lot2_market(ltp, "TP2")
            return

        # TIME exit for lot2
        if candles_held >= self.cfg["hold_candles"]:
            print(f"  TIME EXIT lot2: {candles_held} candles held")
            self._exit_lot2_market(ltp, "TIME")
            return

    # ── Lot1 Exit ──────────────────────────────────────────────────────────────

    def _exit_lot1(self, current_ltp):
        t = self.trade

        # Cancel full-qty SL-M
        self._cancel_order(t["sl_order_id"])

        # Sell lot1 at market
        lot1_order = self._place_order(
            exchange=self.exchange,
            tradingsymbol=t["trading_sym"],
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=t["qty_per_side"],
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
        )

        lot1_exit_price = current_ltp
        if lot1_order and not self.paper:
            deadline = time.time() + 0.5
            while time.time() < deadline:
                fill = self._get_fill_price(lot1_order)
                if fill:
                    lot1_exit_price = fill
                    break
                time.sleep(0.1)

        lot1_pnl = round((lot1_exit_price - t["entry_price"]) * t["qty_per_side"])
        t["lot1_pnl"] = lot1_pnl
        t["lot1_exit_price"] = lot1_exit_price

        # Place new SL-M for lot2 at entry + floor buffer
        lot2_sl_trigger = round(t["entry_price"] + self.cfg["lot2_floor_pts"], 2)
        t["lot2_sl_trigger"] = lot2_sl_trigger
        t["lot2_peak_ltp"] = current_ltp

        lot2_sl_order = self._place_order(
            exchange=self.exchange,
            tradingsymbol=t["trading_sym"],
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=t["qty_per_side"],
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_SLM,
            trigger_price=lot2_sl_trigger,
        )

        if lot2_sl_order is None:
            self._send_alert(f"⚠️ Lot2 SL FAILED — exiting lot2 immediately")
            self._exit_lot2_market(current_ltp, "SL_FAIL")
            return

        t["sl_order_id"] = lot2_sl_order
        self.state = TradeState.LOT1_EXITED
        self._save_state()

        self._send_alert(
            f"💰 LOT1 TP HIT: Sold {t['qty_per_side']} qty @ {lot1_exit_price}\n"
            f"Lot1 P&L: Rs {lot1_pnl:+,}\n"
            f"Lot2 running — SL-M @ {lot2_sl_trigger} | TP2 target: {t['tp2_price']}"
        )

        self._log_trade("LOT1_TP", lot1_exit_price, t["qty_per_side"],
                        pnl=lot1_pnl, notes="lot2 running")

    # ── Lot2 Trail Update ──────────────────────────────────────────────────────

    def _update_lot2_trail(self, current_ltp):
        t = self.trade
        pct = self.cfg["retracement_pct"]
        peak = t["lot2_peak_ltp"]

        # Trail floor = max(entry + floor buffer, peak × (1 - retracement%))
        trail_price = peak * (1.0 - pct)
        floor = t["entry_price"] + self.cfg["lot2_floor_pts"]
        new_sl = round(max(floor, trail_price), 2)

        old_sl = t.get("lot2_sl_trigger", t["tp_price"])
        min_move = self.cfg.get("trail_update_pts", 2)
        if new_sl > old_sl + min_move:
            if self._modify_sl_trigger(t["sl_order_id"], new_sl):
                t["lot2_sl_trigger"] = new_sl
                self._save_state()
                print(f"  Trail updated: SL-M moved to {new_sl} (peak={peak:.1f})")

    # ── Exit Helpers ───────────────────────────────────────────────────────────

    def _exit_all_market(self, current_ltp, reason):
        t = self.trade

        # Cancel SL-M
        self._cancel_order(t["sl_order_id"])

        # Sell all at market
        self._place_order(
            exchange=self.exchange,
            tradingsymbol=t["trading_sym"],
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=t["total_qty"],
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
        )

        pnl = round((current_ltp - t["entry_price"]) * t["total_qty"])
        self._send_alert(
            f"⏱️ {reason} EXIT: Sold {t['total_qty']} qty {t['trading_sym']} @ ~{current_ltp}\n"
            f"Entry: {t['entry_price']} | P&L: Rs {pnl:+,}"
        )
        self._log_trade(f"{reason}_EXIT", current_ltp, t["total_qty"], pnl=pnl)
        self._close_trade(t["candles_held"], pnl=pnl, exit_type=reason)

    def _exit_lot2_market(self, current_ltp, reason):
        t = self.trade

        # Cancel SL-M
        self._cancel_order(t["sl_order_id"])

        # Sell lot2 at market
        order = self._place_order(
            exchange=self.exchange,
            tradingsymbol=t["trading_sym"],
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=t["qty_per_side"],
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_MARKET,
        )

        exit_price = current_ltp
        if order and not self.paper:
            time.sleep(0.5)
            fill = self._get_fill_price(order)
            if fill:
                exit_price = fill

        lot2_pnl = round((exit_price - t["entry_price"]) * t["qty_per_side"])
        lot1_pnl = t.get("lot1_pnl", 0)
        total_pnl = lot1_pnl + lot2_pnl

        emoji = {"TP2": "🎯", "TIME": "⏱️", "SL_FAIL": "⚠️"}.get(reason, "📊")
        self._send_alert(
            f"{emoji} LOT2 {reason}: Sold {t['qty_per_side']} qty @ {exit_price}\n"
            f"Peak LTP: {t['lot2_peak_ltp']:.1f}\n"
            f"Lot1: Rs {lot1_pnl:+,} | Lot2: Rs {lot2_pnl:+,}\n"
            f"Total: Rs {total_pnl:+,}"
        )
        self._log_trade(f"LOT2_{reason}", exit_price, t["qty_per_side"],
                        pnl=total_pnl, notes=f"peak={t['lot2_peak_ltp']:.1f}")
        self._close_trade(t["candles_held"], pnl=total_pnl, exit_type=reason)

    def _close_trade(self, candles_held, pnl=0, exit_type=""):
        self.completed_trades.append({
            "time": datetime.now().strftime("%H:%M"),
            "symbol": self.trade.get("trading_sym", ""),
            "direction": self.trade.get("direction", ""),
            "entry": self.trade.get("entry_price", 0),
            "pnl": pnl,
            "exit_type": exit_type,
        })
        self.session_pnl += pnl

        # Deduct brokerage + taxes from tracked margin.
        # SL hit = 2 executed orders (entry + SL-M). All other exits = 3 (entry + lot1 + lot2).
        actual_lots = self.trade.get("actual_lots", self.num_lots)
        num_orders  = 2 if exit_type == "SL" else 3
        charges     = 20 * num_orders + 12 * actual_lots
        self.available_margin += pnl - charges

        # Record outcome to feedback ledger for next EOD retrain
        if self._pending_signal_ts is not None:
            outcome = ("correct" if exit_type in ("TP", "TP2", "TRAIL") and pnl > 0
                       else "wrong" if exit_type == "SL"
                       else "neutral")
            try:
                self._feedback.record_trade_outcome(
                    candle_ts=self._pending_signal_ts,
                    signal=self._pending_signal_dir,
                    confidence=self._pending_confidence,
                    exit_type=exit_type,
                    outcome=outcome,
                    trading_date=str(self._pending_signal_ts.date()),
                    entry_ltp=self.trade.get("entry_price"),
                )
            except Exception as e:
                print(f"  ⚠ Feedback record failed: {e}")
            self._pending_signal_ts = None

        self.cooldown_remaining = max(1, candles_held)

        if exit_type == "SL":
            self.consecutive_sl += 1
            cb = self.cfg.get("sl_circuit_breaker")
            if cb and self.consecutive_sl >= cb:
                pause = self.cfg.get("circuit_pause_candles", 2)
                self.cooldown_remaining = max(self.cooldown_remaining, pause)
                print(f"  ⚡ Circuit breaker: {self.consecutive_sl} consecutive SLs — pausing {self.cooldown_remaining} candles")
                self.consecutive_sl = 0
        else:
            self.consecutive_sl = 0

        self.state = TradeState.COOLDOWN
        self.trade = {}
        self._save_state()
        print(f"  Trade closed, cooldown={self.cooldown_remaining} candles")

    # ── SL Order Status ────────────────────────────────────────────────────────

    def _is_sl_triggered(self, ltp=None):
        sl = self.trade.get("lot2_sl_trigger", self.trade.get("sl_trigger"))
        return ltp is not None and sl is not None and ltp <= sl

    # ── Fill Price ─────────────────────────────────────────────────────────────

    def _get_fill_price(self, order_id):
        try:
            history = self.kite.order_history(order_id)
            latest = history[-1]
            if latest["status"] == "COMPLETE":
                return latest["average_price"]
        except Exception:
            pass
        return None

    # ── Strike Selection ───────────────────────────────────────────────────────

    def _select_strike(self, spot, option_type, expiry):
        instruments = self._get_instruments()
        T = max((expiry - date.today()).days / 365.0, 1 / 365.0)
        r = RISK_FREE_RATE

        # Collect trading symbols for all strikes near spot
        atm = round(spot / self.strike_step) * self.strike_step
        strikes_range = [atm + i * self.strike_step for i in range(-10, 11)]

        sym_map = {}  # full_symbol → strike
        for s in strikes_range:
            tsym = self._find_trading_symbol(instruments, s, option_type, expiry)
            if not tsym:
                print(f"    Skip {s} {option_type}: no trading symbol")
                continue
            sym_map[f"{self.exchange}:{tsym}"] = (s, tsym)

        if not sym_map:
            print(f"  ⚠ No trading symbols found for {option_type}")
            return None, None

        # Fetch all LTPs in one bulk quote call
        try:
            quotes = self.kite.quote(list(sym_map.keys()))
        except Exception as e:
            print(f"  ⚠ Bulk quote failed: {e}")
            return None, None

        # Build candidates from bulk quote result
        candidates = []
        for full_sym, (strike, tsym) in sym_map.items():
            ltp = quotes.get(full_sym, {}).get("last_price")
            if not ltp or ltp <= 0:
                print(f"    Skip {strike} {option_type}: LTP missing in quote response")
                continue

            iv = implied_vol(spot, strike, T, r, ltp, option_type)
            if iv is None or iv <= 0:
                print(f"    Skip {strike} {option_type}: IV solver failed (ltp={ltp})")
                continue

            delta = bs_delta(spot, strike, T, r, iv, option_type)
            candidates.append({
                "strike":     strike,
                "trading_sym": tsym,
                "ltp":        ltp,
                "delta":      delta,
                "abs_delta":  abs(delta),
            })

        if not candidates:
            print(f"  ⚠ No valid candidates found for {option_type} (all failed LTP/IV)")
            return None, None

        # Log all surviving candidates
        for c in sorted(candidates, key=lambda x: x["strike"]):
            print(f"    Strike {c['strike']} {option_type}: delta={c['delta']:.3f} ltp={c['ltp']}")

        # Pick closest to target delta
        target = self.cfg["target_delta"]
        best = min(candidates, key=lambda c: abs(c["abs_delta"] - target))

        # Sanity check — reject if no candidate is within 0.15 of target
        if abs(best["abs_delta"] - target) > 0.15:
            print(f"  ⚠ Best candidate strike {best['strike']} has delta={best['delta']:.3f}, "
                  f"too far from target {target} — skipping entry")
            return None, None

        print(f"  Selected strike {best['strike']} {option_type} "
              f"delta={best['delta']:.3f} ltp={best['ltp']}")
        return best["strike"], best["trading_sym"], best["ltp"]

    def _find_trading_symbol(self, instruments, strike, option_type, expiry):
        for ins in instruments:
            if (ins["name"] == self.inst_name
                    and ins["expiry"] == expiry
                    and ins["strike"] == strike
                    and ins["instrument_type"] == option_type):
                return ins["tradingsymbol"]
        return None

    def _get_instruments(self):
        today = date.today()
        if self._instruments_cache and self._instruments_date == today:
            return self._instruments_cache
        self._instruments_cache = self.kite.instruments(self.exchange)
        self._instruments_date = today
        return self._instruments_cache

    def _get_nearest_expiry(self):
        instruments = self._get_instruments()
        today = date.today()
        expiries = sorted(set(
            i["expiry"] for i in instruments
            if i["name"] == self.inst_name
            and i["instrument_type"] == "CE"
            and i["expiry"] >= today
        ))
        return expiries[0] if expiries else None

    def _get_spot(self):
        try:
            quote = self.kite.quote(self.spot_symbol)
            return quote[self.spot_symbol]["last_price"]
        except Exception as e:
            print(f"  Spot fetch failed: {e}")
            return None

    # ── State Persistence ──────────────────────────────────────────────────────

    def _init_margin(self):
        """Fetch available margin at session start. Live: from Kite. Paper: from config."""
        if not self.paper:
            try:
                m = self.kite.margins()
                self.available_margin = float(
                    m.get("equity", {}).get("available", {}).get("live_balance", 0)
                )
                print(f"  Live margin: Rs {self.available_margin:,.0f}")
            except Exception as e:
                print(f"  ⚠ Could not fetch margin ({e}) — margin check disabled")
                self.available_margin = float("inf")
        else:
            paper_margin = self.cfg.get("paper_starting_margin")
            if paper_margin and self.available_margin == 0.0:
                # Only set from config on a fresh start; preserve restored value on restarts
                self.available_margin = float(paper_margin)
            print(f"  Paper margin: Rs {self.available_margin:,.0f}")

    def _save_state(self):
        data = {
            "state": self.state.value,
            "trade": self.trade,
            "cooldown_remaining": self.cooldown_remaining,
            "last_candle_count": self.last_candle_count,
            "consecutive_sl": self.consecutive_sl,
            "session_pnl":       self.session_pnl,
            "available_margin":  self.available_margin,
            "day_suspended":     self.day_suspended,
            "saved_at":          datetime.now().isoformat(),
        }
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            saved_state = data.get("state", "IDLE")
            if saved_state in ("POSITION_OPEN", "LOT1_EXITED"):
                self.state = TradeState(saved_state)
                self.trade = data.get("trade", {})
                self.cooldown_remaining = data.get("cooldown_remaining", 0)
                self.last_candle_count = data.get("last_candle_count", 0)

                # Reconcile with Kite — check if position still exists
                print(f"  ⚠ Recovered state: {saved_state}")
                print(f"  Trade: {self.trade.get('trading_sym')} "
                      f"entry={self.trade.get('entry_price')}")

                self._send_alert(
                    f"⚠️ Runner restarted — recovering position\n"
                    f"State: {saved_state}\n"
                    f"Symbol: {self.trade.get('trading_sym')}\n"
                    f"Entry: {self.trade.get('entry_price')}"
                )
            else:
                self.state = TradeState(saved_state)
                self.cooldown_remaining = data.get("cooldown_remaining", 0)
                self.last_candle_count = data.get("last_candle_count", 0)
                self.consecutive_sl = data.get("consecutive_sl", 0)

            # Restore session PnL and paper margin only if state file is from today
            saved_date = data.get("saved_at", "")[:10]
            if saved_date == date.today().isoformat():
                self.session_pnl    = data.get("session_pnl", 0.0)
                self.day_suspended  = data.get("day_suspended", False)
                # Live margin is always re-fetched from Kite in _init_margin;
                # paper margin must be restored from state so losses carry across restarts.
                if self.paper:
                    self.available_margin = data.get("available_margin", 0.0)
            else:
                self.session_pnl   = 0.0
                self.day_suspended = False
                # available_margin reset handled by _init_margin on next startup

        except Exception as e:
            print(f"  State recovery failed: {e}")
            self.state = TradeState.IDLE

    # ── EOD Handling ───────────────────────────────────────────────────────────

    def _on_market_close(self):
        # If still in position, exit at market
        if self.state in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
            ltp = get_ltp_safe(self.kite, self.trade["full_symbol"])
            if self.state == TradeState.POSITION_OPEN:
                self._exit_all_market(ltp or 0, "EOD")
            else:
                self._exit_lot2_market(ltp or 0, "EOD")

        # Send EOD summary
        self._send_eod_summary()

        # Archive trade log on expiry day
        self._archive_trade_log_on_expiry()

        # Refresh feedback ledger from full historical options log
        print("\n  EOD — refreshing feedback ledger...")
        try:
            from backtest_feedback_loop import build_feedback_ledger
            build_feedback_ledger(
                instrument=self.instrument,
                candle_minutes=self.cfg["candle_minutes"],
                verbose=True,
            )
            # Reload the refreshed ledger so _retrain_model picks up new weights
            self._feedback = FeedbackLedger(
                path=f"data/ml_feedback_{self.instrument}.csv"
            )
        except Exception as e:
            print(f"  Feedback refresh failed: {e} — retraining without feedback")

        # Retrain model for tomorrow (applies refreshed feedback weights)
        print("\n  EOD — retraining 5-min model...")
        try:
            self._retrain_model()
        except Exception as e:
            print(f"  Retrain failed: {e}")

        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def _archive_trade_log_on_expiry(self):
        if not os.path.exists(self._trade_log_file):
            return
        try:
            expiry = self._get_nearest_expiry()
        except Exception:
            expiry = None
        if expiry != date.today():
            return
        date_str = date.today().strftime("%d%m%Y")
        archive = self._trade_log_file.replace(".csv", f"_{date_str}.csv")
        os.replace(self._trade_log_file, archive)
        print(f"  📁 Trade log archived: {archive}")

    def _send_eod_summary(self):
        trades = self.completed_trades
        mode_label = "PAPER" if self.paper else "LIVE"
        header = f"📊 EOD Summary — {self.instrument.upper()} [{mode_label}]"

        if not trades:
            self._send_alert(f"{header}\nNo trades today.")
            return

        total_pnl = sum(t["pnl"] for t in trades)
        winners = sum(1 for t in trades if t["pnl"] > 0)
        losers = sum(1 for t in trades if t["pnl"] <= 0)

        exit_counts = {}
        for t in trades:
            ex = t["exit_type"]
            exit_counts[ex] = exit_counts.get(ex, 0) + 1
        exit_str = "  ".join(f"{k}={v}" for k, v in sorted(exit_counts.items()))

        lines = [header, ""]
        for i, t in enumerate(trades, 1):
            emoji = "🟢" if t["pnl"] > 0 else "🔴"
            lines.append(
                f"{emoji} {t['time']} {t['direction']:5s} "
                f"{t['symbol']}  Rs {t['pnl']:+,}  ({t['exit_type']})"
            )

        lines.append("")
        lines.append(f"Trades: {len(trades)} | W: {winners} | L: {losers}")
        lines.append(f"Exits: {exit_str}")
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(f"{pnl_emoji} Net P&L: Rs {total_pnl:+,}")

        self._send_alert("\n".join(lines))

    # ── Timing ─────────────────────────────────────────────────────────────────

    def _wait_until_market_open(self):
        now = datetime.now()
        target = now.replace(hour=9, minute=15, second=0, microsecond=0)
        wait_secs = (target - now).total_seconds()
        if wait_secs > 0:
            print(f"  Pre-market: waiting {int(wait_secs)}s until 9:15...")
            time.sleep(wait_secs)

    def _wait_until_next_minute(self):
        now = datetime.now()
        next_min = now.replace(second=2, microsecond=0) + timedelta(minutes=1)
        sleep_secs = (next_min - now).total_seconds()
        if sleep_secs > 0:
            time.sleep(sleep_secs)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Live Trading Runner")
    parser.add_argument("--instrument", required=True, choices=["nifty", "sensex"])
    parser.add_argument("--lots", type=int, default=None,
                        help="Number of lots (default: from config)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (default: paper)")
    args = parser.parse_args()

    instrument = args.instrument.lower()
    cfg = RUNNER_CONFIGS[instrument]
    num_lots = args.lots if args.lots else cfg["default_lots"]
    paper = args.mode == "paper"

    kite = get_kite_client()
    runner = MLRunner(instrument, num_lots, kite, paper=paper)
    runner.run()


if __name__ == "__main__":
    main()
