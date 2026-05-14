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
from notifier import send_telegram_message
from ml_engine import (
    MLEngine, Resampler, build_targets, add_lag_features,
    PROBA_THRESHOLD, FORWARD_CANDLES,
)
from gamma_engine import _bs_d1, implied_vol
from config import INSTRUMENT_PROFILES, RISK_FREE_RATE


# =============================================================================
# INSTRUMENT CONFIGS (from calibrated backtest results)
# =============================================================================

RUNNER_CONFIGS = {
    "sensex": {
        "sl_pts":           15,
        "tp_pts":           25,
        "runner_ext":       2.16,
        "retracement_pct":  0.10,
        "min_conf_short":   0.55,
        "min_conf_long":    0.75,
        "lot_size":         20,
        "default_lots":     6,
        "hold_candles":     9,         # 45 min / 5 min
        "target_delta":     0.30,
        "candle_minutes":   5,
    },
    "nifty": {
        "sl_pts":           8,
        "tp_pts":           10,
        "runner_ext":       2.2,
        "retracement_pct":  None,      # uses fixed TP_TRAIL
        "min_conf_short":   0.55,
        "min_conf_long":    0.55,
        "lot_size":         65,
        "default_lots":     4,
        "hold_candles":     9,
        "target_delta":     0.30,
        "candle_minutes":   5,
    },
}


# =============================================================================
# TRADE STATE MACHINE
# =============================================================================

class TradeState(Enum):
    IDLE = "IDLE"
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

def place_order_safe(kite, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **kwargs)
            return order_id
        except Exception as e:
            err = str(e)
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

        # Paper trading
        self._paper_order_counter = 0
        self._paper_log_file = f"data/ml_paper_log_{self.instrument}.csv"

        # Session trade log for EOD summary
        self.completed_trades = []

        # ML engine — configured for 5-min candles
        self.engine = MLEngine(self.instrument)
        self.engine.model_xgb = f"data/ml_runner_{self.instrument}.ubj"
        self.engine.model_file = f"data/ml_runner_{self.instrument}.joblib"
        self.engine.dataset_file = f"data/ml_runner_dataset_{self.instrument}.csv"
        self.engine.threshold_file = f"data/ml_runner_threshold_{self.instrument}.txt"

        self.resampler = Resampler(candle_minutes=self.cfg["candle_minutes"])

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

    def _send_alert(self, msg):
        prefix = "📝 [PAPER] " if self.paper else ""
        send_telegram_message(prefix + msg)

    def _log_paper_trade(self, action, price, qty, pnl=None, notes=""):
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
        write_header = not os.path.exists(self._paper_log_file)
        with open(self._paper_log_file, "a", newline="") as f:
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
              f"TP2: {round(self.cfg['tp_pts'] * self.cfg['runner_ext'], 1)}pts")
        trail_desc = (f"{int(self.cfg['retracement_pct']*100)}% retracement"
                      if self.cfg['retracement_pct'] else "TP_TRAIL")
        print(f"  Lot2 floor: {trail_desc}")
        print(f"{'='*60}\n")

        # Load or train model
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
        live_csv = self.profile["csv_file"]
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
        self.engine.train()
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
                # 1. If in position, monitor exits every minute
                if self.state in (TradeState.POSITION_OPEN, TradeState.LOT1_EXITED):
                    self._monitor_position()

                # 2. Check for new ML signal at 5-min boundaries
                if self.state in (TradeState.IDLE, TradeState.COOLDOWN):
                    self._check_signal()

            except Exception as e:
                print(f"  ERROR in main loop: {e}")
                traceback.print_exc()

            self._wait_until_next_minute()

    # ── Signal Detection ───────────────────────────────────────────────────────

    def _check_signal(self):
        csv_file = self.profile["csv_file"]
        if not os.path.exists(csv_file):
            return

        try:
            raw = self.resampler.load_csv(csv_file)
            candles = self.resampler.resample_v2(raw)
        except Exception:
            return

        if len(candles) < 2:
            return

        # Only act on a NEW candle
        if len(candles) == self.last_candle_count:
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
        latest = candles.iloc[-1].to_dict()
        result = self.engine.predict_latest(latest)

        if result["signal"] == 0:
            return

        direction = "LONG" if result["signal"] == 1 else "SHORT"
        confidence = result["confidence"]

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

        print(f"\n  📡 ML SIGNAL: {direction} conf={confidence:.2f}")
        self._enter_trade(direction, confidence)

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

        # Select 0.3 delta strike
        strike, trading_sym = self._select_strike(spot, option_type, expiry)
        if strike is None:
            print("  Cannot find target delta strike — skipping entry")
            return

        full_symbol = f"{self.exchange}:{trading_sym}"
        qty_per_side = self.cfg["lot_size"] * max(1, self.num_lots // 2)
        total_qty = qty_per_side * 2

        # Get option LTP before entry
        option_ltp = get_ltp_safe(self.kite, full_symbol)
        if option_ltp is None:
            print("  Cannot get option LTP — skipping entry")
            return

        self._send_alert(
            f"📡 ML SIGNAL: {direction} conf={confidence:.2f}\n"
            f"{self.instrument.upper()} spot={spot}\n"
            f"Entering {trading_sym} qty={total_qty}"
        )

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
            self._send_alert(f"❌ Entry order FAILED for {trading_sym}")
            return

        # Wait for fill and get actual entry price
        if not self.paper:
            time.sleep(1)
        entry_price = self._get_fill_price(entry_order_id)
        if entry_price is None:
            entry_price = get_ltp_safe(self.kite, full_symbol) or option_ltp

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
        }

        self.state = TradeState.POSITION_OPEN
        self._save_state()

        self._send_alert(
            f"✅ ENTRY: Bought {total_qty} qty {trading_sym} @ {entry_price}\n"
            f"SL-M placed @ {sl_trigger} (order {sl_order_id})\n"
            f"TP1: {tp_price} | TP2: {tp2_price}"
        )
        print(f"  ✅ Entry: {trading_sym} @ {entry_price}, SL-M @ {sl_trigger}")

        if self.paper:
            self._log_paper_trade("ENTRY", entry_price, total_qty,
                                  notes=f"conf={confidence:.2f}")

    # ── Position Monitor ───────────────────────────────────────────────────────

    def _monitor_position(self):
        t = self.trade
        ltp = get_ltp_safe(self.kite, t["full_symbol"])
        if ltp is None:
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

        # Check if SL-M was triggered (order status / paper LTP check)
        if self._is_sl_triggered(ltp):
            pnl = round((t["sl_trigger"] - t["entry_price"]) * t["total_qty"])
            self._send_alert(
                f"🔴 SL HIT: {t['trading_sym']}\n"
                f"Entry: {t['entry_price']} → Exit: {t['sl_trigger']}\n"
                f"P&L: Rs {pnl:+,}"
            )
            if self.paper:
                self._log_paper_trade("SL_HIT", t["sl_trigger"], t["total_qty"],
                                      pnl=pnl)
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
            pnl_lot1 = t.get("lot1_pnl", 0)
            pnl_lot2 = round((sl_price - t["entry_price"]) * t["qty_per_side"])
            total_pnl = pnl_lot1 + pnl_lot2
            self._send_alert(
                f"🔴 Lot2 SL/Trail triggered: {t['trading_sym']}\n"
                f"Lot1 P&L: Rs {pnl_lot1:+,} | Lot2 P&L: Rs {pnl_lot2:+,}\n"
                f"Total: Rs {total_pnl:+,}"
            )
            if self.paper:
                self._log_paper_trade("LOT2_TRAIL", sl_price, t["qty_per_side"],
                                      pnl=total_pnl)
            self._close_trade(candles_held, pnl=total_pnl, exit_type="TRAIL")
            return

        # Update lot2 peak
        if ltp > t["lot2_peak_ltp"]:
            t["lot2_peak_ltp"] = ltp
            # Update trailing SL if retracement mode
            if self.cfg["retracement_pct"]:
                self._update_lot2_trail(ltp)

        # Check TP2 hit
        if ltp >= t["tp2_price"]:
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
            time.sleep(0.5)
            fill = self._get_fill_price(lot1_order)
            if fill:
                lot1_exit_price = fill

        lot1_pnl = round((lot1_exit_price - t["entry_price"]) * t["qty_per_side"])
        t["lot1_pnl"] = lot1_pnl
        t["lot1_exit_price"] = lot1_exit_price

        # Place new SL-M for lot2 at TP floor
        lot2_sl_trigger = round(t["tp_price"], 2)
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

        if self.paper:
            self._log_paper_trade("LOT1_TP", lot1_exit_price, t["qty_per_side"],
                                  pnl=lot1_pnl, notes="lot2 running")

    # ── Lot2 Trail Update ──────────────────────────────────────────────────────

    def _update_lot2_trail(self, current_ltp):
        t = self.trade
        pct = self.cfg["retracement_pct"]
        peak = t["lot2_peak_ltp"]

        # Trail floor = max(TP price, peak × (1 - retracement%))
        trail_price = peak * (1.0 - pct)
        new_sl = round(max(t["tp_price"], trail_price), 2)

        old_sl = t.get("lot2_sl_trigger", t["tp_price"])
        if new_sl > old_sl + 0.5:
            # Cancel old SL and place new one higher
            self._cancel_order(t["sl_order_id"])
            new_order = self._place_order(
                exchange=self.exchange,
                tradingsymbol=t["trading_sym"],
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=t["qty_per_side"],
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_SLM,
                trigger_price=new_sl,
            )
            if new_order:
                t["sl_order_id"] = new_order
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
        if self.paper:
            self._log_paper_trade(f"{reason}_EXIT", current_ltp, t["total_qty"],
                                  pnl=pnl)
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
        if self.paper:
            self._log_paper_trade(f"LOT2_{reason}", exit_price, t["qty_per_side"],
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
        self.cooldown_remaining = max(1, candles_held)
        self.state = TradeState.COOLDOWN
        self.trade = {}
        self._save_state()
        print(f"  Trade closed, cooldown={self.cooldown_remaining} candles")

    # ── SL Order Status ────────────────────────────────────────────────────────

    def _is_sl_triggered(self, ltp=None):
        if self.paper:
            sl = self.trade.get("lot2_sl_trigger", self.trade.get("sl_trigger"))
            return ltp is not None and sl is not None and ltp <= sl
        order_id = self.trade.get("sl_order_id")
        if not order_id:
            return False
        try:
            history = self.kite.order_history(order_id)
            latest = history[-1]
            return latest["status"] == "COMPLETE"
        except Exception:
            return False

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

        # Get all strikes near spot
        atm = round(spot / self.strike_step) * self.strike_step
        strikes_range = [atm + i * self.strike_step
                         for i in range(-10, 11)]

        # Build candidates with delta
        candidates = []
        for s in strikes_range:
            # Find tradingsymbol
            tsym = self._find_trading_symbol(instruments, s, option_type, expiry)
            if not tsym:
                continue

            full_sym = f"{self.exchange}:{tsym}"
            ltp = get_ltp_safe(self.kite, full_sym)
            if ltp is None or ltp <= 0:
                continue

            # Compute IV → delta
            iv = implied_vol(spot, s, T, r, ltp, option_type)
            if iv is None or iv <= 0:
                continue

            delta = bs_delta(spot, s, T, r, iv, option_type)
            candidates.append({
                "strike": s,
                "trading_sym": tsym,
                "ltp": ltp,
                "delta": delta,
                "abs_delta": abs(delta),
            })

        if not candidates:
            return None, None

        # Pick closest to target delta
        target = self.cfg["target_delta"]
        best = min(candidates, key=lambda c: abs(c["abs_delta"] - target))
        print(f"  Selected strike {best['strike']} {option_type} "
              f"delta={best['delta']:.3f} ltp={best['ltp']}")
        return best["strike"], best["trading_sym"]

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

    def _save_state(self):
        data = {
            "state": self.state.value,
            "trade": self.trade,
            "cooldown_remaining": self.cooldown_remaining,
            "last_candle_count": self.last_candle_count,
            "saved_at": datetime.now().isoformat(),
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

        # Retrain model for tomorrow
        print("\n  EOD — retraining 5-min model...")
        try:
            self._retrain_model()
        except Exception as e:
            print(f"  Retrain failed: {e}")

        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

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
