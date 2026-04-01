"""
state.py — MarketState, RegimeTracker, and CSV restore.
One shared `state` instance imported by all modules.
"""

import os
import pandas as pd
from collections import deque
from datetime import datetime

from config import CSV_FILE


# =============================================================================
# REGIME TRACKER
# =============================================================================
class RegimeTracker:
    """
    Dampens regime and bias flips with N-candle confirmation.
    Raw signals flip every tick; confirmed signals only change after
    `min_confirm` consecutive candles agree on a new value.

    Tracks bias and regime independently so a regime can change
    (e.g. PINNING → VOL EXPANSION) even if bias stays RANGE.
    """
    def __init__(self, min_confirm=3):
        self.min_confirm          = min_confirm
        # Confirmed state — what downstream consumers see
        self.confirmed_bias       = "RANGE"
        self.confirmed_regime     = "UNCLEAR"
        self.confirmed_action     = "WAIT"
        self.confirmed_confidence = 0
        # Bias candidate tracking
        self.candidate_bias       = None
        self.candidate_bias_count = 0
        # Regime candidate tracking
        self.candidate_regime       = None
        self.candidate_regime_count = 0
        self.stable_minutes       = 0

    def update(self, new_bias, new_regime, new_action, new_confidence):
        # ── Bias hysteresis ──────────────────────────────────────────
        if new_bias == self.confirmed_bias:
            self.candidate_bias       = None
            self.candidate_bias_count = 0
            self.stable_minutes += 1
        else:
            if new_bias == self.candidate_bias:
                self.candidate_bias_count += 1
            else:
                self.candidate_bias       = new_bias
                self.candidate_bias_count = 1

            if self.candidate_bias_count >= self.min_confirm:
                self.confirmed_bias       = new_bias
                self.candidate_bias       = None
                self.candidate_bias_count = 0
                self.stable_minutes       = 0

        # ── Regime hysteresis ────────────────────────────────────────
        if new_regime == self.confirmed_regime:
            self.candidate_regime       = None
            self.candidate_regime_count = 0
        else:
            if new_regime == self.candidate_regime:
                self.candidate_regime_count += 1
            else:
                self.candidate_regime       = new_regime
                self.candidate_regime_count = 1

            if self.candidate_regime_count >= self.min_confirm:
                self.confirmed_regime       = new_regime
                self.candidate_regime       = None
                self.candidate_regime_count = 0

        # Action and confidence always follow confirmed regime's latest values
        if new_regime == self.confirmed_regime:
            self.confirmed_action     = new_action
            self.confirmed_confidence = new_confidence

        return (self.confirmed_bias, self.confirmed_regime,
                self.confirmed_action, self.confirmed_confidence,
                self.candidate_bias, self.candidate_bias_count,
                self.stable_minutes)

    def force(self, bias, regime, action, confidence):
        """Structural bypass — immediately confirm without waiting."""
        self.confirmed_bias       = bias
        self.confirmed_regime     = regime
        self.confirmed_action     = action
        self.confirmed_confidence = confidence
        self.candidate_bias       = None
        self.candidate_bias_count = 0
        self.candidate_regime       = None
        self.candidate_regime_count = 0
        self.stable_minutes       = 0


# =============================================================================
# MARKET STATE
# =============================================================================
class MarketState:
    def __init__(self):
        self.previous_snapshot   = None
        self.previous_gamma      = None
        self.previous_spot       = None          # FIX: track actual spot price
        self.breakout_counter    = 0
        self.breakout_direction  = None
        self.breakout_strike     = None
        self.straddle_history    = deque(maxlen=20)
        self.oi_velocity_history = deque(maxlen=30)
        self.regime_tracker      = RegimeTracker(min_confirm=3)
        self.last_ml_result      = None
        self.gamma_flip_alerted  = False   # used by flip breakout detector
        self.flip_approach_alerted  = False # used by danger zone approach alert
        self.previous_flip_distance = None  # tracks if spot is closing in
        self.trap_alerted        = None
        self.vacuum_alerted      = None
        self.liq_accel_alerted   = None
        self.active_trade        = None
        self.last_suggestion     = None

        # NEW: wall retreat tracking
        self.call_wall_history   = deque(maxlen=10)
        self.put_wall_history    = deque(maxlen=10)

        # NEW: spot history for trend detection
        self.spot_history        = deque(maxlen=60)  # last 60 minutes
        self.session_high        = None              # intraday high
        self.session_low         = None              # intraday low

        # NEW: gamma history for momentum / rate-of-change tracking
        self.gamma_history       = deque(maxlen=10)  # last 10 candles

        # NEW: heavyweight stock price history for ROC tracking
        self.hw_history          = deque(maxlen=60)   # last 60 minutes of price snapshots

        # NEW: throttle cache for slow-compute signals
        self.throttle_cache      = {}   # {"signal_name": {"tick": N, "result": ...}}
        self.tick_counter        = 0    # incremented every main loop iteration

        # Session character — set from open straddle by 9:30
        # "TREND" = high IV open (straddle/spot >= 1.8%), expect big moves
        # "RANGE" = low IV open, but can be upgraded by session_range later
        # None = not yet determined (pre-9:30)
        self.session_character   = None
        self.open_iv_proxy       = None  # straddle/spot % at open

        # NEW: ML consecutive wrong counter
        self.ml_consecutive_wrong = 0

    def reset_session(self):
        """Called at market close to wipe intraday state."""
        self.previous_snapshot   = None
        self.previous_gamma      = None
        self.previous_spot       = None
        self.breakout_counter    = 0
        self.breakout_direction  = None
        self.breakout_strike     = None
        self.straddle_history.clear()
        self.oi_velocity_history.clear()
        self.last_ml_result      = None
        self.gamma_flip_alerted  = False
        self.flip_approach_alerted  = False
        self.previous_flip_distance = None
        self.trap_alerted        = None
        self.vacuum_alerted      = None
        self.liq_accel_alerted   = None
        self.active_trade        = None
        self.last_suggestion     = None
        self.regime_tracker      = RegimeTracker(min_confirm=3)
        self.call_wall_history.clear()
        self.put_wall_history.clear()
        self.spot_history.clear()
        self.session_high        = None
        self.session_low         = None
        self.hw_history.clear()
        self.throttle_cache.clear()
        self.tick_counter        = 0
        self.session_character   = None
        self.open_iv_proxy       = None
        # ml_consecutive_wrong intentionally not reset — ML reliability persists across sessions


# Singleton — all modules import this instance
state = MarketState()


# =============================================================================
# RESTORE STATE FROM CSV
# =============================================================================
def restore_state_from_csv(debug_mode=False, csv_file=CSV_FILE):
    if not os.path.exists(csv_file):
        return

    try:
        with open(csv_file, "r") as f:
            lines = list(deque(f, 2000))

        df = pd.read_csv(
            pd.io.common.StringIO("".join(lines)),
            on_bad_lines="skip"
        )

        if df.empty:
            return

        required = {"timestamp", "option_type", "strike", "ltp", "oi", "volume"}
        missing  = required - set(df.columns)
        if missing:
            print(f"State restore skipped: CSV missing columns {missing}")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return

        cutoff = df["timestamp"].max() - pd.Timedelta(minutes=30)
        df = df[df["timestamp"] >= cutoff]

        snapshots = []
        for ts, group in df.groupby("timestamp"):
            option_df = group.copy()
            calls = option_df[option_df["option_type"] == "CE"].rename(
                columns={"ltp": "call_ltp", "oi": "call_oi", "volume": "call_vol"}
            )
            puts = option_df[option_df["option_type"] == "PE"].rename(
                columns={"ltp": "put_ltp", "oi": "put_oi", "volume": "put_vol"}
            )
            if calls.empty or puts.empty:
                continue
            merged = pd.merge(
                calls[["strike", "call_ltp", "call_oi", "call_vol"]],
                puts [["strike", "put_ltp",  "put_oi",  "put_vol"]],
                on="strike"
            )
            if not merged.empty:
                snapshots.append(merged)

        if snapshots:
            state.previous_snapshot = snapshots[-1]

        for snap in snapshots[-20:]:
            atm_idx  = (snap["strike"] - snap["strike"].median()).abs().argsort()[:1]
            atm_row  = snap.iloc[atm_idx]
            if atm_row.empty:
                continue
            straddle = atm_row["call_ltp"].values[0] + atm_row["put_ltp"].values[0]
            state.straddle_history.append((datetime.now(), straddle))

        # Restore spot_history and session high/low from spot column
        if "spot" in df.columns:
            spot_series = df.groupby("timestamp")["spot"].first().sort_index()
            # Use last 60 spots for history
            for sp in spot_series.values[-60:]:
                state.spot_history.append(float(sp))
            # Session high/low from ALL data in file (not just 30-min cutoff)
            try:
                with open(csv_file, "r") as f2:
                    full_lines = list(deque(f2, 20000))
                full_df = pd.read_csv(
                    pd.io.common.StringIO("".join(full_lines)),
                    on_bad_lines="skip"
                )
                if "spot" in full_df.columns:
                    all_spots = full_df["spot"].dropna()
                    if len(all_spots) > 0:
                        state.session_high = float(all_spots.max())
                        state.session_low  = float(all_spots.min())
            except Exception:
                pass  # session high/low not critical

        print(f"✅ State restored from CSV ({len(snapshots)} snapshots, "
              f"{len(state.spot_history)} spots)")

    except Exception as e:
        import traceback
        print(f"State restore failed: {e}")
        if debug_mode:
            traceback.print_exc()
