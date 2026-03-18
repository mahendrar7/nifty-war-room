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
    def __init__(self, min_confirm=3):
        self.min_confirm          = min_confirm
        self.confirmed_bias       = "RANGE"
        self.confirmed_regime     = "UNCLEAR"
        self.confirmed_action     = "WAIT"
        self.confirmed_confidence = 0
        self.candidate_bias       = None
        self.candidate_count      = 0
        self.stable_minutes       = 0

    def update(self, new_bias, new_regime, new_action, new_confidence):
        if new_bias == self.confirmed_bias:
            self.candidate_bias  = None
            self.candidate_count = 0
            self.stable_minutes += 1
        else:
            if new_bias == self.candidate_bias:
                self.candidate_count += 1
            else:
                self.candidate_bias  = new_bias
                self.candidate_count = 1

            if self.candidate_count >= self.min_confirm:
                self.confirmed_bias       = new_bias
                self.confirmed_regime     = new_regime
                self.confirmed_action     = new_action
                self.confirmed_confidence = new_confidence
                self.candidate_bias       = None
                self.candidate_count      = 0
                self.stable_minutes       = 0

        if new_bias == self.confirmed_bias:
            self.confirmed_regime     = new_regime
            self.confirmed_action     = new_action
            self.confirmed_confidence = new_confidence

        return (self.confirmed_bias, self.confirmed_regime,
                self.confirmed_action, self.confirmed_confidence,
                self.candidate_bias, self.candidate_count,
                self.stable_minutes)


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

        # NEW: throttle cache for slow-compute signals
        self.throttle_cache      = {}   # {"signal_name": {"tick": N, "result": ...}}
        self.tick_counter        = 0    # incremented every main loop iteration

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
        self.throttle_cache.clear()
        self.tick_counter        = 0
        # ml_consecutive_wrong intentionally not reset — ML reliability persists across sessions


# Singleton — all modules import this instance
state = MarketState()


# =============================================================================
# RESTORE STATE FROM CSV
# =============================================================================
def restore_state_from_csv(debug_mode=False):
    if not os.path.exists(CSV_FILE):
        return

    try:
        with open(CSV_FILE, "r") as f:
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

        print(f"✅ State restored from CSV ({len(snapshots)} snapshots)")

    except Exception as e:
        import traceback
        print(f"State restore failed: {e}")
        if debug_mode:
            traceback.print_exc()