"""
backtest_feedback_loop.py — Walk-forward backtest: baseline vs feedback-weighted ML.

Processes the SENSEX (or NIFTY) options log to simulate trades and measure
whether retraining with NICE/NAUGHTY sample weights (derived from actual
SL/TP outcomes) improves hit rate and P&L over a plain baseline model.

SL/TP outcome is determined via spot-move × delta (identical to backtest_ml.py),
which is the validated simulation approach for this codebase.

Flow per test day:
  1. Resample day's 1-min log to 5-min candles → generate ML signals
  2. For each signal, scan forward through subsequent spot prices
  3. Check if spot moved SL_PTS/delta (wrong) or TP_PTS/delta (correct)
  4. Record outcome in feedback ledger
  5. On retrain: feedback model applies NAUGHTY×4 / NICE×3 sample weights
     to candles where the model was previously wrong / right

Usage:
    python backtest_feedback_loop.py --instrument sensex
    python backtest_feedback_loop.py --instrument sensex --warm-up 7 --retrain-every 1
    python backtest_feedback_loop.py --instrument sensex --save-feedback
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
from ml_engine import (
    Resampler, build_targets, add_lag_features,
    PROBA_THRESHOLD, FeedbackLedger,
    NICE_WEIGHT, NAUGHTY_WEIGHT, NEUTRAL_WEIGHT,
)

STRIKE_STEP = {"sensex": 100, "nifty": 50}
LOT_SIZES   = {"sensex": 20,  "nifty": 65}
SLIPPAGE    = {"sensex": 2,   "nifty": 1}
DELTA       = {"sensex": 0.40, "nifty": 0.30}

# Runner config mirrors ml_runner.py RUNNER_CONFIGS
RUNNER_CFG = {
    "sensex": {"sl_pts": 15, "tp_pts": 25, "num_lots": 6,
               "min_conf_long": 0.75, "min_conf_short": 0.55,
               "candle_minutes": 5},
    "nifty":  {"sl_pts": 8,  "tp_pts": 10, "num_lots": 4,
               "min_conf_long": 0.55, "min_conf_short": 0.55,
               "candle_minutes": 5},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_days(instrument, candle_minutes=5):
    """Load all archived 1-min CSVs, resample to N-min candles per day.
    Returns list of {date, candles} sorted by date.
    The raw 1-min df is NOT kept in memory here — we use the 5-min candles
    to derive predictions and spot prices for the SL/TP walk-forward check.
    """
    pattern = f"data/options_log_1min_{instrument}_????????.csv"
    files = sorted(
        glob.glob(pattern),
        key=lambda f: datetime.strptime(
            os.path.basename(f).split("_")[-1].replace(".csv", ""), "%d%m%Y"
        ),
    )
    if not files:
        print(f"No files found: {pattern}")
        sys.exit(1)

    resampler = Resampler(candle_minutes=candle_minutes)
    days = []
    for f in files:
        try:
            raw = resampler.load_csv(f)
            candles = resampler.resample_v2(raw)
            if not candles.empty:
                date_str = os.path.basename(f).split("_")[-1].replace(".csv", "")
                days.append({"file": f, "date": date_str, "candles": candles})
        except Exception as e:
            print(f"  Skipped {os.path.basename(f)}: {e}")

    print(f"\nLoaded {len(days)} days ({candle_minutes}-min candles)")
    return days


# =============================================================================
# TRADE OUTCOME SIMULATION (spot × delta, same as backtest_ml.py)
# =============================================================================

def simulate_outcome(predictions, signal_idx, direction, sl_pts, tp_pts,
                     delta, candle_minutes, max_candles):
    """
    Scan subsequent candle spot prices to find SL or TP hit.
    Uses spot_move × delta to approximate option P&L — identical logic to
    backtest_ml.simulate_ml_pnl() so results are consistent.

    Returns: (outcome, exit_type, candles_held)
      outcome  : "correct" (TP), "wrong" (SL), "neutral" (TIME/EOD)
    """
    nifty_sl  = sl_pts  / delta   # spot move needed to hit SL
    nifty_tp  = tp_pts  / delta   # spot move needed to hit TP

    entry_spot = predictions[signal_idx]["spot"]

    for j, fp in enumerate(predictions[signal_idx + 1:], start=1):
        if j > max_candles:
            return "neutral", "TIME", j

        move = fp["spot"] - entry_spot
        if direction == "SHORT":
            move = -move

        if move <= -nifty_sl:
            return "wrong", "SL", j
        if move >= nifty_tp:
            return "correct", "TP", j

    return "neutral", "EOD", len(predictions) - signal_idx - 1


# =============================================================================
# MODEL TRAINING WITH INLINE FEEDBACK WEIGHT COMPUTATION
# =============================================================================

def train_model(train_df, instrument, ledger=None):
    """
    Train XGBoost on train_df.
    If ledger is provided, feedback weights are computed from the PROCESSED
    dataframe (after build_targets + add_lag_features) so lengths always match.
    """
    df, x_pts = build_targets(train_df.copy(), instrument=instrument)
    df = add_lag_features(df)
    if len(df) < 20:
        return None, None, None

    exclude = {"target", "future_high", "future_low",
               "spot_open", "spot_high", "spot_low", "trading_date"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["target"].values
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = np.array([label_map[v] for v in y])

    if len(np.unique(y_mapped)) < 2:
        return None, None, None

    sw = compute_sample_weight("balanced", y_mapped)

    if ledger is not None and not ledger._df.empty:
        ts_list   = df.index.tolist()
        date_list = df["trading_date"].tolist() if "trading_date" in df.columns else [""] * len(ts_list)
        fb_weights = ledger.get_sample_weights(ts_list, date_list)
        sw = sw * fb_weights
        n_naughty = int((fb_weights >= NAUGHTY_WEIGHT).sum())
        n_nice    = int((fb_weights == NICE_WEIGHT).sum())
        if n_naughty + n_nice > 0:
            print(f"    Feedback: {n_naughty} NAUGHTY candles (×{NAUGHTY_WEIGHT})  "
                  f"{n_nice} NICE candles (×{NICE_WEIGHT})")

    n = len(X)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=min(200, max(50, n * 2)),
        max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=max(1, n // 20),
        reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y_mapped, sample_weight=sw, verbose=False)
    return model, feature_cols, x_pts


# =============================================================================
# PREDICT
# =============================================================================

def predict_candles(model, feature_cols, x_pts, candles, instrument):
    """Run model on a day's candles. Returns list of prediction dicts."""
    df, _ = build_targets(candles.copy(), x_points=x_pts, instrument=instrument)
    df = add_lag_features(df)
    if df.empty:
        return []

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].values
    probas = model.predict_proba(X)
    inv = {0: -1, 1: 0, 2: 1}

    results = []
    for i, row in enumerate(df.itertuples()):
        pred_class = int(np.argmax(probas[i]))
        conf = float(probas[i][pred_class])
        signal = inv[pred_class]
        if conf < PROBA_THRESHOLD:
            signal = 0
        results.append({
            "timestamp":  row.Index,
            "spot":       float(row.spot_close),
            "signal":     signal,
            "actual":     int(row.target),
            "confidence": round(conf, 3),
        })
    return results


# =============================================================================
# RUN ONE DAY
# =============================================================================

def run_day(model, feature_cols, x_pts, day, instrument, cfg,
            ledger=None, cooldown_ref=None):
    """
    Generate signals, simulate SL/TP via spot move, optionally record to ledger.
    Returns list of trade dicts.
    """
    candles = day["candles"]
    delta     = DELTA.get(instrument, 0.40)
    slip      = SLIPPAGE.get(instrument, 2)
    lot_size  = LOT_SIZES.get(instrument, 20)
    max_candles = 2  # 2 forward candles = 10 min for 5-min candles

    preds = predict_candles(model, feature_cols, x_pts, candles, instrument)

    trades = []
    cooldown = cooldown_ref[0] if cooldown_ref else 0

    for i, pred in enumerate(preds):
        if cooldown > 0:
            cooldown -= 1
            continue
        if pred["signal"] == 0:
            continue

        ts = pred["timestamp"]
        if hasattr(ts, "hour"):
            if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 15):
                continue
            if ts.hour == 9 and ts.minute < 30:
                continue

        direction = "LONG" if pred["signal"] == 1 else "SHORT"
        min_conf = (cfg["min_conf_long"] if direction == "LONG"
                    else cfg["min_conf_short"])
        if pred["confidence"] < min_conf:
            continue

        # Simulate SL/TP on spot (same method as backtest_ml.py)
        outcome, exit_type, candles_held = simulate_outcome(
            preds, i, direction,
            cfg["sl_pts"], cfg["tp_pts"], delta,
            cfg["candle_minutes"], max_candles,
        )

        # P&L
        qty = lot_size * cfg["num_lots"]
        if exit_type == "SL":
            pnl = round((-cfg["sl_pts"] - slip) * qty)
        elif exit_type == "TP":
            pnl = round((cfg["tp_pts"] - slip) * qty)
        else:
            pnl = 0

        cooldown = max(1, candles_held)

        if ledger is not None:
            ledger.record_trade_outcome(
                candle_ts=ts,
                signal=pred["signal"],
                confidence=pred["confidence"],
                exit_type=exit_type,
                outcome=outcome,
                trading_date=str(ts.date()),
            )

        trades.append({
            "timestamp":      ts,
            "direction":      direction,
            "confidence":     pred["confidence"],
            "exit_type":      exit_type,
            "outcome":        outcome,
            "pnl":            pnl,
            "signal_correct": pred["signal"] == pred["actual"],
        })

    if cooldown_ref is not None:
        cooldown_ref[0] = cooldown
    return trades


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def _day_stats(trades):
    n = len(trades)
    if n == 0:
        return {"trades": 0, "tp": 0, "sl": 0, "neutral": 0,
                "tp_rate": 0.0, "pnl": 0}
    tp  = sum(1 for t in trades if t["exit_type"] == "TP")
    sl  = sum(1 for t in trades if t["exit_type"] == "SL")
    pnl = sum(t["pnl"] for t in trades)
    tp_rate = tp / (tp + sl) if (tp + sl) > 0 else 0.0
    return {"trades": n, "tp": tp, "sl": sl, "neutral": n - tp - sl,
            "tp_rate": tp_rate, "pnl": pnl}


def _fmt_stats(stats):
    tp_c  = Fore.GREEN if stats["tp_rate"] >= 0.5 else Fore.RED
    pnl_c = Fore.GREEN if stats["pnl"] >= 0 else Fore.RED
    return (f"trades={stats['trades']:>3d}  "
            f"TP={stats['tp']:>2d} SL={stats['sl']:>2d}  "
            f"{tp_c}hit={stats['tp_rate']:.0%}{Style.RESET_ALL}  "
            f"{pnl_c}Rs{stats['pnl']:>+,}{Style.RESET_ALL}")


def _print_final(all_base, all_fb):
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — BASELINE vs FEEDBACK")
    print(f"{'='*70}")

    for label, trades in [("BASELINE", all_base), ("FEEDBACK", all_fb)]:
        if not trades:
            continue
        tp  = sum(1 for t in trades if t["exit_type"] == "TP")
        sl  = sum(1 for t in trades if t["exit_type"] == "SL")
        pnl = sum(t["pnl"] for t in trades)
        tp_rate = tp / (tp + sl) if (tp + sl) > 0 else 0.0
        avg_pnl = pnl / len(trades) if trades else 0
        pnl_c = Fore.GREEN if pnl >= 0 else Fore.RED
        tp_c  = Fore.GREEN if tp_rate >= 0.5 else Fore.RED
        print(f"\n  {label}:")
        print(f"    Trades     : {len(trades)}")
        print(f"    TP / SL    : {tp} / {sl}  (neutral={len(trades)-tp-sl})")
        print(f"    TP hit rate: {tp_c}{tp_rate:.1%}{Style.RESET_ALL}")
        print(f"    Net P&L    : {pnl_c}Rs {pnl:+,}{Style.RESET_ALL}")
        print(f"    Avg/trade  : Rs {avg_pnl:+,.0f}")

    base_pnl = sum(t["pnl"] for t in all_base)
    fb_pnl   = sum(t["pnl"] for t in all_fb)
    base_tp  = sum(1 for t in all_base if t["exit_type"] == "TP")
    fb_tp    = sum(1 for t in all_fb   if t["exit_type"] == "TP")
    base_sl  = sum(1 for t in all_base if t["exit_type"] == "SL")
    fb_sl    = sum(1 for t in all_fb   if t["exit_type"] == "SL")

    d_color = Fore.GREEN if fb_pnl >= base_pnl else Fore.RED
    print(f"\n  DELTA (Feedback − Baseline):")
    print(f"    P&L  : {d_color}Rs {fb_pnl-base_pnl:+,}{Style.RESET_ALL}")
    print(f"    TP   : {fb_tp-base_tp:+d}  (positive = more TPs with feedback)")
    print(f"    SL   : {fb_sl-base_sl:+d}  (negative = fewer SLs with feedback)")
    print(f"{'='*70}\n")


# =============================================================================
# EOD AUTOMATION — importable by ml_runner
# =============================================================================

def build_feedback_ledger(instrument="sensex", candle_minutes=5,
                           warm_up=7, retrain_every=3, verbose=True):
    """
    Rebuild the feedback ledger from all archived options log CSVs.

    Called automatically from ml_runner._on_market_close() before EOD retrain.
    Runs in feedback-only mode (no baseline comparison, no console table).

    Steps:
      1. Reload all archived 1-min CSVs → 5-min candles
      2. Walk-forward: train feedback-weighted model, simulate SL/TP per signal
      3. Record every outcome to data/ml_feedback_<instrument>.csv (rebuilt fresh)

    Returns: count of outcome records written.
    """
    cfg = dict(RUNNER_CFG[instrument])
    cfg["candle_minutes"] = candle_minutes
    save_path = f"data/ml_feedback_{instrument}.csv"

    if verbose:
        print(f"\n  📊 Rebuilding feedback ledger from options log ({instrument.upper()})...")

    days = load_all_days(instrument, candle_minutes)
    if len(days) <= warm_up:
        if verbose:
            print(f"  ⚠ Not enough days ({len(days)}) for warm-up ({warm_up}) — skipping")
        return 0

    # Start fresh — full rebuild ensures ledger is consistent with
    # current signal logic and avoids stale/duplicate entries
    ledger = FeedbackLedger(path=save_path)
    ledger._df = ledger._empty_df()

    model = fcols = xpts = None
    cooldown_ref = [0]
    outcomes = 0

    for idx, day in enumerate(days):
        if idx < warm_up:
            continue

        train_df = pd.concat([d["candles"] for d in days[:idx]])

        if model is None or (idx - warm_up) % retrain_every == 0:
            model, fcols, xpts = train_model(train_df, instrument, ledger=ledger)
            if model is None:
                continue

        trades = run_day(model, fcols, xpts, day, instrument, cfg,
                         ledger=ledger, cooldown_ref=cooldown_ref)
        outcomes += len(trades)

    if verbose:
        correct = (ledger._df["outcome"] == "correct").sum() if not ledger._df.empty else 0
        wrong   = (ledger._df["outcome"] == "wrong").sum()   if not ledger._df.empty else 0
        print(f"  ✅ Feedback ledger rebuilt: {len(ledger._df)} records  "
              f"(TP={correct}  SL={wrong}  "
              f"hit={correct/(correct+wrong)*100:.1f}%)" if (correct+wrong) > 0
              else f"  ✅ Feedback ledger rebuilt: {len(ledger._df)} records")

    return len(ledger._df)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-forward feedback loop backtest")
    parser.add_argument("--instrument",    default="sensex", choices=["sensex", "nifty"])
    parser.add_argument("--candle",        type=int, default=5)
    parser.add_argument("--warm-up",       type=int, default=7,
                        help="Training warm-up days before testing starts")
    parser.add_argument("--retrain-every", type=int, default=3,
                        help="Retrain both models every N test days")
    parser.add_argument("--save-feedback", action="store_true",
                        help="Persist feedback ledger to data/ml_feedback_<inst>.csv")
    args = parser.parse_args()

    instrument     = args.instrument.lower()
    candle_minutes = args.candle
    warm_up        = args.warm_up
    retrain_every  = args.retrain_every
    cfg = dict(RUNNER_CFG[instrument])
    cfg["candle_minutes"] = candle_minutes

    fb_path = f"data/ml_feedback_{instrument}.csv" if args.save_feedback else None

    print(f"\n{'='*70}")
    print(f"  FEEDBACK LOOP WALK-FORWARD — {instrument.upper()}")
    print(f"  Candle: {candle_minutes}min | SL: {cfg['sl_pts']}pts | "
          f"TP: {cfg['tp_pts']}pts | Warm-up: {warm_up}d | "
          f"Retrain: every {retrain_every}d")
    print(f"  Outcome: spot × delta={DELTA[instrument]} (SL/TP in spot pts)")
    print(f"  Feedback: NAUGHTY={NAUGHTY_WEIGHT}× | NICE={NICE_WEIGHT}×")
    print(f"{'='*70}\n")

    days = load_all_days(instrument, candle_minutes)
    if len(days) <= warm_up:
        print(f"Need more than {warm_up} days. Have {len(days)}.")
        sys.exit(1)

    # Feedback ledger for the feedback model (baseline has none)
    # Use a temp path for baseline so it doesn't persist
    ledger_base = FeedbackLedger(path="data/ml_fb_baseline_tmp.csv")
    ledger_fdbk = FeedbackLedger(path=fb_path or f"data/ml_feedback_{instrument}.csv")

    base_model = fb_model = None
    base_fcols = fb_fcols = None
    base_xpts  = fb_xpts  = None
    all_base = []
    all_fb   = []
    base_cooldown = [0]
    fb_cooldown   = [0]
    test_day_count = 0

    for idx, day in enumerate(days):
        date = day["date"]

        if idx < warm_up:
            print(f"  {date}: warm-up ({idx+1}/{warm_up})")
            continue

        # Build expanding training window (all days before today)
        train_df = pd.concat([d["candles"] for d in days[:idx]])

        # (Re)train when needed
        if base_model is None or test_day_count % retrain_every == 0:
            print(f"\n  --- Retraining on {idx} days of history ---")

            # Baseline: balanced class weights only, no feedback
            base_model, base_fcols, base_xpts = train_model(
                train_df, instrument, ledger=None
            )

            # Feedback: balanced weights × ledger NICE/NAUGHTY multipliers
            fb_model, fb_fcols, fb_xpts = train_model(
                train_df, instrument, ledger=ledger_fdbk
            )

            if base_model is None or fb_model is None:
                print(f"  Training failed for {date}")
                continue

        # Run both models on today
        b_trades = run_day(base_model, base_fcols, base_xpts,
                           day, instrument, cfg,
                           ledger=ledger_base, cooldown_ref=base_cooldown)

        f_trades = run_day(fb_model, fb_fcols, fb_xpts,
                           day, instrument, cfg,
                           ledger=ledger_fdbk, cooldown_ref=fb_cooldown)

        all_base.extend(b_trades)
        all_fb.extend(f_trades)
        test_day_count += 1

        b_s = _day_stats(b_trades)
        f_s = _day_stats(f_trades)
        print(f"  {date}")
        print(f"    BASE : {_fmt_stats(b_s)}")
        print(f"    FDBK : {_fmt_stats(f_s)}")

    _print_final(all_base, all_fb)

    # Clean up temp baseline ledger
    if os.path.exists("data/ml_fb_baseline_tmp.csv"):
        os.remove("data/ml_fb_baseline_tmp.csv")

    if args.save_feedback:
        print(f"  Feedback ledger saved → data/ml_feedback_{instrument}.csv")
        print(f"  Run: python ml_engine.py retrain sensex   to apply weights to live model.")


if __name__ == "__main__":
    main()
