"""
backtest_ml.py — Walk-forward backtest of the ML model on NIFTY data.

Trains on the first N days, predicts on subsequent days, and rolls forward.
Reports accuracy, precision, and simulated P&L per day.

Usage:
    python backtest_ml.py                          # default: nifty, 15-min, 5-day warm-up
    python backtest_ml.py --candle 5               # 5-minute candles
    python backtest_ml.py --candle 3               # 3-minute candles
    python backtest_ml.py --warm-up 7              # 7-day initial training window
    python backtest_ml.py --instrument sensex
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from colorama import Fore, Style

from ml_engine import (
    Resampler, build_targets, add_lag_features, MLEngine,
    PROBA_THRESHOLD, FORWARD_CANDLES,
)


def load_all_days(instrument="nifty", candle_minutes=15):
    """Load all archived 1-min CSVs, resample to N-min candles per day."""
    pattern = f"data/options_log_1min_{instrument}_????????.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        sys.exit(1)

    resampler = Resampler(candle_minutes=candle_minutes)
    day_candles = []

    for f in files:
        try:
            raw = resampler.load_csv(f)
            candles = resampler.resample_v2(raw)
            if not candles.empty:
                date_str = os.path.basename(f).split("_")[-1].replace(".csv", "")
                day_candles.append({"file": f, "date": date_str, "candles": candles})
        except Exception as e:
            print(f"  Skipped {os.path.basename(f)}: {e}")

    print(f"Loaded {len(day_candles)} days of data ({candle_minutes}-min candles)")
    return day_candles


def train_model(train_df, instrument="nifty"):
    """Train XGBoost on provided dataframe, return model + metadata."""
    df, x_points = build_targets(train_df.copy(), instrument=instrument)
    df = add_lag_features(df)

    if len(df) < 20:
        return None, None, None, None

    exclude = {"target", "future_high", "future_low",
               "spot_open", "spot_high", "spot_low", "trading_date"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["target"].values

    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = np.array([label_map[v] for v in y])

    if len(np.unique(y_mapped)) < 2:
        return None, None, None, None

    sample_weights = compute_sample_weight("balanced", y_mapped)
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
    model.fit(X, y_mapped, sample_weight=sample_weights, verbose=False)

    return model, feature_cols, x_points, label_map


def predict_day(model, feature_cols, x_points, test_candles, instrument="nifty"):
    """Run predictions on a single day's candles. Returns list of prediction dicts."""
    df, _ = build_targets(test_candles.copy(), x_points=x_points, instrument=instrument)
    df = add_lag_features(df)

    if df.empty:
        return []

    X = df[feature_cols].values
    y_true = df["target"].values
    spots = df["spot_close"].values
    timestamps = df.index

    probas = model.predict_proba(X)
    inv = {0: -1, 1: 0, 2: 1}

    results = []
    for i in range(len(X)):
        pred_class = np.argmax(probas[i])
        confidence = probas[i][pred_class]
        signal = inv[pred_class]

        if confidence < PROBA_THRESHOLD:
            signal = 0

        results.append({
            "timestamp": timestamps[i],
            "spot": spots[i],
            "signal": signal,
            "actual": y_true[i],
            "confidence": confidence,
            "p_bearish": probas[i][0],
            "p_no_move": probas[i][1],
            "p_bullish": probas[i][2],
        })

    return results


def simulate_ml_pnl(predictions, x_points, candle_minutes=15,
                     delta=0.3, lot_size=65, num_lots=2,
                     stop_pts=15, target_pts=25, slippage_pts=2,
                     hold_minutes=45, min_conf=0.55):
    """
    Strict fixed SL/TP simulation on option value.

    - Entry: buy option at market when signal fires
    - SL: option drops stop_pts from entry → exit
    - TP: option gains target_pts from entry → exit
    - TIME: exit at hold_minutes if neither hit
    - No trailing, no scale-out, no breakeven moves.

    Option move estimated as: nifty_move * delta
    So SL in nifty terms = stop_pts / delta, TP = target_pts / delta
    """
    max_candles = max(2, hold_minutes // candle_minutes)
    qty = lot_size * num_lots
    trades = []
    cooldown = 0

    for i, pred in enumerate(predictions):
        if cooldown > 0:
            cooldown -= 1
            continue

        if pred["signal"] == 0:
            continue

        # Confidence filter
        if pred["confidence"] < min_conf:
            continue

        # No trades after cutoff (e.g. 15:15) — avoid EOD chop
        ts = pred["timestamp"]
        if hasattr(ts, 'hour'):
            if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 15):
                continue

        direction = "LONG" if pred["signal"] == 1 else "SHORT"
        entry_spot = pred["spot"]

        nifty_stop = stop_pts / delta     # ~50 pts nifty
        nifty_target = target_pts / delta  # ~83 pts nifty

        future = predictions[i + 1:]
        candles_held = 0
        exit_reason = ""
        option_pnl = 0.0
        peak_move = 0.0

        for j, fp in enumerate(future):
            candles_held = j + 1
            move = fp["spot"] - entry_spot
            if direction == "SHORT":
                move = -move

            if move > peak_move:
                peak_move = move

            # Strict SL
            if move <= -nifty_stop:
                option_pnl = -stop_pts
                exit_reason = "SL"
                break

            # Strict TP
            if move >= nifty_target:
                option_pnl = target_pts
                exit_reason = "TP"
                break

            # Time limit
            if candles_held >= max_candles:
                option_pnl = move * delta
                exit_reason = "TIME"
                break
        else:
            # End of data
            option_pnl = (move if future else 0) * delta
            exit_reason = "EOD"

        # Apply slippage (entry + exit)
        net_pnl = round((option_pnl - slippage_pts) * qty)

        cooldown = max(1, candles_held)

        trades.append({
            "timestamp": pred["timestamp"],
            "direction": direction,
            "entry_spot": entry_spot,
            "confidence": pred["confidence"],
            "signal_correct": (pred["signal"] == pred["actual"]),
            "pnl": net_pnl,
            "option_pnl": round(option_pnl, 1),
            "exit_reason": exit_reason,
            "peak_pts": round(peak_move, 1),
            "candles_held": candles_held,
        })

    return trades


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="nifty")
    parser.add_argument("--candle", type=int, default=15,
                        help="Candle timeframe in minutes (e.g. 5, 10, 15)")
    parser.add_argument("--sl", type=int, default=15,
                        help="Stop loss in option points")
    parser.add_argument("--tp", type=int, default=25,
                        help="Take profit in option points")
    parser.add_argument("--hold", type=int, default=None,
                        help="Max hold time in minutes (default: 45)")
    parser.add_argument("--min-conf", type=float, default=0.55,
                        help="Minimum confidence to take trade (default: 0.55)")
    parser.add_argument("--warm-up", type=int, default=5,
                        help="Number of initial days for training before testing starts")
    parser.add_argument("--retrain-every", type=int, default=3,
                        help="Retrain model every N days (expanding window)")
    args = parser.parse_args()

    instrument = args.instrument.lower()
    candle_minutes = args.candle
    stop_pts = args.sl
    target_pts = args.tp
    hold_minutes = args.hold if args.hold else 45
    min_conf = args.min_conf
    warm_up = args.warm_up
    retrain_every = args.retrain_every

    print(f"\n{'='*70}")
    print(f"  ML WALK-FORWARD BACKTEST — {instrument.upper()}")
    print(f"  Candle: {candle_minutes}-min | SL: {stop_pts}pts | TP: {target_pts}pts | Hold: {hold_minutes}min | MinConf: {min_conf}")
    print(f"  Warm-up: {warm_up} days | Retrain every: {retrain_every} days")
    print(f"  Confidence threshold: {PROBA_THRESHOLD}")
    print(f"{'='*70}")

    day_data = load_all_days(instrument, candle_minutes=candle_minutes)
    if len(day_data) <= warm_up:
        print(f"Need more than {warm_up} days of data. Have {len(day_data)}.")
        sys.exit(1)

    # Aggregate results
    all_predictions = []
    all_trades = []
    day_results = []
    model = None
    feature_cols = None
    x_points = None

    for day_idx in range(len(day_data)):
        day = day_data[day_idx]
        date = day["date"]

        if day_idx < warm_up:
            print(f"  {date}: warm-up (training data)")
            continue

        # (Re)train if needed
        if model is None or (day_idx - warm_up) % retrain_every == 0:
            # Build training set from all days up to (not including) today
            train_frames = [d["candles"] for d in day_data[:day_idx]]
            train_df = pd.concat(train_frames)
            n_train_days = day_idx
            n_train_candles = len(train_df)

            model, feature_cols, x_points, label_map = train_model(train_df, instrument)
            if model is None:
                print(f"  {date}: training failed (not enough data/classes)")
                continue
            print(f"  Trained on {n_train_days} days ({n_train_candles} candles), X={x_points:.1f}pts")

        # Predict on today
        preds = predict_day(model, feature_cols, x_points, day["candles"], instrument)
        if not preds:
            print(f"  {date}: no valid candles for prediction")
            continue

        # Score
        signals = [p for p in preds if p["signal"] != 0]
        correct_signals = [p for p in signals if p["signal"] == p["actual"]]
        all_preds_accuracy = sum(1 for p in preds if p["signal"] == p["actual"]) / len(preds) if preds else 0

        # P&L
        trades = simulate_ml_pnl(preds, x_points, candle_minutes=candle_minutes,
                                 stop_pts=stop_pts, target_pts=target_pts,
                                 hold_minutes=hold_minutes, min_conf=min_conf)
        day_pnl = sum(t["pnl"] for t in trades)

        day_result = {
            "date": date,
            "total_candles": len(preds),
            "signals_fired": len(signals),
            "signals_correct": len(correct_signals),
            "signal_precision": len(correct_signals) / len(signals) if signals else 0,
            "overall_accuracy": all_preds_accuracy,
            "trades": len(trades),
            "wins": sum(1 for t in trades if t["pnl"] > 0),
            "pnl": day_pnl,
        }
        day_results.append(day_result)
        all_predictions.extend(preds)
        all_trades.extend(trades)

        # Print day summary
        sig_str = f"{len(correct_signals)}/{len(signals)}" if signals else "0/0"
        pnl_color = Fore.GREEN if day_pnl >= 0 else Fore.RED
        print(f"  {date}:  signals={sig_str:>5s}  "
              f"precision={day_result['signal_precision']:>5.0%}  "
              f"trades={len(trades)}  "
              f"{pnl_color}P&L=Rs {day_pnl:>+,}{Style.RESET_ALL}")

        for t in trades:
            tc = Fore.GREEN if t["pnl"] > 0 else Fore.RED
            sig_tag = "OK" if t["signal_correct"] else "WRONG"
            print(f"           {str(t['timestamp']).split(' ')[-1][:5]}  "
                  f"{t['direction']:>5s}  conf={t['confidence']:.2f}  "
                  f"peak={t['peak_pts']:+.0f}  "
                  f"{t['exit_reason']:>4s}  opt={t['option_pnl']:+.1f}  "
                  f"{tc}Rs{t['pnl']:>+,}{Style.RESET_ALL}  [{sig_tag}]")

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {instrument.upper()} ML BACKTEST")
    print(f"{'='*70}")

    if not day_results:
        print("  No test days produced results.")
        return

    total_signals = sum(d["signals_fired"] for d in day_results)
    total_correct = sum(d["signals_correct"] for d in day_results)
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_tp = sum(1 for t in all_trades if t["exit_reason"] == "TP")
    total_sl = sum(1 for t in all_trades if t["exit_reason"] == "SL")
    total_time = sum(1 for t in all_trades if t["exit_reason"] in ("TIME", "EOD"))

    # Day-by-day P&L table
    print(f"\n  {'Date':<12s} {'Signals':>8s} {'Precision':>10s} {'Trades':>7s} {'Wins':>5s} {'P&L':>12s}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*7} {'─'*5} {'─'*12}")
    cum_pnl = 0
    for d in day_results:
        cum_pnl += d["pnl"]
        pnl_color = Fore.GREEN if d["pnl"] >= 0 else Fore.RED
        print(f"  {d['date']:<12s} "
              f"{d['signals_correct']}/{d['signals_fired']:>3d}     "
              f"{d['signal_precision']:>8.0%}   "
              f"{d['trades']:>5d}   "
              f"{d['wins']:>3d}   "
              f"{pnl_color}Rs {d['pnl']:>+,}{Style.RESET_ALL}")

    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*7} {'─'*5} {'─'*12}")

    # Totals
    overall_precision = total_correct / total_signals if total_signals else 0
    win_rate = total_wins / total_trades if total_trades else 0
    avg_pnl = total_pnl / total_trades if total_trades else 0
    profitable_days = sum(1 for d in day_results if d["pnl"] > 0)
    test_days = len(day_results)

    print(f"\n  Test days:         {test_days}")
    print(f"  Profitable days:   {profitable_days}/{test_days} ({profitable_days/test_days*100:.0f}%)")
    print(f"  Total signals:     {total_signals}")
    print(f"  Signal precision:  {overall_precision:.0%} ({total_correct}/{total_signals})")
    print(f"  Total trades:      {total_trades}")
    print(f"  Exit breakdown:    TP={total_tp}  SL={total_sl}  TIME/EOD={total_time}")
    print(f"  Win rate:          {win_rate:.0%} ({total_wins}/{total_trades})")
    print(f"  Avg P&L/trade:     Rs {avg_pnl:+,.0f}")
    if total_tp + total_sl > 0:
        print(f"  TP hit rate:       {total_tp/(total_tp+total_sl)*100:.0f}% (of TP+SL exits)")

    pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(f"\n  {pnl_color}NET P&L:  Rs {total_pnl:+,}{Style.RESET_ALL}")

    # Equity curve
    if all_trades:
        print(f"\n  Equity curve (cumulative):")
        cum = 0
        for t in all_trades:
            cum += t["pnl"]
            bar_len = int(abs(cum) / 500)
            bar_char = "█" if cum >= 0 else "░"
            color = Fore.GREEN if cum >= 0 else Fore.RED
            ts_str = str(t["timestamp"]).split(" ")[-1][:5]
            print(f"    {str(t['timestamp']).split(' ')[0]} {ts_str}  "
                  f"{color}{'':>1}{bar_char * min(bar_len, 40)} Rs {cum:>+,}{Style.RESET_ALL}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
