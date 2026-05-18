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
from datetime import datetime
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
    files = sorted(glob.glob(pattern),
                   key=lambda f: datetime.strptime(
                       os.path.basename(f).split("_")[-1].replace(".csv", ""),
                       "%d%m%Y"))
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


LOT_SIZES = {"nifty": 65, "sensex": 20}


def simulate_ml_pnl(predictions, x_points, candle_minutes=15,
                     delta=0.3, lot_size=65, num_lots=2,
                     stop_pts=15, target_pts=25, slippage_pts=2,
                     hold_minutes=45, min_conf=0.55, runner=False,
                     runner_ext=1.3, runner_be=False, direction_filter=None,
                     long_conf=None, retracement_pct=None, lot2_entry_buffer=None):
    """
    Strict fixed SL/TP simulation on option value.

    runner=True: two-lot mode.
      Lot 1 exits at TP. Lot 2 rides with TP as trailing floor, exits at
      TP*1.3 (extended target) or when price drops back to TP, whichever first.
      Both lots share the same SL.
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

        direction = "LONG" if pred["signal"] == 1 else "SHORT"
        threshold = long_conf if (long_conf is not None and direction == "LONG") else min_conf
        if pred["confidence"] < threshold:
            continue

        ts = pred["timestamp"]
        if hasattr(ts, 'hour'):
            if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 15):
                continue
            if ts.hour == 9 and ts.minute < 30:
                continue

        if direction_filter and direction != direction_filter:
            continue
        entry_spot = pred["spot"]

        nifty_stop = stop_pts / delta
        nifty_target = target_pts / delta
        lot2_target_pts = round(target_pts * runner_ext, 1)
        lot2_nifty_target = lot2_target_pts / delta

        future = predictions[i + 1:]
        candles_held = 0
        exit_reason = ""
        option_pnl = 0.0
        peak_move = 0.0
        worst_move = 0.0

        # Runner-mode per-lot tracking
        lot1_pnl = None
        lot2_pnl = None
        lot1_exit = None
        lot2_exit = None
        lot1_tp_candle = -1
        lot2_peak = 0.0  # tracks peak spot move after lot1 exits (for retracement trail)

        for j, fp in enumerate(future):
            candles_held = j + 1
            move = fp["spot"] - entry_spot
            if direction == "SHORT":
                move = -move

            if move > peak_move:
                peak_move = move
            if move < worst_move:
                worst_move = move

            if runner:
                if lot1_pnl is None:
                    # Both lots open
                    if move <= -nifty_stop:
                        lot1_pnl = lot2_pnl = -stop_pts
                        lot1_exit = lot2_exit = "SL"
                        exit_reason = "SL"
                        break
                    if move >= nifty_target:
                        lot1_pnl = target_pts
                        lot1_exit = "TP"
                        lot1_tp_candle = j
                        lot2_peak = move  # seed lot2 peak at TP exit point
                        # lot 2 continues — don't break
                    if lot1_pnl is None and candles_held >= max_candles:
                        opt_val = move * delta
                        lot1_pnl = lot2_pnl = opt_val
                        lot1_exit = lot2_exit = "TIME"
                        exit_reason = "TIME"
                        break

                if lot1_pnl is not None and lot2_pnl is None and j > lot1_tp_candle:
                    # Lot 2 running — update peak for trailing stop
                    if move > lot2_peak:
                        lot2_peak = move
                    # Resolve floor in spot and option pts
                    if lot2_entry_buffer is not None:
                        floor_spot = lot2_entry_buffer / delta
                        floor_opt  = lot2_entry_buffer
                        floor_label_base = f"ENTRY+{lot2_entry_buffer}pt"
                    elif runner_be:
                        floor_spot = 0.0
                        floor_opt  = 0.0
                        floor_label_base = "BE"
                    else:
                        floor_spot = nifty_target
                        floor_opt  = target_pts
                        floor_label_base = "TP_TRAIL"
                    # Compute floor: retracement trail or fixed floor
                    if retracement_pct and lot2_peak > floor_spot:
                        trail_spot = lot2_peak * (1.0 - retracement_pct)
                        lot2_floor = max(floor_spot, trail_spot)
                        lot2_floor_pts = lot2_floor * delta
                        lot2_floor_label = f"TRAIL{int(retracement_pct*100)}%"
                    else:
                        lot2_floor = floor_spot
                        lot2_floor_pts = floor_opt
                        lot2_floor_label = floor_label_base
                    if move <= lot2_floor:
                        lot2_pnl = lot2_floor_pts
                        lot2_exit = lot2_floor_label
                        exit_reason = lot2_floor_label
                        break
                    if move >= lot2_nifty_target:
                        lot2_pnl = lot2_target_pts
                        lot2_exit = "TP2"
                        exit_reason = "TP2"
                        break
                    if candles_held >= max_candles:
                        opt_val = move * delta
                        lot2_pnl = opt_val
                        lot2_exit = "TIME"
                        exit_reason = "TIME"
                        break
            else:
                if move <= -nifty_stop:
                    option_pnl = -stop_pts
                    exit_reason = "SL"
                    break
                if move >= nifty_target:
                    option_pnl = target_pts
                    exit_reason = "TP"
                    break
                if candles_held >= max_candles:
                    option_pnl = move * delta
                    exit_reason = "TIME"
                    break
        else:
            opt_val = (move if future else 0) * delta
            if runner:
                if lot1_pnl is None:
                    lot1_pnl = lot2_pnl = opt_val
                    lot1_exit = lot2_exit = "EOD"
                else:
                    lot2_pnl = opt_val
                    lot2_exit = "EOD"
                exit_reason = "EOD"
            else:
                option_pnl = opt_val
                exit_reason = "EOD"

        # Edge: lot1 hit TP but loop ended before lot2 found an exit
        if runner and lot1_pnl is not None and lot2_pnl is None:
            lot2_pnl = target_pts
            lot2_exit = "EOD"
            exit_reason = "EOD"

        if runner:
            qty_per_side = lot_size * max(1, num_lots // 2)
            net_pnl = (round((lot1_pnl - slippage_pts) * qty_per_side) +
                       round((lot2_pnl - slippage_pts) * qty_per_side))
            option_pnl = round((lot1_pnl + lot2_pnl) / 2, 1)
        else:
            net_pnl = round((option_pnl - slippage_pts) * qty)

        cooldown = max(1, candles_held)

        mfe_pts = round(peak_move * delta, 1)
        mae_pts = round(worst_move * delta, 1)

        trade = {
            "timestamp": pred["timestamp"],
            "direction": direction,
            "entry_spot": entry_spot,
            "confidence": pred["confidence"],
            "signal_correct": (pred["signal"] == pred["actual"]),
            "pnl": net_pnl,
            "option_pnl": round(option_pnl, 1),
            "exit_reason": exit_reason,
            "peak_pts": round(peak_move, 1),
            "mfe_pts": mfe_pts,
            "mae_pts": mae_pts,
            "candles_held": candles_held,
        }
        if runner:
            trade["lot1_pnl"] = round(lot1_pnl, 1)
            trade["lot2_pnl"] = round(lot2_pnl, 1)
            trade["lot1_exit"] = lot1_exit
            trade["lot2_exit"] = lot2_exit

        trades.append(trade)

    return trades


def _print_summary(instrument, day_results, all_trades, runner):
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
    total_tp2 = sum(1 for t in all_trades if t.get("lot2_exit") == "TP2")
    total_tp_trail = sum(1 for t in all_trades if t.get("lot2_exit") == "TP_TRAIL")
    total_trail_pct = sum(1 for t in all_trades if t.get("lot2_exit", "").startswith("TRAIL") and t.get("lot2_exit") != "TP_TRAIL")
    total_lot1_tp = sum(1 for t in all_trades if t.get("lot1_exit") == "TP")

    # Direction breakdown
    longs  = [t for t in all_trades if t["direction"] == "LONG"]
    shorts = [t for t in all_trades if t["direction"] == "SHORT"]
    long_sl  = sum(1 for t in longs  if t["exit_reason"] == "SL")
    short_sl = sum(1 for t in shorts if t["exit_reason"] == "SL")

    print(f"\n  {'Date':<12s} {'Signals':>8s} {'Precision':>10s} {'Trades':>7s} {'Wins':>5s} {'P&L':>12s}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*7} {'─'*5} {'─'*12}")
    for d in day_results:
        pnl_color = Fore.GREEN if d["pnl"] >= 0 else Fore.RED
        print(f"  {d['date']:<12s} "
              f"{d['signals_correct']}/{d['signals_fired']:>3d}     "
              f"{d['signal_precision']:>8.0%}   "
              f"{d['trades']:>5d}   "
              f"{d['wins']:>3d}   "
              f"{pnl_color}Rs {d['pnl']:>+,}{Style.RESET_ALL}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*7} {'─'*5} {'─'*12}")

    overall_precision = total_correct / total_signals if total_signals else 0
    win_rate = total_wins / total_trades if total_trades else 0
    avg_pnl = total_pnl / total_trades if total_trades else 0
    profitable_days = sum(1 for d in day_results if d["pnl"] > 0)
    test_days = len(day_results)

    print(f"\n  Test days:         {test_days}")
    print(f"  Profitable days:   {profitable_days}/{test_days} ({profitable_days/test_days*100:.0f}%)")
    print(f"  Total signals:     {total_signals}")
    print(f"  Signal precision:  {overall_precision:.0%} ({total_correct}/{total_signals})")
    print(f"  Total trades:      {total_trades}  (LONG={len(longs)}  SHORT={len(shorts)})")
    if runner:
        trail_str = f"  TP_TRAIL={total_tp_trail}"
        if total_trail_pct:
            trail_str += f"  PCT_TRAIL={total_trail_pct}"
        print(f"  Exit breakdown:    SL={total_sl}  TP2={total_tp2}{trail_str}  TIME/EOD={total_time}")
        print(f"  Lot1 TP hits:      {total_lot1_tp}/{total_trades}")
    else:
        print(f"  Exit breakdown:    TP={total_tp}  SL={total_sl}  TIME/EOD={total_time}")
    print(f"  SL breakdown:      LONG SL={long_sl} ({long_sl/len(longs)*100:.0f}%)  SHORT SL={short_sl} ({short_sl/len(shorts)*100:.0f}%)" if longs and shorts else "")
    print(f"  Win rate:          {win_rate:.0%} ({total_wins}/{total_trades})")
    print(f"  Avg P&L/trade:     Rs {avg_pnl:+,.0f}")
    if not runner and total_tp + total_sl > 0:
        print(f"  TP hit rate:       {total_tp/(total_tp+total_sl)*100:.0f}% (of TP+SL exits)")

    if all_trades:
        avg_mfe = sum(t["mfe_pts"] for t in all_trades) / total_trades
        avg_mae = sum(t["mae_pts"] for t in all_trades) / total_trades
        print(f"  Avg MFE (opt pts): {avg_mfe:+.1f}")
        print(f"  Avg MAE (opt pts): {avg_mae:+.1f}")

    pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(f"\n  {pnl_color}NET P&L:  Rs {total_pnl:+,}{Style.RESET_ALL}")

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
    parser.add_argument("--runner", action="store_true",
                        help="Two-lot runner: lot1 exits at TP, lot2 rides to TP*ext or drops back to TP")
    parser.add_argument("--runner-ext", type=float, default=1.3,
                        help="Lot2 target multiplier on TP (default: 1.3 = TP*1.3)")
    parser.add_argument("--runner-be", action="store_true",
                        help="Lot2 floor moves to entry (breakeven) instead of TP after lot1 exits")
    parser.add_argument("--lots", type=int, default=2,
                        help="Total number of lots (split evenly between lot1/lot2 in runner mode)")
    parser.add_argument("--direction", choices=["LONG", "SHORT"], default=None,
                        help="Only take signals in this direction (LONG=calls, SHORT=puts)")
    parser.add_argument("--long-conf", type=float, default=None,
                        help="Override min-conf for LONG signals only (hybrid mode)")
    parser.add_argument("--last-n-days", type=int, default=None,
                        help="Validation mode: train on all days before last N, test only last N")
    parser.add_argument("--slippage", type=float, default=2.0,
                        help="Slippage in option points per side (default: 2.0)")
    parser.add_argument("--retracement", type=float, default=None,
                        help="Lot2 trailing stop: exit when price retraces this fraction from peak (e.g. 0.30 = 30%%)")
    parser.add_argument("--lot2-floor", type=float, default=None, dest="lot2_floor",
                        help="Lot2 SL floor in option pts above entry after lot1 TP (e.g. 10)")
    args = parser.parse_args()

    instrument = args.instrument.lower()
    candle_minutes = args.candle
    stop_pts = args.sl
    target_pts = args.tp
    hold_minutes = args.hold if args.hold else 45
    min_conf = args.min_conf
    warm_up = args.warm_up
    retrain_every = args.retrain_every
    lot_size = LOT_SIZES.get(instrument, 65)
    num_lots = args.lots
    runner = args.runner
    runner_ext = args.runner_ext
    runner_be = args.runner_be
    direction_filter = args.direction
    long_conf = args.long_conf
    last_n_days = args.last_n_days
    slippage_pts = args.slippage
    retracement_pct = args.retracement
    lot2_floor = args.lot2_floor

    print(f"\n{'='*70}")
    print(f"  ML WALK-FORWARD BACKTEST — {instrument.upper()}")
    floor_label = "BE(entry)" if runner_be else f"TP_TRAIL({target_pts}pts)"
    mode_str = f"  Mode: TWO-LOT RUNNER (lot1→TP={target_pts}, lot2→TP*{runner_ext}={round(target_pts*runner_ext,1)} or {floor_label})" if runner else ""
    if mode_str:
        print(mode_str)
    if last_n_days:
        print(f"  Validation mode: testing last {last_n_days} days only")
    print(f"  Candle: {candle_minutes}-min | SL: {stop_pts}pts | TP: {target_pts}pts | Hold: {hold_minutes}min | MinConf: {min_conf}")
    if long_conf:
        print(f"  Hybrid conf: SHORT≥{min_conf}  LONG≥{long_conf}")
    print(f"  Confidence threshold: {PROBA_THRESHOLD}")
    print(f"{'='*70}")

    day_data = load_all_days(instrument, candle_minutes=candle_minutes)
    if len(day_data) <= warm_up:
        print(f"Need more than {warm_up} days of data. Have {len(day_data)}.")
        sys.exit(1)

    # In validation mode: train on everything before last N days, test only last N
    if last_n_days:
        if len(day_data) <= last_n_days:
            print(f"Not enough days: have {len(day_data)}, need more than {last_n_days}.")
            sys.exit(1)
        split = len(day_data) - last_n_days
        train_df = pd.concat([d["candles"] for d in day_data[:split]])
        val_model, val_features, val_xpts, _ = train_model(train_df, instrument)
        if val_model is None:
            print("Training failed.")
            sys.exit(1)
        print(f"  Trained on {split} days. Testing on last {last_n_days} days.\n")
        all_trades = []
        day_results = []
        for day in day_data[split:]:
            preds = predict_day(val_model, val_features, val_xpts, day["candles"], instrument)
            if not preds:
                continue
            signals = [p for p in preds if p["signal"] != 0]
            correct  = [p for p in signals if p["signal"] == p["actual"]]
            trades = simulate_ml_pnl(preds, val_xpts, candle_minutes=candle_minutes,
                                     lot_size=lot_size, num_lots=num_lots,
                                     stop_pts=stop_pts, target_pts=target_pts,
                                     slippage_pts=slippage_pts,
                                     hold_minutes=hold_minutes, min_conf=min_conf,
                                     runner=runner, runner_ext=runner_ext,
                                     runner_be=runner_be, direction_filter=direction_filter,
                                     long_conf=long_conf, retracement_pct=retracement_pct,
                                     lot2_entry_buffer=lot2_floor)
            day_pnl = sum(t["pnl"] for t in trades)
            day_results.append({
                "date": day["date"], "signals_fired": len(signals),
                "signals_correct": len(correct),
                "signal_precision": len(correct)/len(signals) if signals else 0,
                "trades": len(trades), "wins": sum(1 for t in trades if t["pnl"] > 0),
                "pnl": day_pnl,
            })
            all_trades.extend(trades)
            sig_str = f"{len(correct)}/{len(signals)}"
            pnl_color = Fore.GREEN if day_pnl >= 0 else Fore.RED
            print(f"  {day['date']}:  signals={sig_str:>5s}  "
                  f"precision={day_results[-1]['signal_precision']:>5.0%}  "
                  f"trades={len(trades)}  "
                  f"{pnl_color}P&L=Rs {day_pnl:>+,}{Style.RESET_ALL}")
            for t in trades:
                tc = Fore.GREEN if t["pnl"] > 0 else Fore.RED
                sig_tag = "OK" if t["signal_correct"] else "WRONG"
                if "lot1_pnl" in t:
                    print(f"           {str(t['timestamp']).split(' ')[-1][:5]}  "
                          f"{t['direction']:>5s}  conf={t['confidence']:.2f}  "
                          f"MFE={t['mfe_pts']:+.1f}  MAE={t['mae_pts']:+.1f}  "
                          f"L1={t['lot1_pnl']:+.0f}({t['lot1_exit']})  "
                          f"L2={t['lot2_pnl']:+.0f}({t['lot2_exit']})  "
                          f"{tc}Rs{t['pnl']:>+,}{Style.RESET_ALL}  [{sig_tag}]")
                else:
                    print(f"           {str(t['timestamp']).split(' ')[-1][:5]}  "
                          f"{t['direction']:>5s}  conf={t['confidence']:.2f}  "
                          f"MFE={t['mfe_pts']:+.1f}  MAE={t['mae_pts']:+.1f}  "
                          f"{t['exit_reason']:>4s}  opt={t['option_pnl']:+.1f}  "
                          f"{tc}Rs{t['pnl']:>+,}{Style.RESET_ALL}  [{sig_tag}]")

        # Jump straight to summary using collected data
        _print_summary(instrument, day_results, all_trades, runner)
        return

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
                                 lot_size=lot_size, num_lots=num_lots,
                                 stop_pts=stop_pts, target_pts=target_pts,
                                 slippage_pts=slippage_pts,
                                 hold_minutes=hold_minutes, min_conf=min_conf,
                                 runner=runner, runner_ext=runner_ext,
                                 runner_be=runner_be, direction_filter=direction_filter,
                                 long_conf=long_conf, retracement_pct=retracement_pct)
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
            if "lot1_pnl" in t:
                print(f"           {str(t['timestamp']).split(' ')[-1][:5]}  "
                      f"{t['direction']:>5s}  conf={t['confidence']:.2f}  "
                      f"MFE={t['mfe_pts']:+.1f}  MAE={t['mae_pts']:+.1f}  "
                      f"L1={t['lot1_pnl']:+.0f}({t['lot1_exit']})  "
                      f"L2={t['lot2_pnl']:+.0f}({t['lot2_exit']})  "
                      f"{tc}Rs{t['pnl']:>+,}{Style.RESET_ALL}  [{sig_tag}]")
            else:
                print(f"           {str(t['timestamp']).split(' ')[-1][:5]}  "
                      f"{t['direction']:>5s}  conf={t['confidence']:.2f}  "
                      f"MFE={t['mfe_pts']:+.1f}  MAE={t['mae_pts']:+.1f}  "
                      f"{t['exit_reason']:>4s}  opt={t['option_pnl']:+.1f}  "
                      f"{tc}Rs{t['pnl']:>+,}{Style.RESET_ALL}  [{sig_tag}]")

    _print_summary(instrument, day_results, all_trades, runner)


if __name__ == "__main__":
    main()
