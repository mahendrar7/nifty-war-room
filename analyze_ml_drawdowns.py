"""
analyze_ml_drawdowns.py — Drawdown distribution after ML signal fires.

Finds the most common adverse move range to identify where noise ends
and real stops should begin.
"""

import numpy as np
from collections import Counter
from backtest_ml import load_all_days, train_model, predict_day
from ml_engine import PROBA_THRESHOLD

DELTA = 0.3
CANDLE_MINUTES = 5


def collect_drawdowns():
    day_data = load_all_days("nifty", candle_minutes=CANDLE_MINUTES)
    warm_up = 5
    retrain_every = 3

    model = None
    feature_cols = None
    x_points = None

    all_drawdowns = []      # max adverse option move per trade
    correct_drawdowns = []
    wrong_drawdowns = []

    for day_idx in range(len(day_data)):
        day = day_data[day_idx]
        if day_idx < warm_up:
            continue

        if model is None or (day_idx - warm_up) % retrain_every == 0:
            import pandas as pd
            train_frames = [d["candles"] for d in day_data[:day_idx]]
            train_df = pd.concat(train_frames)
            model, feature_cols, x_points, _ = train_model(train_df, "nifty")
            if model is None:
                continue

        preds = predict_day(model, feature_cols, x_points, day["candles"], "nifty")
        if not preds:
            continue

        cooldown = 0
        for i, pred in enumerate(preds):
            if cooldown > 0:
                cooldown -= 1
                continue
            if pred["signal"] == 0:
                continue

            ts = pred["timestamp"]
            if hasattr(ts, 'hour'):
                if ts.hour > 15 or (ts.hour == 15 and ts.minute >= 15):
                    continue

            direction = "LONG" if pred["signal"] == 1 else "SHORT"
            entry_spot = pred["spot"]
            future = preds[i + 1:]

            max_adverse_nifty = 0.0
            peak_move = 0.0

            for j, fp in enumerate(future):
                if j >= 9:  # 45 min lookahead
                    break
                move = fp["spot"] - entry_spot
                if direction == "SHORT":
                    move = -move

                if move < max_adverse_nifty:
                    max_adverse_nifty = move
                if move > peak_move:
                    peak_move = move

            adverse_option = abs(round(max_adverse_nifty * DELTA, 1))

            entry = {
                "adverse_option": adverse_option,
                "adverse_nifty": abs(round(max_adverse_nifty, 1)),
                "peak_option": round(peak_move * DELTA, 1),
                "correct": pred["signal"] == pred["actual"],
                "confidence": pred["confidence"],
            }
            all_drawdowns.append(entry)
            if entry["correct"]:
                correct_drawdowns.append(entry)
            else:
                wrong_drawdowns.append(entry)

            cooldown = 1

    return all_drawdowns, correct_drawdowns, wrong_drawdowns


def print_distribution(label, drawdowns):
    if not drawdowns:
        return

    vals = [d["adverse_option"] for d in drawdowns]

    print(f"\n  {label} ({len(drawdowns)} trades)")
    print(f"  {'─'*60}")

    # 1-point buckets for fine-grained view
    print(f"\n  Drawdown distribution (1pt buckets, option value):")
    max_bucket = int(max(vals)) + 1
    buckets = list(range(0, min(max_bucket + 1, 35)))

    counts = []
    for b in buckets:
        count = sum(1 for v in vals if b <= v < b + 1)
        counts.append((b, count))

    max_count = max(c for _, c in counts) if counts else 1
    for b, count in counts:
        pct = count / len(vals) * 100
        bar = "█" * int(count / max_count * 40) if count > 0 else ""
        print(f"    {b:>2d}-{b+1:<2d}pts: {count:>4d} ({pct:>5.1f}%) {bar}")

    # Cumulative — "X% of trades have drawdown <= N pts"
    print(f"\n  Cumulative (what % of trades survive each SL):")
    for sl in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        survived = sum(1 for v in vals if v <= sl)
        pct = survived / len(vals) * 100
        bar = "█" * int(pct / 2.5)
        print(f"    SL={sl:>2d}pts: {survived:>4d}/{len(vals)} survive ({pct:>5.1f}%) {bar}")

    # Stats
    print(f"\n  Stats:")
    print(f"    Mean:     {np.mean(vals):.1f} pts")
    print(f"    Median:   {np.median(vals):.1f} pts")
    print(f"    Mode rng: {Counter(int(v) for v in vals).most_common(3)}")
    print(f"    P75:      {np.percentile(vals, 75):.1f} pts")
    print(f"    P90:      {np.percentile(vals, 90):.1f} pts")


def main():
    all_dd, correct_dd, wrong_dd = collect_drawdowns()

    print(f"\n{'='*70}")
    print(f"  ML SIGNAL DRAWDOWN ANALYSIS — 5-min candles, delta={DELTA}")
    print(f"{'='*70}")

    print_distribution("ALL SIGNALS", all_dd)
    print_distribution("CORRECT SIGNALS (direction was right)", correct_dd)
    print_distribution("WRONG SIGNALS (direction was wrong)", wrong_dd)

    # Compare correct vs wrong — where do they separate?
    if correct_dd and wrong_dd:
        print(f"\n  {'─'*60}")
        print(f"  CORRECT vs WRONG — Survival rate by SL level:")
        print(f"  (shows what SL level best separates good from bad trades)\n")
        print(f"    {'SL':>4s}  {'Correct survive':>16s}  {'Wrong survive':>14s}  {'Edge':>6s}")
        print(f"    {'─'*4}  {'─'*16}  {'─'*14}  {'─'*6}")
        correct_vals = [d["adverse_option"] for d in correct_dd]
        wrong_vals = [d["adverse_option"] for d in wrong_dd]
        for sl in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            c_surv = sum(1 for v in correct_vals if v <= sl) / len(correct_vals) * 100
            w_surv = sum(1 for v in wrong_vals if v <= sl) / len(wrong_vals) * 100
            edge = c_surv - w_surv
            marker = " ◄◄" if edge >= 15 else ""
            print(f"    {sl:>2d}pts  {c_surv:>14.1f}%  {w_surv:>12.1f}%  {edge:>+5.1f}%{marker}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
