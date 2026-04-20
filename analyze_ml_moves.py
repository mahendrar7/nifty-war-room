"""
analyze_ml_moves.py — Measure actual option value movement after ML signal fires.

For each signal, tracks:
  - Peak favorable nifty move before reversal
  - Peak favorable option move (nifty * delta)
  - How many candles to reach peak
  - Max adverse move before peak

Usage:
    python analyze_ml_moves.py
"""

import numpy as np
from colorama import Fore, Style
from backtest_ml import load_all_days, train_model, predict_day
from ml_engine import PROBA_THRESHOLD

DELTA = 0.3
CANDLE_MINUTES = 5


def analyze_moves():
    day_data = load_all_days("nifty", candle_minutes=CANDLE_MINUTES)
    warm_up = 5
    retrain_every = 3

    model = None
    feature_cols = None
    x_points = None

    all_moves = []       # every signal
    correct_moves = []   # signal matched actual direction
    wrong_moves = []

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

            # Track move candle by candle until it reverses from peak
            peak_move = 0.0
            peak_candle = 0
            max_adverse = 0.0
            reversal_move = 0.0
            moves_by_candle = []

            for j, fp in enumerate(future):
                move = fp["spot"] - entry_spot
                if direction == "SHORT":
                    move = -move

                moves_by_candle.append(move)

                if move > peak_move:
                    peak_move = move
                    peak_candle = j + 1

                if move < 0 and move < max_adverse:
                    max_adverse = move

                # Stop tracking after reversal: price pulled back 50%+ from peak
                # or 15 candles (75 min) — enough to see the full move
                if peak_move > 10 and move < peak_move * 0.5:
                    reversal_move = move
                    break

                if j >= 14:  # 15 candles max lookahead
                    reversal_move = move
                    break

            peak_option = round(peak_move * DELTA, 1)
            adverse_option = round(max_adverse * DELTA, 1)

            entry = {
                "timestamp": ts,
                "direction": direction,
                "confidence": pred["confidence"],
                "signal_correct": pred["signal"] == pred["actual"],
                "peak_nifty": round(peak_move, 1),
                "peak_option": peak_option,
                "peak_candle": peak_candle,
                "max_adverse_nifty": round(max_adverse, 1),
                "max_adverse_option": adverse_option,
                "candles_tracked": len(moves_by_candle),
            }

            all_moves.append(entry)
            if pred["signal"] == pred["actual"]:
                correct_moves.append(entry)
            else:
                wrong_moves.append(entry)

            cooldown = max(1, peak_candle)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ML SIGNAL MOVE ANALYSIS — 5-min candles, delta={DELTA}")
    print(f"{'='*70}")

    for label, moves in [("ALL SIGNALS", all_moves),
                         ("CORRECT SIGNALS", correct_moves),
                         ("WRONG SIGNALS", wrong_moves)]:
        if not moves:
            continue

        peaks_opt = [m["peak_option"] for m in moves]
        peaks_nifty = [m["peak_nifty"] for m in moves]
        adverse_opt = [m["max_adverse_option"] for m in moves]
        peak_candles = [m["peak_candle"] for m in moves]

        print(f"\n  {label} ({len(moves)} trades)")
        print(f"  {'─'*60}")

        # Option move stats
        print(f"  Peak favorable option move before reversal:")
        print(f"    Mean:     {np.mean(peaks_opt):>+6.1f} pts")
        print(f"    Median:   {np.median(peaks_opt):>+6.1f} pts")
        print(f"    P25:      {np.percentile(peaks_opt, 25):>+6.1f} pts")
        print(f"    P75:      {np.percentile(peaks_opt, 75):>+6.1f} pts")
        print(f"    Max:      {np.max(peaks_opt):>+6.1f} pts")

        # Nifty move stats
        print(f"  Peak favorable nifty move:")
        print(f"    Mean:     {np.mean(peaks_nifty):>+6.1f} pts")
        print(f"    Median:   {np.median(peaks_nifty):>+6.1f} pts")

        # Time to peak
        print(f"  Candles to peak ({CANDLE_MINUTES}min each):")
        print(f"    Mean:     {np.mean(peak_candles):>5.1f} candles ({np.mean(peak_candles)*CANDLE_MINUTES:.0f} min)")
        print(f"    Median:   {np.median(peak_candles):>5.1f} candles ({np.median(peak_candles)*CANDLE_MINUTES:.0f} min)")

        # Adverse move before peak
        print(f"  Max adverse option move (drawdown before peak):")
        print(f"    Mean:     {np.mean(adverse_opt):>+6.1f} pts")
        print(f"    Median:   {np.median(adverse_opt):>+6.1f} pts")
        print(f"    P75:      {np.percentile(adverse_opt, 75):>+6.1f} pts  (75% of trades see this or less)")
        print(f"    Worst:    {np.min(adverse_opt):>+6.1f} pts")

        # Distribution buckets for peak option move
        buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 999)]
        print(f"\n  Peak option move distribution:")
        for lo, hi in buckets:
            count = sum(1 for p in peaks_opt if lo <= p < hi)
            pct = count / len(peaks_opt) * 100
            bar = "█" * int(pct / 2)
            label_str = f"{lo}-{hi}pts" if hi < 999 else f"{lo}+pts"
            print(f"    {label_str:<10s} {count:>4d} ({pct:>4.0f}%) {bar}")

    # ── Confidence breakdown ─────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  BY CONFIDENCE LEVEL:")
    conf_buckets = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
    for lo, hi in conf_buckets:
        subset = [m for m in all_moves if lo <= m["confidence"] < hi]
        if not subset:
            continue
        peaks = [m["peak_option"] for m in subset]
        correct = sum(1 for m in subset if m["signal_correct"])
        print(f"    conf {lo:.2f}-{hi:.2f}: {len(subset):>3d} trades  "
              f"median peak={np.median(peaks):>+5.1f}  "
              f"mean={np.mean(peaks):>+5.1f}  "
              f"accuracy={correct/len(subset)*100:.0f}%")

    # ── Suggested TP ─────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    correct_peaks = [m["peak_option"] for m in correct_moves]
    if correct_peaks:
        p25 = np.percentile(correct_peaks, 25)
        p50 = np.median(correct_peaks)
        print(f"  SUGGESTED TP LEVELS (based on correct signals):")
        print(f"    Conservative (P25): {p25:>+.0f} pts  — 75% of correct signals reach this")
        print(f"    Moderate (P50):     {p50:>+.0f} pts  — 50% of correct signals reach this")
        # What % of correct signals would hit various TP levels
        for tp in [5, 8, 10, 12, 15, 20, 25]:
            hit = sum(1 for p in correct_peaks if p >= tp)
            pct = hit / len(correct_peaks) * 100
            print(f"    TP={tp:>2d}pts: {hit:>3d}/{len(correct_peaks)} correct signals hit ({pct:.0f}%)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    analyze_moves()
