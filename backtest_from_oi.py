"""
backtest_from_oi.py — Backtest sniper from raw options_log CSV only.

Reconstructs all signals from per-strike OI data + spot/gamma/straddle.
Does NOT need a signals_log CSV. Works on any day with options_log_1min_*.csv.

Usage:
    python backtest_from_oi.py data/options_log_1min_nifty_18032026.csv
    python backtest_from_oi.py data/options_log_1min_nifty_*.csv   # all days

Does NOT modify any files.
"""

import csv
import sys
import os
from collections import deque

import pandas as pd
from colorama import Fore, Style

from config import (
    TREND_WINDOW_MINUTES, TREND_MIN_MOVE_MULT,
    SNIPER_TAKE_TRADE, SNIPER_SEND_IT,
    GAMMA_SIGMA, LOT_SIZE,
)
from gamma_engine import compute_gamma_pressure
from detectors import (
    detect_liquidity_vacuum, detect_wall_break_vacuum,
    detect_gamma_flip_breakout, detect_liquidity_acceleration,
    compute_move_probability,
)
from sniper_display import (
    _score_gamma, _score_gamma_momentum, _score_straddle, _score_spot_vs_walls,
    _score_oi_velocity, _score_iv, _score_move_prob,
    _score_structural, _score_trend,
    compute_gamma_momentum,
    _classify_setup, _resolve_direction, _decide_action,
    W,
)


def parse_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_day(options_path):
    """
    Load an options_log CSV and return per-timestamp snapshots.
    Each snapshot has: spot, gamma, straddle, bias, atm, days_to_expiry, df (strike-level OI)
    """
    with open(options_path, "r") as f:
        rows = list(csv.DictReader(f))

    by_ts = {}
    for row in rows:
        ts = row["timestamp"]
        by_ts.setdefault(ts, []).append(row)

    snapshots = []
    for ts in sorted(by_ts.keys()):
        ts_rows = by_ts[ts]
        first = ts_rows[0]

        strikes = {}
        for r in ts_rows:
            strike = float(r["strike"])
            oi = int(float(r.get("oi", 0)))
            otype = r["option_type"]
            if strike not in strikes:
                strikes[strike] = {"strike": strike, "call_oi": 0, "put_oi": 0}
            if otype == "CE":
                strikes[strike]["call_oi"] = oi
            else:
                strikes[strike]["put_oi"] = oi

        df = pd.DataFrame(list(strikes.values())).sort_values("strike").reset_index(drop=True)

        spot = float(first["spot"])
        atm = float(first["atm_strike"])

        # gamma_pressure: use CSV column if available, else compute from OI
        if "gamma_pressure" in first:
            gamma = float(first["gamma_pressure"])
        else:
            gamma = compute_gamma_pressure(df, spot)

        # straddle: use CSV column if available, else compute from ATM CE+PE LTP
        if "straddle" in first:
            straddle = float(first["straddle"])
        else:
            # Sum ATM CE and PE LTPs
            atm_ltp = 0.0
            for r in ts_rows:
                if float(r["strike"]) == atm:
                    atm_ltp += parse_float(r.get("ltp", 0))
            straddle = atm_ltp if atm_ltp > 0 else 200.0  # fallback

        snapshots.append({
            "timestamp": ts,
            "spot": spot,
            "gamma": gamma,
            "straddle": straddle,
            "bias": first.get("market_bias", "RANGE"),
            "atm": atm,
            "days_to_expiry": int(float(first.get("days_to_expiry", 4))),
            "df": df,
        })

    return snapshots


def compute_walls(df, spot):
    """Compute call/put walls from OI data."""
    above = df[df["strike"] >= spot]
    below = df[df["strike"] <= spot]

    call_wall = above.loc[above["call_oi"].idxmax(), "strike"] if not above.empty and above["call_oi"].max() > 0 else spot + 100
    put_wall = below.loc[below["put_oi"].idxmax(), "strike"] if not below.empty and below["put_oi"].max() > 0 else spot - 100

    return call_wall, put_wall


def compute_flip_level(df, spot):
    """Estimate flip level from OI — strike where call_oi ~ put_oi."""
    df = df.copy()
    df["diff"] = abs(df["call_oi"] - df["put_oi"])
    near = df[(df["strike"] >= spot - 300) & (df["strike"] <= spot + 300)]
    if near.empty:
        return None
    return near.loc[near["diff"].idxmin(), "strike"]


def recompute_trend(spot_history, spot, expected_move):
    if len(spot_history) < 10:
        return {"trending": False, "direction": None, "move_pts": 0, "duration_minutes": 0}
    window = list(spot_history)[-TREND_WINDOW_MINUTES:]
    move = spot - window[0]
    threshold = expected_move * TREND_MIN_MOVE_MULT
    if abs(move) >= threshold:
        return {"trending": True, "direction": "UP" if move > 0 else "DOWN",
                "move_pts": round(abs(move), 1), "duration_minutes": len(window)}
    return {"trending": False, "direction": None, "move_pts": 0, "duration_minutes": 0}


def identify_swings(spots, timestamps, min_pts=25):
    if len(spots) < 3:
        return []
    swings = []
    swing_start_idx = 0
    swing_high = spots[0]
    swing_low = spots[0]
    swing_high_idx = 0
    swing_low_idx = 0

    for i in range(1, len(spots)):
        if spots[i] > swing_high:
            swing_high = spots[i]
            swing_high_idx = i
        if spots[i] < swing_low:
            swing_low = spots[i]
            swing_low_idx = i

        if swing_high - spots[i] >= min_pts and swing_high - spots[swing_start_idx] >= min_pts:
            if swing_high_idx > swing_start_idx:
                swings.append({
                    "direction": "UP", "start_idx": swing_start_idx, "end_idx": swing_high_idx,
                    "start_time": timestamps[swing_start_idx], "end_time": timestamps[swing_high_idx],
                    "start_spot": spots[swing_start_idx], "end_spot": spots[swing_high_idx],
                    "pts": round(spots[swing_high_idx] - spots[swing_start_idx], 1),
                })
                swing_start_idx = swing_high_idx
                swing_low = spots[swing_high_idx]
                swing_low_idx = swing_high_idx

        if spots[i] - swing_low >= min_pts and spots[swing_start_idx] - swing_low >= min_pts:
            if swing_low_idx > swing_start_idx:
                swings.append({
                    "direction": "DOWN", "start_idx": swing_start_idx, "end_idx": swing_low_idx,
                    "start_time": timestamps[swing_start_idx], "end_time": timestamps[swing_low_idx],
                    "start_spot": spots[swing_start_idx], "end_spot": spots[swing_low_idx],
                    "pts": round(spots[swing_start_idx] - spots[swing_low_idx], 1),
                })
                swing_start_idx = swing_low_idx
                swing_high = spots[swing_low_idx]
                swing_high_idx = swing_low_idx

    return swings


def run_one_day(options_path, min_pts=25):
    """Run backtest for a single day. Returns summary dict."""
    snapshots = load_day(options_path)
    if not snapshots:
        return None

    date_label = os.path.basename(options_path)

    spot_history = deque(maxlen=TREND_WINDOW_MINUTES)
    gamma_history = deque(maxlen=10)
    straddle_history = deque(maxlen=10)
    oi_vel_history = deque(maxlen=30)
    prev_spot = None
    prev_df = None
    results = []
    spots = []
    timestamps = []

    for snap in snapshots:
        spot = snap["spot"]
        gamma = snap["gamma"]
        straddle = snap["straddle"]
        bias = snap["bias"]
        atm = snap["atm"]
        days_to_expiry = snap["days_to_expiry"]
        df = snap["df"]
        timestamp = snap["timestamp"].split(" ")[-1] if " " in snap["timestamp"] else snap["timestamp"]

        spots.append(spot)
        timestamps.append(timestamp)
        spot_history.append(spot)
        gamma_history.append(gamma)
        straddle_history.append(straddle)

        # Compute walls
        call_wall, put_wall = compute_walls(df, spot)

        # Compute flip level
        flip_level = compute_flip_level(df, spot)

        # Regime
        if gamma > 0 and abs(spot - atm) < 40:
            regime = "GAMMA PINNING"
        elif gamma < 0:
            regime = "VOLATILITY EXPANSION"
        else:
            regime = "NEUTRAL"

        # Straddle momentum
        if len(straddle_history) >= 6:
            prev_straddle = straddle_history[-6]
            mom_5m = ((straddle - prev_straddle) / (prev_straddle + 1e-9)) * 100
        else:
            mom_5m = 0.0

        if len(straddle_history) >= 10:
            prev_straddle_15 = straddle_history[0]
            mom_15m = ((straddle - prev_straddle_15) / (prev_straddle_15 + 1e-9)) * 100
        else:
            mom_15m = 0.0

        momentum_data = {
            "momentum_5m": mom_5m,
            "momentum_15m": mom_15m,
            "status": "FAST EXPANDING" if mom_5m > 3.0 else ("EXPANDING" if mom_5m > 1.5 else ""),
        }

        trap = {"type": "NONE", "confidence": 0}
        squeeze = False

        # OI velocity from raw data
        velocity = ""
        call_oi_speed = None
        put_oi_speed = None
        if prev_df is not None:
            merged = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
            call_oi_speed = (merged["call_oi"] - merged["call_oi_prev"]).sum()
            put_oi_speed = (merged["put_oi"] - merged["put_oi_prev"]).sum()
            oi_vel_history.append(abs(call_oi_speed) + abs(put_oi_speed))

            if len(oi_vel_history) >= 5:
                import numpy as np
                hist = np.array(oi_vel_history)
                threshold = hist.mean() + 1.5 * hist.std()
            else:
                threshold = 150_000

            if call_oi_speed < -threshold:
                velocity = "BULLISH SURGE (Call Covering)"
            elif put_oi_speed > threshold:
                velocity = "BULLISH BUILDUP (Put Writing)"
            elif put_oi_speed < -threshold:
                velocity = "BEARISH SURGE (Put Covering)"
            elif call_oi_speed > threshold:
                velocity = "BEARISH BUILDUP (Call Writing)"

        # Structural events from raw OI
        vacuum = detect_liquidity_vacuum(df, prev_df, spot, gamma, mom_5m)
        wall_break_vac = detect_wall_break_vacuum(df, prev_df, spot, gamma)
        flip_breakout = detect_gamma_flip_breakout(spot, prev_spot, flip_level, mom_5m)
        liq_accel = detect_liquidity_acceleration(spot, prev_spot, momentum_data, None, None)

        # Trend
        expected_move = straddle / 2
        trend = recompute_trend(spot_history, spot, expected_move)

        # MPM
        move_prob = compute_move_probability(
            gamma=gamma, momentum_data=momentum_data, velocity=velocity,
            vacuum=vacuum, wall_break=wall_break_vac,
            flip_breakout=flip_breakout, acceleration=liq_accel,
            momentum_strikes=None, trend=trend,
        )

        # Gamma momentum
        gamma_mom = compute_gamma_momentum(list(gamma_history))

        # Score
        scores = {
            "gamma_structure":   _score_gamma(gamma, flip_level, spot, call_wall, put_wall),
            "gamma_momentum":    _score_gamma_momentum(gamma_mom),
            "straddle_momentum": _score_straddle(momentum_data),
            "spot_vs_walls":     _score_spot_vs_walls(spot, call_wall, put_wall, gamma),
            "oi_velocity":       _score_oi_velocity(velocity, call_oi_speed, put_oi_speed),
            "iv_premium":        _score_iv(momentum_data, straddle, days_to_expiry),
            "move_prob":         _score_move_prob(move_prob),
            "structural_event":  _score_structural(vacuum, wall_break_vac, flip_breakout, liq_accel, squeeze),
            "trend":             _score_trend(trend),
        }

        total = sum(scores[k] * W[k] for k in scores)

        setup = _classify_setup(
            vacuum, wall_break_vac, flip_breakout, liq_accel,
            squeeze, trap, velocity, gamma, momentum_data,
            spot, call_wall, put_wall, trend, gamma_mom=gamma_mom,
        )
        direction, _ = _resolve_direction(
            bias, move_prob, flip_breakout, vacuum,
            wall_break_vac, liq_accel, trend, velocity,
        )

        if bias == "RANGE" and direction != "NEUTRAL":
            total -= 0.5
        if (bias == "BULLISH" and direction == "SHORT") or \
           (bias == "BEARISH" and direction == "LONG"):
            total -= 1.0

        total = round(max(0.0, min(10.0, total)), 1)

        action, _, _ = _decide_action(
            total, setup, 0, trap, bias, days_to_expiry,
            regime=regime, direction=direction, gamma=gamma,
            gamma_mom=gamma_mom,
        )

        structural_fired = []
        if vacuum.get("detected") and vacuum.get("status") == "CONFIRMED":
            structural_fired.append(f"VAC({vacuum.get('score', 0)})")
        if wall_break_vac.get("detected"):
            structural_fired.append("WB")
        if flip_breakout.get("detected"):
            structural_fired.append("FLIP")
        if liq_accel.get("detected"):
            structural_fired.append(f"ACCEL({liq_accel.get('conviction', '?')})")

        results.append({
            "timestamp": timestamp, "spot": spot, "score": total,
            "action": action, "setup": setup, "direction": direction,
            "regime": regime, "bias": bias, "scores": scores,
            "trend": trend, "move_prob": move_prob.get("probability", 0),
            "gamma": gamma, "gamma_mom": gamma_mom,
            "structural_fired": structural_fired,
        })

        prev_spot = spot
        prev_df = df

    # Swings
    swings = identify_swings(spots, timestamps, min_pts=min_pts)
    take_trades = [r for r in results if r["action"] in ("TAKE TRADE", "SEND IT")]
    stalks = [r for r in results if r["action"] == "STALK — WAIT FOR TRIGGER"]

    # Swing analysis
    caught = 0
    stalked = 0
    for sw in swings:
        sw_dir = "LONG" if sw["direction"] == "UP" else "SHORT"
        start_i, end_i = sw["start_idx"], sw["end_idx"]
        mid_i = start_i + max(1, (end_i - start_i) // 2)
        all_window = results[max(0, start_i - 3):mid_i + 1]

        fired = any(r["action"] in ("TAKE TRADE", "SEND IT") for r in all_window)
        stalked_it = any(r["action"] == "STALK — WAIT FOR TRIGGER" for r in all_window)

        if fired:
            caught += 1
        elif stalked_it:
            stalked += 1

    total_swings = len(swings)
    missed = total_swings - caught - stalked

    return {
        "file": date_label,
        "candles": len(results),
        "swings": total_swings,
        "caught": caught,
        "stalked": stalked,
        "missed": missed,
        "take_trades": len(take_trades),
        "stalks": len(stalks),
        "structural": sum(1 for r in results if r["structural_fired"]),
        "results": results,
        "swings_list": swings,
    }


def print_day_detail(summary):
    """Print detailed output for a single day."""
    results = summary["results"]
    swings = summary["swings_list"]

    print(f"\n  {'─'*66}")
    print(f"  {summary['file']}")
    print(f"  Candles: {summary['candles']}  |  Swings: {summary['swings']}  |  Structural: {summary['structural']}")
    print(f"  {'─'*66}")

    take_trades = [r for r in results if r["action"] in ("TAKE TRADE", "SEND IT")]
    if take_trades:
        print(f"  {Fore.GREEN}TAKE TRADE: {len(take_trades)}{Style.RESET_ALL}")
        for r in take_trades:
            trend_tag = f" TREND {r['trend']['direction']} {r['trend']['move_pts']}pts" if r["trend"]["trending"] else ""
            struct_tag = f" [{'+'.join(r['structural_fired'])}]" if r['structural_fired'] else ""
            print(f"    {r['timestamp']}  {r['spot']:>8.1f}  [{r['score']}/10]  "
                  f"{r['setup']:20s}  {r['direction']}{trend_tag}{struct_tag}")

    for sw in swings:
        sw_dir = "LONG" if sw["direction"] == "UP" else "SHORT"
        start_i, end_i = sw["start_idx"], sw["end_idx"]
        mid_i = start_i + max(1, (end_i - start_i) // 2)
        all_window = results[max(0, start_i - 3):mid_i + 1]
        best = max(all_window, key=lambda r: r["score"]) if all_window else None

        fired = any(r["action"] in ("TAKE TRADE", "SEND IT") for r in all_window)
        stalked_it = any(r["action"] == "STALK — WAIT FOR TRIGGER" for r in all_window)

        if fired:
            status = f"{Fore.GREEN}CAUGHT{Style.RESET_ALL}"
        elif stalked_it:
            status = f"{Fore.YELLOW}STALKED{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}MISSED{Style.RESET_ALL}"

        print(f"    {sw['direction']:5s} {sw['pts']:+5.0f}pts  "
              f"{sw['start_time']}-{sw['end_time']}  "
              f"({sw['start_spot']:.0f}→{sw['end_spot']:.0f})  "
              f"{status}  best=[{best['score'] if best else 0}/10]")


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_from_oi.py [--min-pts N] <options_log_csv> [...]")
        sys.exit(1)

    args = sys.argv[1:]
    min_pts = 25
    if "--min-pts" in args:
        idx = args.index("--min-pts")
        min_pts = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    files = args
    summaries = []

    for f in sorted(files):
        if not os.path.exists(f):
            print(f"  SKIP: {f} not found")
            continue
        print(f"  Processing {f}...")
        summary = run_one_day(f, min_pts=min_pts)
        if summary:
            summaries.append(summary)
            print_day_detail(summary)

    if len(summaries) > 1:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS ({len(summaries)} days)")
        print(f"{'='*70}")
        print(f"  {'Date':<50s} {'Swings':>6s} {'Caught':>7s} {'Stalk':>6s} {'Miss':>5s} {'Rate':>5s}")
        print(f"  {'─'*50} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*5}")

        total_swings = 0
        total_caught = 0
        total_stalked = 0
        total_missed = 0

        for s in summaries:
            rate = f"{(s['caught'] + s['stalked']) / s['swings'] * 100:.0f}%" if s['swings'] > 0 else "N/A"
            print(f"  {s['file']:<50s} {s['swings']:>6d} {s['caught']:>7d} {s['stalked']:>6d} {s['missed']:>5d} {rate:>5s}")
            total_swings += s['swings']
            total_caught += s['caught']
            total_stalked += s['stalked']
            total_missed += s['missed']

        overall_rate = f"{(total_caught + total_stalked) / total_swings * 100:.0f}%" if total_swings > 0 else "N/A"
        print(f"  {'─'*50} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*5}")
        print(f"  {'TOTAL':<50s} {total_swings:>6d} {total_caught:>7d} {total_stalked:>6d} {total_missed:>5d} {overall_rate:>5s}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
