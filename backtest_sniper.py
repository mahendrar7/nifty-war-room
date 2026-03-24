"""
backtest_sniper.py — Replay signals + raw OI data through the updated sniper.

Reads the signals CSV for per-candle market state (spot, gamma, straddle,
bias, regime, etc.) and the options log CSV for raw per-strike OI data.
Recomputes structural events (vacuum, wall break, flip breakout, liq accel)
and trend/MPM from scratch using CURRENT config thresholds.

Usage:
    python backtest_sniper.py data/signals_log_nifty_20032026.csv

Automatically finds the matching options_log_1min_*.csv in the same folder.
Does NOT modify any files — pure read-only analysis.
"""

import csv
import os
import re
import sys
from collections import deque

import pandas as pd
from colorama import Fore, Style

from config import (
    TREND_WINDOW_MINUTES, TREND_MIN_MOVE_MULT,
    MPM_WEIGHTS, MPM_GAMMA_AMP, MPM_GAMMA_DAMP,
    SNIPER_TAKE_TRADE, SNIPER_SEND_IT,
)
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


def parse_bool(val):
    return str(val).strip().lower() == "true"


def find_options_log(signals_path):
    """Find matching options_log_1min CSV from a signals_log path."""
    dirname = os.path.dirname(signals_path)
    basename = os.path.basename(signals_path)
    # Extract instrument and date: signals_log_nifty_20032026.csv
    m = re.search(r'signals_log_(\w+)_(\d+)\.csv', basename)
    if not m:
        return None
    instrument, date_str = m.group(1), m.group(2)
    options_path = os.path.join(dirname, f"options_log_1min_{instrument}_{date_str}.csv")
    return options_path if os.path.exists(options_path) else None


def load_options_snapshots(options_path):
    """
    Load options CSV and group into per-timestamp DataFrames.
    Returns dict: { "HH:MM:SS" -> DataFrame with columns [strike, call_oi, put_oi] }
    """
    with open(options_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by timestamp
    by_ts = {}
    for row in rows:
        ts_full = row["timestamp"]  # "2026-03-20 09:15:00"
        ts_time = ts_full.split(" ")[1] if " " in ts_full else ts_full
        # Normalise to HH:MM:SS
        parts = ts_time.split(":")
        if len(parts) == 2:
            ts_time = ts_time + ":00"
        by_ts.setdefault(ts_time, []).append(row)

    snapshots = {}
    for ts, ts_rows in by_ts.items():
        # Pivot: one row per strike with call_oi and put_oi
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

        df = pd.DataFrame(list(strikes.values()))
        if not df.empty:
            df = df.sort_values("strike").reset_index(drop=True)
        snapshots[ts] = df

    return snapshots


def recompute_trend(spot_history, spot, expected_move, session_high=None, session_low=None):
    """Recompute trend with pullback detection (deque + session H/L)."""
    history = list(spot_history)
    base = {"trending": False, "direction": None, "move_pts": 0,
            "duration_minutes": 0, "pullback": False,
            "broader_direction": None, "broader_move_pts": 0}
    if len(history) < 10:
        return base
    window = history[-TREND_WINDOW_MINUTES:]
    move = spot - window[0]
    threshold = expected_move * TREND_MIN_MOVE_MULT
    broad_move = spot - history[0]
    broad_direction = "UP" if broad_move > 0 else "DOWN" if broad_move < 0 else None
    broad_abs = round(abs(broad_move), 1)
    if abs(move) >= threshold:
        direction = "UP" if move > 0 else "DOWN"
        pullback = (direction != broad_direction and abs(broad_move) >= threshold)
        if not pullback and session_high is not None and session_low is not None:
            session_range = session_high - session_low
            if session_range >= threshold * 2:
                position = (spot - session_low) / session_range
                rally_from_low = spot - session_low
                drop_from_high = session_high - spot
                if direction == "DOWN" and position > 0.5 and rally_from_low > abs(move) * 2:
                    pullback = True
                    broad_direction = "UP"
                    broad_abs = round(rally_from_low, 1)
                elif direction == "UP" and position < 0.5 and drop_from_high > abs(move) * 2:
                    pullback = True
                    broad_direction = "DOWN"
                    broad_abs = round(drop_from_high, 1)
        return {"trending": True, "direction": direction,
                "move_pts": round(abs(move), 1), "duration_minutes": len(window),
                "pullback": pullback, "broader_direction": broad_direction,
                "broader_move_pts": broad_abs}
    return {**base, "broader_direction": broad_direction, "broader_move_pts": broad_abs}


def normalise_timestamp(ts):
    """Normalise signal CSV timestamp (HH:MM:SS or HH:MM) to HH:MM:00."""
    parts = ts.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}:00"
    return ts


def identify_swings(spots, timestamps, min_pts=25):
    """Identify swing moves of at least min_pts from the spot series."""
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
                    "direction": "UP",
                    "start_idx": swing_start_idx, "end_idx": swing_high_idx,
                    "start_time": timestamps[swing_start_idx],
                    "end_time": timestamps[swing_high_idx],
                    "start_spot": spots[swing_start_idx],
                    "end_spot": spots[swing_high_idx],
                    "pts": round(spots[swing_high_idx] - spots[swing_start_idx], 1),
                })
                swing_start_idx = swing_high_idx
                swing_low = spots[swing_high_idx]
                swing_low_idx = swing_high_idx

        if spots[i] - swing_low >= min_pts and spots[swing_start_idx] - swing_low >= min_pts:
            if swing_low_idx > swing_start_idx:
                swings.append({
                    "direction": "DOWN",
                    "start_idx": swing_start_idx, "end_idx": swing_low_idx,
                    "start_time": timestamps[swing_start_idx],
                    "end_time": timestamps[swing_low_idx],
                    "start_spot": spots[swing_start_idx],
                    "end_spot": spots[swing_low_idx],
                    "pts": round(spots[swing_start_idx] - spots[swing_low_idx], 1),
                })
                swing_start_idx = swing_low_idx
                swing_high = spots[swing_low_idx]
                swing_high_idx = swing_low_idx

    return swings


def run_backtest(signals_path):
    # ── Load signals CSV ─────────────────────────────────────────────
    with open(signals_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Empty CSV")
        return

    # ── Load raw OI data ─────────────────────────────────────────────
    options_path = find_options_log(signals_path)
    oi_snapshots = {}
    if options_path:
        print(f"  Loading raw OI data from {options_path}...")
        oi_snapshots = load_options_snapshots(options_path)
        print(f"  Loaded {len(oi_snapshots)} OI snapshots")
    else:
        print(f"  WARNING: No options_log CSV found — structural events won't be recomputed")

    print(f"\n{'='*70}")
    print(f"  SNIPER BACKTEST — {signals_path}")
    print(f"  Rows: {len(rows)}  |  TAKE={SNIPER_TAKE_TRADE}  SEND={SNIPER_SEND_IT}")
    print(f"  Raw OI: {'YES' if oi_snapshots else 'NO (using logged values)'}")
    print(f"{'='*70}\n")

    # ── Pass 1: Score every row ──────────────────────────────────────
    spot_history = deque(maxlen=60)
    gamma_history = deque(maxlen=10)
    prev_spot = None
    prev_df = None
    results = []
    spots = []
    timestamps = []
    session_high = None
    session_low = None

    for row in rows:
        spot = parse_float(row["spot"])
        gamma = parse_float(row["gamma_pressure"])
        straddle = parse_float(row["straddle"])
        call_wall = parse_float(row["call_wall"])
        put_wall = parse_float(row["put_wall"])
        flip_level = parse_float(row["flip_level"]) or None
        bias = row.get("bias", "RANGE")
        confidence = int(parse_float(row.get("confidence", 0)))
        regime = row.get("regime", "")
        days_to_expiry = int(parse_float(row.get("days_to_expiry", 4)))
        velocity = row.get("oi_velocity", "") or ""
        squeeze = parse_bool(row.get("squeeze", ""))
        timestamp = row.get("timestamp", "")

        spots.append(spot)
        timestamps.append(timestamp)
        spot_history.append(spot)
        if session_high is None or spot > session_high:
            session_high = spot
        if session_low is None or spot < session_low:
            session_low = spot
        gamma_history.append(gamma)

        # Reconstruct momentum_data
        mom_5m = parse_float(row.get("straddle_mom_5m"))
        mom_15m = parse_float(row.get("straddle_mom_15m"))
        momentum_data = {
            "momentum_5m": mom_5m,
            "momentum_15m": mom_15m,
            "status": "FAST EXPANDING" if mom_5m > 3.0 else ("EXPANDING" if mom_5m > 1.5 else ""),
        }

        trap = {
            "type": row.get("trap_type", "NONE") or "NONE",
            "confidence": int(parse_float(row.get("trap_confidence", 0))),
        }

        # ── Recompute structural events from raw OI ──────────────────
        ts_key = normalise_timestamp(timestamp)
        curr_df = oi_snapshots.get(ts_key)

        if curr_df is not None and not curr_df.empty:
            # Vacuum
            vacuum = detect_liquidity_vacuum(
                curr_df, prev_df, spot, gamma, mom_5m,
            )
            # Wall break
            wall_break_vac = detect_wall_break_vacuum(
                curr_df, prev_df, spot, gamma,
            )
            # Flip breakout
            flip_breakout = detect_gamma_flip_breakout(
                spot, prev_spot, flip_level, mom_5m,
            )
            # Liq acceleration
            liq_accel = detect_liquidity_acceleration(
                spot, prev_spot, momentum_data, None, None,
            )
            prev_df = curr_df
        else:
            # Fallback to logged values
            vac_status = row.get("vacuum_status", "NONE") or "NONE"
            vacuum = {
                "status": vac_status, "detected": vac_status != "NONE",
                "score": int(parse_float(row.get("vacuum_score", 0))),
                "direction": row.get("vacuum_dir"),
            }
            wall_break_vac = {
                "detected": parse_bool(row.get("wall_break", False)),
                "direction": row.get("wall_break_dir"),
            }
            flip_breakout = {
                "detected": parse_bool(row.get("flip_breakout", False)),
                "direction": row.get("flip_breakout_dir"),
            }
            liq_accel = {
                "detected": parse_bool(row.get("liq_accel", False)),
                "direction": row.get("liq_accel_dir"),
                "score": int(parse_float(row.get("liq_accel_score", 0))),
                "conviction": "HIGH" if parse_float(row.get("liq_accel_score", 0)) >= 75 else "MODERATE",
            }

        # ── Recompute trend with NEW thresholds ──────────────────────
        expected_move = straddle / 2
        trend = recompute_trend(spot_history, spot, expected_move, session_high, session_low)

        # ── Recompute MPM with trend + structural events ─────────────
        move_prob = compute_move_probability(
            gamma=gamma, momentum_data=momentum_data, velocity=velocity,
            vacuum=vacuum, wall_break=wall_break_vac,
            flip_breakout=flip_breakout, acceleration=liq_accel,
            momentum_strikes=None, trend=trend,
        )

        # ── Gamma momentum ───────────────────────────────────────────
        gamma_mom = compute_gamma_momentum(list(gamma_history))

        # ── Score ────────────────────────────────────────────────────
        scores = {
            "gamma_structure":   _score_gamma(gamma, flip_level, spot, call_wall, put_wall),
            "gamma_momentum":    _score_gamma_momentum(gamma_mom),
            "straddle_momentum": _score_straddle(momentum_data),
            "spot_vs_walls":     _score_spot_vs_walls(spot, call_wall, put_wall, gamma),
            "oi_velocity":       _score_oi_velocity(velocity, None, None),
            "iv_premium":        _score_iv(momentum_data, straddle, days_to_expiry),
            "move_prob":         _score_move_prob(move_prob),
            "structural_event":  _score_structural(vacuum, wall_break_vac, flip_breakout, liq_accel, squeeze),
            "trend":             _score_trend(trend, spot_history=list(spot_history),
                                             spot=spot, session_high=session_high,
                                             session_low=session_low),
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
            spot=spot, session_high=session_high, session_low=session_low,
        )

        # Direction conflict penalty
        if bias == "RANGE" and direction != "NEUTRAL":
            total -= 0.5
        if (bias == "BULLISH" and direction == "SHORT") or \
           (bias == "BEARISH" and direction == "LONG"):
            total -= 1.0

        total = round(max(0.0, min(10.0, total)), 1)

        action, _, _ = _decide_action(
            total, setup, confidence, trap, bias, days_to_expiry,
            regime=regime, direction=direction, gamma=gamma,
            gamma_mom=gamma_mom,
        )

        # Track structural events that fired (for reporting)
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
            "timestamp": timestamp,
            "spot": spot,
            "score": total,
            "action": action,
            "setup": setup,
            "direction": direction,
            "regime": regime,
            "bias": bias,
            "scores": scores,
            "trend": trend,
            "move_prob": move_prob.get("probability", 0),
            "gamma": gamma,
            "gamma_mom": gamma_mom,
            "structural_fired": structural_fired,
        })

        prev_spot = spot

    # ── Pass 2: Identify swings ──────────────────────────────────────
    swings = identify_swings(spots, timestamps, min_pts=25)

    # ── Pass 3: Report actionable signals ────────────────────────────
    take_trades = [r for r in results if r["action"] in ("TAKE TRADE", "SEND IT")]
    stalks = [r for r in results if r["action"] == "STALK — WAIT FOR TRIGGER"]

    print(f"  ACTIONABLE SIGNALS")
    print(f"  {'─'*66}")

    if take_trades:
        print(f"  {Fore.GREEN}TAKE TRADE / SEND IT: {len(take_trades)} signals{Style.RESET_ALL}")
        for r in take_trades:
            trend_tag = ""
            if r["trend"]["trending"]:
                trend_tag = f" | TREND {r['trend']['direction']} {r['trend']['move_pts']}pts"
            struct_tag = f" | {'+'.join(r['structural_fired'])}" if r['structural_fired'] else ""
            print(f"    {r['timestamp']}  {r['spot']:>8.1f}  "
                  f"[{r['score']}/10]  {r['action']:20s}  "
                  f"{r['setup']:20s}  {r['direction']:6s}"
                  f"  MPM={r['move_prob']}{trend_tag}{struct_tag}")
    else:
        print(f"  {Fore.RED}TAKE TRADE / SEND IT: 0 signals{Style.RESET_ALL}")

    print()
    if stalks:
        print(f"  {Fore.YELLOW}STALK: {len(stalks)} signals{Style.RESET_ALL}")
        stalk_groups = []
        current_group = [stalks[0]]
        for i in range(1, len(stalks)):
            prev_idx = next(j for j, r in enumerate(results) if r is stalks[i-1])
            curr_idx = next(j for j, r in enumerate(results) if r is stalks[i])
            if curr_idx - prev_idx <= 2:
                current_group.append(stalks[i])
            else:
                stalk_groups.append(current_group)
                current_group = [stalks[i]]
        stalk_groups.append(current_group)

        for group in stalk_groups:
            first, last = group[0], group[-1]
            dur = f"{first['timestamp']}-{last['timestamp']}" if len(group) > 1 else first['timestamp']
            spot_range = f"{min(r['spot'] for r in group):.0f}-{max(r['spot'] for r in group):.0f}"
            score_range = f"{min(r['score'] for r in group):.1f}-{max(r['score'] for r in group):.1f}"
            dir_set = set(r['direction'] for r in group)
            setup_set = set(r['setup'] for r in group)
            struct_any = set()
            for r in group:
                struct_any.update(r['structural_fired'])
            struct_tag = f" [{'+'.join(struct_any)}]" if struct_any else ""
            print(f"    {dur:25s}  spot {spot_range:15s}  "
                  f"[{score_range}]  {'/'.join(setup_set)}  {'/'.join(dir_set)}"
                  f"  ({len(group)} candles){struct_tag}")

    # ── Structural events summary ────────────────────────────────────
    struct_count = sum(1 for r in results if r['structural_fired'])
    if struct_count > 0:
        print(f"\n  {Fore.CYAN}Structural events fired: {struct_count} candles{Style.RESET_ALL}")
        # Show first few
        shown = 0
        for r in results:
            if r['structural_fired'] and shown < 15:
                print(f"    {r['timestamp']}  {r['spot']:>8.1f}  "
                      f"{'+'.join(r['structural_fired']):20s}  "
                      f"[{r['score']}/10] {r['action']}")
                shown += 1
        if struct_count > 15:
            print(f"    ... and {struct_count - 15} more")
    else:
        print(f"\n  {Fore.RED}Structural events fired: 0{Style.RESET_ALL}")

    # ── Pass 4: Swing analysis ───────────────────────────────────────
    print(f"\n  {'='*66}")
    print(f"  SWING ANALYSIS (25+ pt moves)")
    print(f"  {'─'*66}")

    caught = 0
    stalked = 0

    for sw in swings:
        sw_dir = "LONG" if sw["direction"] == "UP" else "SHORT"
        start_i, end_i = sw["start_idx"], sw["end_idx"]

        mid_i = start_i + max(1, (end_i - start_i) // 2)
        early_window = results[start_i:mid_i + 1]
        pre_window = results[max(0, start_i - 3):start_i]
        all_window = pre_window + early_window

        best = max(all_window, key=lambda r: r["score"]) if all_window else None
        fired = any(r["action"] in ("TAKE TRADE", "SEND IT") for r in all_window)
        stalked_it = any(r["action"] == "STALK — WAIT FOR TRIGGER" for r in all_window)
        correct_dir = any(
            r["direction"] == sw_dir
            for r in all_window
            if r["action"] in ("TAKE TRADE", "SEND IT", "STALK — WAIT FOR TRIGGER")
        )

        if fired and correct_dir:
            status = f"{Fore.GREEN}CAUGHT{Style.RESET_ALL}"
            caught += 1
        elif fired:
            status = f"{Fore.YELLOW}CAUGHT (wrong dir){Style.RESET_ALL}"
            caught += 1
        elif stalked_it and correct_dir:
            status = f"{Fore.YELLOW}STALKED{Style.RESET_ALL}"
            stalked += 1
        elif stalked_it:
            status = f"{Fore.YELLOW}STALKED (wrong dir){Style.RESET_ALL}"
            stalked += 1
        else:
            status = f"{Fore.RED}MISSED{Style.RESET_ALL}"

        breakdown = ""
        if best:
            parts = []
            for k in ("gamma_structure", "gamma_momentum", "straddle_momentum",
                       "spot_vs_walls", "oi_velocity", "move_prob",
                       "structural_event", "trend"):
                val = best["scores"].get(k, 0) * W.get(k, 0)
                if val > 0:
                    label = {"gamma_structure": "Gam", "gamma_momentum": "GΔ",
                             "straddle_momentum": "Str",
                             "spot_vs_walls": "Spt", "oi_velocity": "OI",
                             "move_prob": "MPM", "structural_event": "Evt",
                             "trend": "Trd", "iv_premium": "IV"}[k]
                    parts.append(f"{label}:{val:.1f}")
            breakdown = " | ".join(parts)

        print(f"\n  {sw['direction']:5s}  {sw['pts']:+6.0f}pts  "
              f"{sw['start_time']}-{sw['end_time']}  "
              f"({sw['start_spot']:.0f} → {sw['end_spot']:.0f})  "
              f"{status}")
        if best:
            trend_info = ""
            if best["trend"]["trending"]:
                trend_info = f"  trend={best['trend']['direction']} {best['trend']['move_pts']}pts"
            struct_info = f"  struct={'+'.join(best['structural_fired'])}" if best['structural_fired'] else ""
            print(f"    Best: {best['timestamp']} [{best['score']}/10] "
                  f"{best['action']}  {best['setup']}  {best['direction']}"
                  f"  MPM={best['move_prob']}{trend_info}{struct_info}")
            print(f"    Breakdown: {breakdown}")

    # ── Summary ──────────────────────────────────────────────────────
    total_swings = len(swings)
    missed = total_swings - caught - stalked
    print(f"\n  {'='*66}")
    print(f"  SUMMARY")
    print(f"  {'─'*66}")
    print(f"  Total 25+ pt swings:  {total_swings}")
    print(f"  {Fore.GREEN}Caught (TAKE/SEND):     {caught}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Stalked (early warning): {stalked}{Style.RESET_ALL}")
    print(f"  {Fore.RED}Missed:                  {missed}{Style.RESET_ALL}")
    if total_swings > 0:
        print(f"  Catch rate:            {(caught + stalked) / total_swings * 100:.0f}%")
    print(f"  {'='*66}\n")

    # ── Score distribution ───────────────────────────────────────────
    score_buckets = {"0-2": 0, "2-4": 0, "4-5.5": 0, "5.5-7": 0, "7+": 0}
    for r in results:
        s = r["score"]
        if s >= 7:
            score_buckets["7+"] += 1
        elif s >= 5.5:
            score_buckets["5.5-7"] += 1
        elif s >= 4:
            score_buckets["4-5.5"] += 1
        elif s >= 2:
            score_buckets["2-4"] += 1
        else:
            score_buckets["0-2"] += 1

    print(f"  SCORE DISTRIBUTION ({len(results)} candles)")
    print(f"  {'─'*40}")
    for bucket, count in score_buckets.items():
        bar = "█" * (count // 5) if count > 0 else ""
        pct = count / len(results) * 100
        label = ""
        if bucket == "7+":
            label = " ← SEND IT"
        elif bucket == "5.5-7":
            label = " ← TAKE TRADE"
        elif bucket == "4-5.5":
            label = " ← STALK"
        print(f"    {bucket:>6s}:  {count:4d} ({pct:4.1f}%) {bar}{label}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backtest_sniper.py <signals_csv>")
        sys.exit(1)
    run_backtest(sys.argv[1])
