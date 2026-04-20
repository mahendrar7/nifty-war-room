"""
backtest_sniper_frequency.py — Compare sniper PnL at different sampling frequencies.

Runs today's (or given day's) options_log through two variants:
  A) Current live:  sniper every 1 min,  radar every 3 min (cached between)
  B) Proposed:      sniper every 15 min,  radar just before each sniper eval

In both variants, state (spot/gamma/straddle history) is updated every 1 min
from the options_log.  Only the DECISION frequency changes.  Structural
detectors compare the current strike-OI frame against the previous frame at
the same cadence (so Variant A compares 1-min deltas, Variant B compares
15-min deltas).

Usage:
    python backtest_sniper_frequency.py data/options_log_1min_nifty_09042026.csv
"""

import os
import sys
from collections import deque

import numpy as np

from config import (
    INSTRUMENT_PROFILES,
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
from radar import radar_scan

from backtest_from_oi import (
    load_day, compute_walls, compute_flip_level,
    recompute_trend, simulate_pnl,
)


def _infer_profile_name(path):
    b = os.path.basename(path).lower()
    if "sensex" in b:
        return "SENSEX"
    return "NIFTY"


def run_variant(snapshots, sniper_interval_min, radar_interval_min,
                profile_name="NIFTY", include_radar=True,
                sniper_phase=0, radar_phase=0):
    """
    Walk the 1-min snapshots. State updates every minute. Sniper evaluates
    every `sniper_interval_min` minutes; radar every `radar_interval_min`.
    Returns a results list compatible with simulate_pnl().
    """
    profile = INSTRUMENT_PROFILES.get(profile_name, {})

    spot_history = deque(maxlen=60)
    gamma_history = deque(maxlen=10)
    straddle_history = deque(maxlen=10)
    oi_vel_history = deque(maxlen=30)

    prev_df_sniper = None
    prev_df_radar = None
    prev_spot = None
    session_high = None
    session_low = None
    last_radar = None

    results = []

    for i, snap in enumerate(snapshots):
        spot = snap["spot"]
        gamma = snap["gamma"]
        straddle = snap["straddle"]
        bias = snap["bias"]
        atm = snap["atm"]
        days_to_expiry = snap["days_to_expiry"]
        df = snap["df"]
        timestamp = snap["timestamp"].split(" ")[-1] if " " in snap["timestamp"] else snap["timestamp"]

        # ── Update state every minute ──
        spot_history.append(spot)
        if session_high is None or spot > session_high:
            session_high = spot
        if session_low is None or spot < session_low:
            session_low = spot
        gamma_history.append(gamma)
        straddle_history.append(straddle)

        # Decision gates
        sniper_eligible = ((i - sniper_phase) % sniper_interval_min == 0 and i >= sniper_phase)
        radar_eligible = ((i - radar_phase) % radar_interval_min == 0 and i >= radar_phase)

        # Walls (needed for radar too)
        call_wall, put_wall = compute_walls(df, spot)
        flip_level = compute_flip_level(df, spot)

        # Straddle momentum (always computed — uses 1-min state)
        mom_5m = 0.0
        mom_15m = 0.0
        if len(straddle_history) >= 6:
            prev_s = straddle_history[-6]
            mom_5m = ((straddle - prev_s) / (prev_s + 1e-9)) * 100
        if len(straddle_history) >= 10:
            prev_s15 = straddle_history[0]
            mom_15m = ((straddle - prev_s15) / (prev_s15 + 1e-9)) * 100

        momentum_data = {
            "momentum_5m": mom_5m,
            "momentum_15m": mom_15m,
            "status": "FAST EXPANDING" if mom_5m > 3.0 else ("EXPANDING" if mom_5m > 1.5 else ""),
        }

        # ── Radar scan ──
        if include_radar and radar_eligible:
            last_radar = radar_scan(
                spot=spot, flip_level=flip_level,
                call_wall=call_wall, put_wall=put_wall,
                df=df, prev_df=prev_df_radar,
                momentum_data=momentum_data,
                straddle_history=list(straddle_history),
                profile=profile,
                spot_history=list(spot_history),
            )
            prev_df_radar = df

        # Default "no action" row so simulate_pnl can iterate
        action = "WAIT"
        total = 0.0
        direction = "NEUTRAL"
        setup = "DEVELOPING"
        regime = "NEUTRAL"
        trend = {"trending": False, "direction": None, "move_pts": 0,
                 "duration_minutes": 0, "pullback": False,
                 "broader_direction": None, "broader_move_pts": 0}
        structural_fired = []

        if sniper_eligible:
            # Regime
            if gamma > 0 and abs(spot - atm) < 40:
                regime = "GAMMA PINNING"
            elif gamma < 0:
                regime = "VOLATILITY EXPANSION"
            else:
                regime = "NEUTRAL"

            # OI velocity — compared against sniper's prev frame
            velocity = ""
            call_oi_speed = None
            put_oi_speed = None
            if prev_df_sniper is not None:
                merged = df.merge(prev_df_sniper, on="strike", suffixes=("", "_prev"))
                call_oi_speed = (merged["call_oi"] - merged["call_oi_prev"]).sum()
                put_oi_speed = (merged["put_oi"] - merged["put_oi_prev"]).sum()
                oi_vel_history.append(abs(call_oi_speed) + abs(put_oi_speed))

                if len(oi_vel_history) >= 5:
                    hist = np.array(list(oi_vel_history))
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

            # Structural events — compared against sniper's prev frame
            vacuum = detect_liquidity_vacuum(df, prev_df_sniper, spot, gamma, mom_5m)
            wall_break_vac = detect_wall_break_vacuum(df, prev_df_sniper, spot, gamma)
            flip_breakout = detect_gamma_flip_breakout(spot, prev_spot, flip_level, mom_5m)
            liq_accel = detect_liquidity_acceleration(spot, prev_spot, momentum_data, None, None)

            # Trend
            expected_move = straddle / 2
            trend = recompute_trend(spot_history, spot, expected_move,
                                    session_high, session_low)

            # MPM
            move_prob = compute_move_probability(
                gamma=gamma, momentum_data=momentum_data, velocity=velocity,
                vacuum=vacuum, wall_break=wall_break_vac,
                flip_breakout=flip_breakout, acceleration=liq_accel,
                momentum_strikes=None, trend=trend,
            )

            gamma_mom = compute_gamma_momentum(list(gamma_history))

            trap = {"type": "NONE", "confidence": 0}
            squeeze = False

            scores = {
                "gamma_structure":   _score_gamma(gamma, flip_level, spot, call_wall, put_wall),
                "gamma_momentum":    _score_gamma_momentum(gamma_mom),
                "straddle_momentum": _score_straddle(momentum_data),
                "spot_vs_walls":     _score_spot_vs_walls(spot, call_wall, put_wall, gamma),
                "oi_velocity":       _score_oi_velocity(velocity, call_oi_speed, put_oi_speed),
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

            # Bias conflict penalties (same as live)
            if bias == "RANGE" and direction != "NEUTRAL":
                total -= 0.5
            if (bias == "BULLISH" and direction == "SHORT") or \
               (bias == "BEARISH" and direction == "LONG"):
                total -= 1.0

            # Radar confluence (same ±0.5 / -1.0 rule as live)
            if include_radar and last_radar and last_radar.get("active") \
                    and direction in ("LONG", "SHORT"):
                r_dir = (last_radar.get("direction") or "").upper()
                if r_dir == direction:
                    total += 0.5
                elif r_dir and r_dir != direction:
                    total -= 1.0

            total = round(max(0.0, min(10.0, total)), 1)

            action, _, _ = _decide_action(
                total, setup, 0, trap, bias, days_to_expiry,
                regime=regime, direction=direction, gamma=gamma,
                gamma_mom=gamma_mom,
            )

            if vacuum.get("detected") and vacuum.get("status") == "CONFIRMED":
                structural_fired.append(f"VAC({vacuum.get('score', 0)})")
            if wall_break_vac.get("detected"):
                structural_fired.append("WB")
            if flip_breakout.get("detected"):
                structural_fired.append("FLIP")
            if liq_accel.get("detected"):
                structural_fired.append(f"ACCEL({liq_accel.get('conviction', '?')})")

            prev_df_sniper = df

        results.append({
            "timestamp": timestamp,
            "spot": spot,
            "score": total,
            "action": action,
            "setup": setup,
            "direction": direction,
            "regime": regime,
            "bias": bias,
            "trend": trend,
            "structural_fired": structural_fired,
        })

        prev_spot = spot

    return results


def dump_decisions(results, sniper_interval_min, sniper_phase=0):
    """Print every sniper evaluation — not just fires — so we see what it saw."""
    print(f"  All sniper evaluations (every {sniper_interval_min}m, phase={sniper_phase}):")
    for i, r in enumerate(results):
        if (i - sniper_phase) % sniper_interval_min != 0 or i < sniper_phase:
            continue
        struct = f" [{'+'.join(r['structural_fired'])}]" if r['structural_fired'] else ""
        trd = ""
        if r['trend']['trending']:
            trd = f" TREND {r['trend']['direction']} {r['trend']['move_pts']}pts"
        print(f"    {r['timestamp']}  {r['spot']:>8.1f}  [{r['score']:>4.1f}/10]  "
              f"{r['action']:<22s}  {r['setup']:<20s}  {r['direction']:<6s}{trd}{struct}")


def summarise(label, results, lot_size=65):
    trades = simulate_pnl({"results": results}, lot_size=lot_size)
    take = [r for r in results if r["action"] in ("TAKE TRADE", "SEND IT")]
    stalk = [r for r in results if r["action"] == "STALK — WAIT FOR TRIGGER"]

    pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    stops = sum(1 for t in trades if t["result"] == "STOP")

    print(f"\n  {label}")
    print(f"  {'─' * 66}")
    print(f"  Sniper signals: TAKE/SEND={len(take)}  STALK={len(stalk)}")

    if take:
        print(f"  Fired at:")
        for r in take[:20]:
            struct = f" [{'+'.join(r['structural_fired'])}]" if r['structural_fired'] else ""
            print(f"    {r['timestamp']}  {r['spot']:>8.1f}  [{r['score']}/10]  "
                  f"{r['setup']:20s}  {r['direction']:6s}{struct}")
        if len(take) > 20:
            print(f"    ... and {len(take) - 20} more")

    print(f"\n  Trades: {len(trades)}  Wins: {wins}  Stops: {stops}")
    if trades:
        for j, t in enumerate(trades):
            print(f"    #{j+1}  {t.get('candles',0):>3d}m  peak {t.get('pts',0):+6.1f}  "
                  f"L1:{t.get('lot1_exit','?'):>6s} {t.get('lot1_pnl',0):+7,}  "
                  f"L2:{t.get('lot2_exit','?'):>6s} {t.get('lot2_pnl',0):+7,}  "
                  f"= {t['pnl']:+,}")
    print(f"  Net P&L:  Rs {pnl:+,.0f}")
    return {"label": label, "pnl": pnl, "trades": len(trades),
            "wins": wins, "stops": stops, "signals": len(take)}


def run_day_variant(snapshots, sniper_int, radar_int, profile_name, lot_size,
                    s_phase=0, r_phase=0):
    """Run one (day, variant) combination and return summary stats."""
    results = run_variant(
        snapshots,
        sniper_interval_min=sniper_int,
        radar_interval_min=radar_int,
        profile_name=profile_name,
        include_radar=True,
        sniper_phase=s_phase,
        radar_phase=r_phase,
    )
    trades = simulate_pnl({"results": results}, lot_size=lot_size)
    take = [r for r in results if r["action"] in ("TAKE TRADE", "SEND IT")]
    pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    stops = sum(1 for t in trades if t["result"] == "STOP")
    return {
        "pnl": pnl, "trades": len(trades), "wins": wins,
        "stops": stops, "signals": len(take),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_sniper_frequency.py <options_log_csv> [...]")
        sys.exit(1)

    files = [f for f in sys.argv[1:] if os.path.exists(f)]
    if not files:
        print("No valid files provided")
        sys.exit(1)

    # Variants to run: (label, sniper_interval, radar_interval)
    variants = [
        ("1m sniper / 3m radar  (current)", 1, 3),
        ("5m sniper / 5m radar  (just before)", 5, 5),
        ("15m sniper / 15m radar (just before)", 15, 15),
    ]

    # Group files by instrument for separate aggregation
    by_instrument = {}
    for f in sorted(files):
        inst = _infer_profile_name(f)
        by_instrument.setdefault(inst, []).append(f)

    for instrument, paths in by_instrument.items():
        lot_size = 65 if instrument == "NIFTY" else 10
        print(f"\n{'=' * 78}")
        print(f"  {instrument}  ({len(paths)} day{'s' if len(paths) > 1 else ''})"
              f"   lot={lot_size}   TAKE={SNIPER_TAKE_TRADE}  SEND={SNIPER_SEND_IT}")
        print(f"{'=' * 78}")

        # Per-day table header
        v_labels = [v[0] for v in variants]
        header = f"  {'Date':<10s}"
        for lbl in v_labels:
            header += f"  {lbl[:22]:>22s}"
        print(header)
        print(f"  {'─' * 10}" + ("  " + "─" * 22) * len(variants))

        # Totals per variant
        totals = [{"pnl": 0, "trades": 0, "wins": 0, "stops": 0, "signals": 0}
                  for _ in variants]

        for path in paths:
            date_tag = os.path.basename(path).split("_")[-1].replace(".csv", "")
            snapshots = load_day(path)
            if not snapshots:
                continue

            line = f"  {date_tag:<10s}"
            for i, (lbl, s_int, r_int) in enumerate(variants):
                s = run_day_variant(snapshots, s_int, r_int, instrument, lot_size)
                cell = f"{s['trades']:>2d}t {s['wins']:>2d}w Rs{s['pnl']:>+8,.0f}"
                line += f"  {cell:>22s}"
                for k in totals[i]:
                    totals[i][k] += s[k]
            print(line)

        # Aggregate totals
        print(f"  {'─' * 10}" + ("  " + "─" * 22) * len(variants))
        line = f"  {'TOTAL':<10s}"
        for t in totals:
            cell = f"{t['trades']:>2d}t {t['wins']:>2d}w Rs{t['pnl']:>+8,.0f}"
            line += f"  {cell:>22s}"
        print(line)

        # Detailed aggregate below
        print(f"\n  Aggregate across {len(paths)} days:")
        print(f"  {'Variant':<40s} {'Sigs':>6s} {'Trds':>5s} {'Wins':>5s} "
              f"{'Stops':>6s} {'Win%':>6s} {'P&L':>14s}")
        print(f"  {'─' * 40} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 6} "
              f"{'─' * 6} {'─' * 14}")
        for i, (lbl, _, _) in enumerate(variants):
            t = totals[i]
            win_rate = (t['wins'] / t['trades'] * 100) if t['trades'] else 0.0
            print(f"  {lbl:<40s} {t['signals']:>6d} {t['trades']:>5d} "
                  f"{t['wins']:>5d} {t['stops']:>6d} {win_rate:>5.0f}% "
                  f"Rs {t['pnl']:>+11,.0f}")
        print()


if __name__ == "__main__":
    main()
