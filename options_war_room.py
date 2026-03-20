# =============================================================================
# NIFTY OPTIONS WAR ROOM — BS GAMMA + FLIP LEVEL + POSITION SIZING EDITION
# =============================================================================
#
# CHANGELOG
# ---------
#   - True Black-Scholes gamma replaces OI-proxy gamma
#   - IV computed per strike via Newton-Raphson implied vol solver
#   - compute_gamma_pressure and compute_gamma_wall now use real GEX
#   - Graceful fallback to OI proxy if BS solve fails for a strike
#   - LOT_SIZE = 65, RISK_FREE_RATE = 6.5% added to CONFIG
#   - find_gamma_flip_level(): scans ±500pts to find exact zero-GEX price level
#   - MAP line shows gamma flip level in color (yellow / green danger / red crossed)
#   - One-shot Telegram alert when spot enters flip danger zone (no spam)
#   - Position sizing engine: regime-aware risk, expiry scaling, ML agreement
#   - Expiry-aware sizing: lot caps, stop widening, risk scalar on 0-2 DTE
#   - Capital deployment guard: never > MAX_CAPITAL_PCT in one trade
#   - Strike selection: dynamic OTM distance from gamma, regime, flip level
#   - Gamma Flip Breakout trigger: fires when spot crosses flip + straddle > 2%
#   - OI Velocity SURGE overrides RANGE bias → TREND FOLLOW action
#   - Trap confidence *= 0.4 when gamma < 0 — suppresses false traps in squeezes
#   - Wall Break Vacuum: OI drop >25% + spot crosses strike → 🌪 LIQUIDITY VACUUM
#   - Sniper decides direction → suggest_trade() computes strike, sizing, stop/target
#   - Gamma pinning regime uses gamma sign directly, not flip event
#   - Regime tracker bypass for structural events (no 3-candle delay)
#   - Liquidity acceleration detector: price + IV + OI all firing = launch phase
#   - Hold the Line engine: HOLD / TRAIL / EXIT verdict for running trades
#   - Move probability meter: synthesises all signals into 0-100 score + direction
#   - Meta+T hotkey: log trade entry from last suggestion or manual input
#   - Meta+X hotkey: manually exit active trade
#   - Minimal terminal layout: actionable data only, all verbose reasons → debug
#==============================================================================
# This file is intentionally thin. All domain logic lives in:
#
#   config.py          — all constants and thresholds
#   state.py           — MarketState, RegimeTracker, restore_from_csv
#   market_data.py     — Kite data fetching, instrument cache
#   gamma_engine.py    — BS gamma, IV solver, GEX, flip level
#   oi_signals.py      — walls, velocity, straddle, bias, gravity, magnet
#   detectors.py       — trap, vacuum, wall break, flip breakout,
#                        liq accel, hold the line, move probability
#   position_sizing.py — interpret_market, suggest_trade, sizing engine
#
# This file handles:
#   - print_dashboard() — minimal terminal layout
#   - ML signal display
#   - run_logger() — main loop
#   - save_rows() / archive_daily_log() — CSV persistence
#   - start_hotkey_listener() — Meta+D/T/X
# =============================================================================
#
# KNOWN BUGS / QUICK FIXES NEEDED
# --------------------------------
#   - htl variable referenced in debug block even when state.active_trade is None
#     → fix: initialize htl = None before the active trade check, gate debug
#       block on `if htl is not None`
#   - atm_row_d in debug block re-queries same df redundantly
#     → fix: reuse atm_row computed earlier in the function
#
# =============================================================================
#
# BACKLOG
# -------
#   1. TRADE LOG TO CSV
#      Persist every trade entry/exit to a trade_log.csv with:
#        entry_time, strike, option_type, entry_price, lots, exit_price,
#        exit_time, htl_verdict_at_exit, move_prob_at_entry, trade_type,
#        pnl_pts, pnl_inr
#      After 20-30 trades: backtest win rates by trade_type (directional /
#      trap_fade / flip_breakout / liq_accel) and adjust position sizing
#      multipliers accordingly.
#
#   2. EXPIRY DAY CALIBRATION
#      ACCEL_PRICE_SPEED_MIN = 3.0 is too sensitive on 0 DTE — bid/ask
#      spread noise can trigger it. Auto-scale to 5.0 when days_to_expiry == 0.
#      Similarly HTL_WALL_EXIT_PTS = 25 may be too tight on high-vol days —
#      add a volatility scalar based on straddle size relative to 30-day avg.
#
#   3. REFACTOR INTO MODULES
#      File is ~3500 lines and growing. Split into:
#        war_room.py          — main loop, dashboard, hotkeys (~300 lines)
#        gamma_engine.py      — BS gamma, IV solver, GEX, flip level
#        oi_signals.py        — walls, velocity, PCR, imbalance, momentum
#        detectors.py         — trap, vacuum, wall break, flip breakout,
#                               liq accel, move probability
#        position_sizing.py   — regime risk, expiry params, suggest_trade,
#                               hold_the_line
#        state.py             — MarketState, RegimeTracker, restore_from_csv
#      Each module independently testable. war_room.py becomes an orchestrator.
#
#   4. MOVE PROBABILITY BACKTESTING
#      Log move_prob score and direction every candle alongside actual next-candle
#      move. After 2-3 weeks: calibrate whether 60% prob actually correlates with
#      60% of candles moving in that direction. Adjust weights in MPM_WEIGHTS if
#      not well-calibrated.
#
#   5. ADAPTIVE POSITION SIZING FROM TRADE LOG
#      Once trade log (item 1) has 30+ trades, compute per-path win rate and
#      average RR. Feed back into compute_position_size() as a multiplier so
#      high-accuracy paths automatically get more size and underperforming paths
#      get throttled without manual tuning.
#
# =============================================================================

import os
import re
import sys
import csv
import time
import signal
import platform
import subprocess
import unicodedata
import atexit
from datetime import datetime, timedelta

from colorama import Fore, Style, init
from notifier import send_telegram_message
from sniper_display import sniper_display, sniper_notify

# =============================================================================
# AUTO-CAFFEINATE — prevent macOS sleep while war room is running
# =============================================================================
_caffeinate_proc = None

def _start_caffeinate():
    """
    Launch caffeinate as a child process on macOS.
    -i = prevent idle sleep, -s = prevent system sleep on AC power.
    The process dies automatically when the war room exits (atexit + signal).
    """
    global _caffeinate_proc
    if platform.system() != "Darwin":
        return  # only macOS
    try:
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-i", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("☕ caffeinate active — machine will not sleep while war room runs")
    except FileNotFoundError:
        print("⚠ caffeinate not found — sleep prevention disabled")
    except Exception as e:
        print(f"⚠ caffeinate failed: {e}")

def _stop_caffeinate():
    global _caffeinate_proc
    if _caffeinate_proc is not None:
        try:
            _caffeinate_proc.terminate()
            _caffeinate_proc.wait(timeout=3)
        except Exception:
            pass
        _caffeinate_proc = None

atexit.register(_stop_caffeinate)

# Also handle SIGTERM/SIGINT so caffeinate dies on Ctrl+C or kill
def _signal_handler(signum, frame):
    _stop_caffeinate()
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

from config import (
    MARKET_OPEN, MARKET_CLOSE, INSTRUMENT_PROFILES,
)

# ── Instrument selection via CLI arg ─────────────────────────────────────────
_instrument_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "NIFTY"
if _instrument_arg not in INSTRUMENT_PROFILES:
    print(f"Unknown instrument '{_instrument_arg}'. Choose from: {list(INSTRUMENT_PROFILES)}")
    sys.exit(1)
PROFILE = INSTRUMENT_PROFILES[_instrument_arg]
CSV_FILE              = PROFILE["csv_file"]
LOT_SIZE              = PROFILE["lot_size"]
GAMMA_FLIP_DANGER_ZONE = PROFILE["gamma_flip_danger_zone"]
print(f"▶ Running for {_instrument_arg}")
from state import state, restore_state_from_csv
from market_data import (
    load_instruments, get_spot, get_nearest_expiry,
    get_strikes, build_symbol_list, fetch_quotes, build_option_dataframe,
)
from gamma_engine import (
    compute_strike_gammas, compute_gamma_pressure, compute_gamma_wall,
    detect_gamma_flip, find_gamma_flip_level,
    compute_gamma_shift, classify_gamma_shift, detect_gamma_squeeze,
)
from oi_signals import (
    compute_oi_walls, compute_oi_change, compute_oi_imbalance, compute_pcr,
    compute_oi_velocity, detect_momentum, compute_straddle, update_straddle,
    compute_premium_gravity, compute_dealer_magnet, best_option_to_buy,
    compute_market_bias,
)
from detectors import (
    classify_option_trap, build_trap_telegram,
    breakout_countdown, detect_strike_war, detect_strike_war_break,
    detect_liquidity_vacuum, build_vacuum_telegram,
    detect_wall_break_vacuum, build_wall_break_telegram,
    detect_gamma_flip_breakout, detect_liquidity_acceleration,
    compute_next_wall_distance, hold_the_line,
    compute_move_probability,
    detect_persistent_trend, should_compute, cache_result, get_cached,
)
from position_sizing import (
    compute_trade_mode, interpret_market,
    _get_regime_risk, _get_expiry_params,
    suggest_trade,
)

init(autoreset=True)

# Global — toggled by Ctrl+D hotkey
DEBUG_MODE = False

# When False, only sniper TAKE TRADE / SEND IT alerts and trade mgmt messages fire.
# When True, individual signal alerts (vacuum, flip, trap, squeeze, etc.) also fire.
TELEGRAM_VERBOSE = False

# =============================================================================
# FORMATTING HELPERS
# =============================================================================
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')


def visible_len(s):
    clean = ansi_escape.sub('', s)
    width = 0
    for ch in clean:
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ('W', 'F') else 1
    return width


def print_two_columns(left_lines, right_lines, width=45):
    for i in range(max(len(left_lines), len(right_lines))):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        pad = max(0, width - visible_len(left))
        print(left + " " * pad + " │ " + right)


def print_divider(width=95, char="─"):
    print(char * width)


def colored_bias(bias):
    if bias == "BULLISH":
        return Fore.GREEN + "🟢 BULLISH"
    elif bias == "BEARISH":
        return Fore.RED + "🔴 BEARISH"
    else:
        return Fore.YELLOW + "🟡 RANGE"


def notify(message):
    send_telegram_message(f"[{_instrument_arg}] {message}")

# =============================================================================
# DASHBOARD
# =============================================================================
def print_dashboard(df, spot, atm, momentum_strikes, expiry):
    prev = state.previous_snapshot

    # ── Increment tick counter for throttle system ─────────────────────────
    state.tick_counter += 1

    # ── Compute all signals ────────────────────────────────────────────────
    df = compute_strike_gammas(df, spot, expiry)
    call_wall, put_wall = compute_oi_walls(df)

    # FIX: track wall history for retreat detection
    state.call_wall_history.append(call_wall)
    state.put_wall_history.append(put_wall)

    straddle = compute_straddle(df, atm)
    momentum_data = update_straddle(straddle)
    gamma = compute_gamma_pressure(df, spot, expiry, lot_size=PROFILE["lot_size"], sigma=PROFILE["gamma_sigma"])
    gamma_flip = detect_gamma_flip(gamma, state.previous_gamma)
    gamma_wall = compute_gamma_wall(df, spot, expiry, lot_size=PROFILE["lot_size"], sigma=PROFILE["gamma_sigma"])

    # Throttled signals — use cache if not due for recompute
    if should_compute("gravity"):
        gravity = compute_premium_gravity(df)
        cache_result("gravity", gravity)
    else:
        gravity = get_cached("gravity", int(df["strike"].median()))

    oi_signal = compute_oi_change(df, prev)

    if should_compute("pcr"):
        pcr_signal, pcr_val = compute_pcr(df)
        cache_result("pcr", (pcr_signal, pcr_val))
    else:
        pcr_signal, pcr_val = get_cached("pcr", ("NEUTRAL", 1.0))

    if should_compute("best_option"):
        best_call, best_put = best_option_to_buy(df, spot)
        cache_result("best_option", (best_call, best_put))
    else:
        best_call, best_put = get_cached("best_option", (atm + 100, atm - 100))

    bias, confidence = compute_market_bias(
        spot, gravity, call_wall, put_wall, oi_signal, pcr_signal
    )

    # FIX: use actual previous spot price, not call_ltp mean
    prev_spot = state.previous_spot
    # Track spot history for trend detection
    state.spot_history.append(spot)

    days_to_expiry = (expiry - datetime.now().date()).days
    momentum_5m = momentum_data["momentum_5m"] if momentum_data else None

    # FIX: Trend detection — feeds into trap suppression and trade suggestion
    expected_move = straddle / 2
    trend = detect_persistent_trend(spot, expected_move)

    # Trap — throttled to 5 minutes
    if should_compute("trap"):
        trap = classify_option_trap(
            spot=spot, prev_spot=prev_spot, call_wall=call_wall,
            put_wall=put_wall, gamma=gamma, gamma_wall=gamma_wall,
            oi_signal=oi_signal, straddle=straddle,
            straddle_momentum=momentum_5m, days_to_expiry=days_to_expiry, atm=atm,
        )
        cache_result("trap", trap)
    else:
        trap = get_cached("trap", {"type": "NONE", "confidence": 0,
                                    "fade_strike": None, "fade_type": None,
                                    "reversal_lvl": None, "reason": []})
    trap_prob = trap["confidence"]

    count, direction, strike = breakout_countdown(
        spot, call_wall, put_wall, momentum_strikes, gamma
    )

    mode = compute_trade_mode(bias, trap_prob, count, momentum_strikes, gamma)
    gamma_shift = classify_gamma_shift(compute_gamma_shift(gamma, state.previous_gamma))

    if should_compute("magnet"):
        magnet_strike, magnet_prob = compute_dealer_magnet(df, spot)
        cache_result("magnet", (magnet_strike, magnet_prob))
    else:
        magnet_strike, magnet_prob = get_cached("magnet", (atm, 50))

    flip_level = find_gamma_flip_level(df, spot, sigma=PROFILE["gamma_sigma"])
    squeeze = detect_gamma_squeeze(gamma, momentum_data, oi_signal)

    if should_compute("strike_war"):
        war_strike, war_status = detect_strike_war(df, spot)
        war_break = detect_strike_war_break(df, prev, war_strike, spot)
        cache_result("strike_war", (war_strike, war_status, war_break))
    else:
        war_strike, war_status, war_break = get_cached("strike_war", (None, None, None))

    velocity, call_oi_speed, put_oi_speed = compute_oi_velocity(
        df, prev, straddle_momentum=momentum_5m
    )

    if should_compute("oi_imbalance"):
        oi_imbalance = compute_oi_imbalance(df)
        cache_result("oi_imbalance", oi_imbalance)
    else:
        oi_imbalance = get_cached("oi_imbalance", None)

    if should_compute("vacuum"):
        vacuum = detect_liquidity_vacuum(
            df=df, prev_df=prev, spot=spot,
            gamma=gamma, straddle_momentum=momentum_5m,
        )
        cache_result("vacuum", vacuum)
    else:
        vacuum = get_cached("vacuum", {"detected": False, "status": "NONE", "score": 0,
                                        "direction": None, "reason": []})

    if should_compute("wall_break"):
        wall_break_vac = detect_wall_break_vacuum(df, prev, spot, gamma)
        cache_result("wall_break", wall_break_vac)
    else:
        wall_break_vac = get_cached("wall_break", {"detected": False, "direction": None,
                                                     "reason": []})

    liq_accel = detect_liquidity_acceleration(
        spot=spot, prev_spot=prev_spot, momentum_data=momentum_data,
        call_oi_speed=call_oi_speed, put_oi_speed=put_oi_speed,
    )
    flip_breakout = detect_gamma_flip_breakout(
        spot=spot, prev_spot=prev_spot,
        flip_level=flip_level, straddle_momentum=momentum_5m,
    )

    if should_compute("move_prob"):
        move_prob = compute_move_probability(
            gamma=gamma, momentum_data=momentum_data, velocity=velocity,
            vacuum=vacuum, wall_break=wall_break_vac, flip_breakout=flip_breakout,
            acceleration=liq_accel, momentum_strikes=momentum_strikes,
        )
        cache_result("move_prob", move_prob)
    else:
        move_prob = get_cached("move_prob", {"probability": 0, "direction": "UNCLEAR",
                                              "conviction": "LOW", "active_signals": [],
                                              "reasons": [], "conflicted": False})

    regime, action = interpret_market(
        spot, atm, bias, confidence, gamma, gamma_flip,
        trap_prob, count, momentum_data, oi_signal,
        vacuum=vacuum, velocity=velocity,
    )

    # ── Structural bypass — skip 3-candle confirmation delay ──────────────
    bypassed = False

    if flip_breakout and flip_breakout["detected"]:
        fb_dir = flip_breakout["direction"]
        regime = "VOLATILITY EXPANSION"
        action = (f"TREND FOLLOW {fb_dir} — GAMMA FLIP BREAKOUT CONFIRMED | "
                  f"Flip level {flip_breakout['flip_level']}")
        bypassed = True

    elif vacuum and vacuum["status"] == "CONFIRMED" and vacuum["score"] >= 60:
        d = vacuum["direction"]
        target = vacuum["target_wall"]
        regime = "LIQUIDITY VACUUM"
        action = f"FOLLOW {d} MOVE — TARGET {target}" if target else f"FOLLOW {d} MOVE"
        bypassed = True

    elif wall_break_vac and wall_break_vac["detected"]:
        d = wall_break_vac["direction"]
        target = wall_break_vac["target_wall"]
        regime = "LIQUIDITY VACUUM"
        action = (f"WALL BREAK — FOLLOW {d} | "
                  + (f"Target {target}" if target else "Next wall is target"))
        bypassed = True

    elif gamma < 0 and momentum_5m is not None and momentum_5m > 2.0:
        regime = "VOLATILITY EXPANSION"
        action = "TREND MODE — NEGATIVE GAMMA + IV EXPANDING"
        bypassed = True

    elif liq_accel and liq_accel["detected"] and liq_accel["conviction"] == "HIGH":
        d = liq_accel["direction"]
        regime = "VOLATILITY EXPANSION"
        action = f"LIQUIDITY ACCELERATION {d} — LAUNCH PHASE DETECTED"
        bypassed = True

    state.previous_gamma = gamma
    state.previous_spot  = spot   # FIX: store actual spot for next tick

    # ── ML signal agreement ───────────────────────────────────────────────
    if state.last_ml_result is not None:
        ml_bias_str = _ml_bias_to_str(state.last_ml_result["signal"])

        # FIX: ML kill switch — suppress ML when it's been consistently wrong
        from config import ML_CONSECUTIVE_WRONG_KILL
        ml_killed = state.ml_consecutive_wrong >= ML_CONSECUTIVE_WRONG_KILL

        if ml_killed or state.last_ml_result["signal"] == 0:
            ml_signal = "neutral"
        elif ml_bias_str == bias:
            ml_signal = "agree"
        else:
            ml_signal = "conflict"
    else:
        ml_signal = "neutral"

    # ── Gamma flip danger zone Telegram ──────────────────────────────────
    if flip_level:
        dist = abs(spot - flip_level)
        prev_dist = state.previous_flip_distance
        just_entered = (prev_dist is None or prev_dist > GAMMA_FLIP_DANGER_ZONE) and dist <= GAMMA_FLIP_DANGER_ZONE
        if just_entered and not state.flip_approach_alerted:
            if TELEGRAM_VERBOSE:
                notify(
                    f"⚡ GAMMA FLIP ZONE: Spot {spot} approaching flip level {flip_level} "
                    f"({dist:.0f}pts away) — dealer regime may change!"
                )
            state.flip_approach_alerted = True
        elif dist > GAMMA_FLIP_DANGER_ZONE * 2:
            state.flip_approach_alerted = False
        state.previous_flip_distance = dist

    # ==========================================================================
    # PRINT — minimal actionable layout
    # ==========================================================================
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 63} {ts}")

    # Row 1: Spot / ATM / Flip
    flip_tag = ""
    if flip_level:
        dist = abs(spot - flip_level)
        arrow = "↑" if spot > flip_level else "↓"
        if dist <= GAMMA_FLIP_DANGER_ZONE:
            flip_tag = Fore.LIGHTGREEN_EX + f"  Flip:⚡{flip_level}⚡({dist:.0f}pts)" + Style.RESET_ALL
        else:
            flip_tag = Fore.YELLOW + f"  Flip:{flip_level}({dist:.0f}pts{arrow})" + Style.RESET_ALL
    print(f"Spot:{round(spot, 1)}  ATM:{atm}{flip_tag}")

    # Row 2: MAP
    print(f"MAP: {put_wall} → {gravity} → {call_wall}")

    # Row 3: Straddle
    if momentum_data:
        m5 = momentum_data["momentum_5m"]
        straddle_color = Fore.YELLOW if m5 > 2 else (Fore.CYAN if m5 < -2 else "")
        print(straddle_color +
              f"Straddle:{round(straddle, 1)}  ±{round(straddle / 2, 1)}  "
              f"5m:{m5:+.1f}%  {momentum_data['status']}" + Style.RESET_ALL)
    else:
        print(f"Straddle:{round(straddle, 1)}  ±{round(straddle / 2, 1)}")

    # Row 4: Bias / Regime
    rt = state.regime_tracker
    if bypassed:
        tracker_tag = f"  {Fore.LIGHTCYAN_EX}[⚡ structural bypass]{Style.RESET_ALL}"
    elif rt.candidate_bias and rt.candidate_count > 0:
        tracker_tag = (f"  {Fore.YELLOW}({rt.candidate_bias} "
                       f"{rt.candidate_count}/{rt.min_confirm}){Style.RESET_ALL}")
    else:
        tracker_tag = f"  {Fore.WHITE}[{rt.stable_minutes}m]{Style.RESET_ALL}"
    print(f"{colored_bias(bias)}  Conf:{confidence}%{tracker_tag}  │  {regime}")

    # Conditional context
    if pcr_signal != "NEUTRAL":
        pcr_color = Fore.GREEN if pcr_signal == "BULLISH" else Fore.RED
        print(pcr_color + f"PCR:{pcr_val} ({pcr_signal})" + Style.RESET_ALL)

    atm_row = df[df["strike"] == atm]
    if not atm_row.empty:
        c_iv = atm_row["call_iv"].values[0]
        p_iv = atm_row["put_iv"].values[0]
        if c_iv is not None and p_iv is not None and abs(c_iv - p_iv) > 0.03:
            skew_color = Fore.RED if p_iv > c_iv else Fore.GREEN
            print(skew_color +
                  f"IV Skew  C:{c_iv * 100:.1f}%  P:{p_iv * 100:.1f}%  "
                  f"({'put skew — bearish lean' if p_iv > c_iv else 'call skew — bullish lean'})"
                  + Style.RESET_ALL)

    if oi_imbalance:
        print(Fore.BLUE + f"OI: {oi_imbalance}" + Style.RESET_ALL)
    if abs(spot - gamma_wall) <= 50:
        print(Fore.BLUE + f"Gamma Wall:{gamma_wall} ({abs(spot - gamma_wall):.0f}pts)" + Style.RESET_ALL)
    if war_strike:
        print(Fore.YELLOW + f"Strike War: {war_strike}" + Style.RESET_ALL)

    # ==========================================================================
    # TRIGGERS
    # ==========================================================================
    print_divider()
    any_trigger = False

    # Move probability — only when >= 60
    if move_prob["probability"] >= 60:
        prob = move_prob["probability"]
        d = move_prob["direction"]
        conv = move_prob["conviction"]
        bar = "█" * (prob // 10) + "░" * (10 - prob // 10)
        conflict_tag = " ⚠CONFLICT" if move_prob["conflicted"] else ""
        mpm_color = Fore.RED if conv == "VERY HIGH" else Fore.YELLOW
        print(mpm_color +
              f"📊 MOVE PROB:{prob}%  [{bar}]  {d}  {conv}{conflict_tag}"
              + Style.RESET_ALL)
        any_trigger = True

    # Vacuum
    if vacuum and vacuum["detected"]:
        score = vacuum["score"]
        status = vacuum["status"]
        d = vacuum["direction"]
        target = vacuum["target_wall"]
        target_str = f"  → {target}" if target else ""
        vac_color = Fore.RED if status == "CONFIRMED" else Fore.YELLOW
        emoji = "🌪" if status == "CONFIRMED" else "⚠"
        print(vac_color + f"{emoji} VACUUM {status}  {d}  Score:{score}{target_str}" + Style.RESET_ALL)
        any_trigger = True
        vac_key = f"{vacuum['direction']}_{vacuum['score'] // 20}"
        if state.vacuum_alerted != vac_key and status == "CONFIRMED":
            if TELEGRAM_VERBOSE:
                tg = build_vacuum_telegram(vacuum, spot)
                if tg:
                    notify(tg)
            state.vacuum_alerted = vac_key
    else:
        state.vacuum_alerted = None

    # Wall break vacuum
    if wall_break_vac and wall_break_vac["detected"]:
        wb = wall_break_vac
        target_str = f"  → {wb['target_wall']}" if wb["target_wall"] else ""
        print(Fore.RED +
              f"🌪 WALL BREAK  {wb['direction']}  "
              f"Wall:{wb['broken_wall']} lost {wb['oi_drop_pct'] * 100:.0f}%{target_str}"
              + Style.RESET_ALL)
        any_trigger = True
        wb_key = f"wb_{wb['broken_wall']}"
        if state.vacuum_alerted != wb_key:
            if TELEGRAM_VERBOSE:
                tg = build_wall_break_telegram(wb, spot)
                if tg:
                    notify(tg)
            state.vacuum_alerted = wb_key

    # Gamma flip breakout
    if flip_breakout and flip_breakout["detected"]:
        fb_dir = flip_breakout["direction"]
        fb_lvl = flip_breakout["flip_level"]
        print(Fore.LIGHTCYAN_EX +
              f"⚡ GAMMA FLIP BREAKOUT  {fb_dir}  Crossed:{fb_lvl}"
              + Style.RESET_ALL)
        any_trigger = True
        if not state.gamma_flip_alerted and TELEGRAM_VERBOSE:
            notify(
                f"⚡ GAMMA FLIP BREAKOUT {fb_dir} | "
                f"Spot {spot} crossed {fb_lvl} with IV expanding"
            )
            state.gamma_flip_alerted = True
    else:
        state.gamma_flip_alerted = False  # reset only flip breakout flag, not approach flag

    # Trap — only >= 60
    if trap["confidence"] >= 60:
        t = trap["type"]
        conf = trap["confidence"]
        fade = trap["fade_strike"]
        rev = trap["reversal_lvl"]
        ft = trap["fade_type"]
        rev_str = f"  Rev:{rev}" if rev else ""
        t_color = Fore.RED if conf >= 80 else Fore.YELLOW
        t_emoji = "🔥" if conf >= 80 else "🪤"
        print(t_color + f"{t_emoji} {t}  {conf}%  │  Fade:{fade} {ft}{rev_str}" + Style.RESET_ALL)
        any_trigger = True
        trap_key = f"{t}_{conf // 20}"
        if state.trap_alerted != trap_key:
            if TELEGRAM_VERBOSE:
                tg = build_trap_telegram(trap, spot)
                if tg:
                    notify(tg)
            state.trap_alerted = trap_key
    else:
        state.trap_alerted = None

    # Strike war break
    if war_break:
        msg = f"⚠ WAR BREAK: {war_break}  Spot:{spot}"
        print(Fore.RED + msg + Style.RESET_ALL)
        if TELEGRAM_VERBOSE:
            notify(msg)
        any_trigger = True

    # Gamma squeeze
    if squeeze:
        print(Fore.RED + f"🚀 {squeeze}" + Style.RESET_ALL)
        if TELEGRAM_VERBOSE:
            notify(f"🚀 GAMMA SQUEEZE: {squeeze} | Spot {spot}")
        any_trigger = True

    # OI velocity — SURGE only
    if velocity and "SURGE" in velocity and "CONFLICTED" not in velocity:
        vel_color = Fore.GREEN if "BULLISH" in velocity else Fore.RED
        speed_str = (f"  [{call_oi_speed:+,.0f} / {put_oi_speed:+,.0f}]"
                     if call_oi_speed is not None else "")
        print(vel_color + f"⚡ {velocity}{speed_str}" + Style.RESET_ALL)
        if TELEGRAM_VERBOSE:
            notify(
                f"⚡ {'BULLISH' if 'BULLISH' in velocity else 'BEARISH'} FLOW: "
                f"{velocity} | Spot {spot}"
            )
        any_trigger = True

    # Liquidity acceleration
    if liq_accel and liq_accel["detected"]:
        d = liq_accel["direction"]
        conv = liq_accel["conviction"]
        sc = liq_accel["score"]
        la_color = Fore.RED if conv == "HIGH" else Fore.YELLOW
        emoji = "🚀" if conv == "HIGH" else "⚡"
        print(la_color + f"{emoji} ACCEL {d}  Score:{sc}  {conv}" + Style.RESET_ALL)
        any_trigger = True
        accel_key = f"{d}_{sc // 25}"
        if state.liq_accel_alerted != accel_key and conv == "HIGH" and TELEGRAM_VERBOSE:
            notify(
                f"🚀 LIQUIDITY ACCELERATION {d} | Score:{sc} | Spot {spot}"
            )
            state.liq_accel_alerted = accel_key
    else:
        state.liq_accel_alerted = None

    if not any_trigger:
        print(Fore.WHITE + "—" + Style.RESET_ALL)

    # ==========================================================================
    # ACTION + TRADE
    # ==========================================================================
    print_divider()

    # Show trend status if detected
    if trend and trend["trending"]:
        trend_color = Fore.GREEN if trend["direction"] == "UP" else Fore.RED
        trend_arrow = "📈" if trend["direction"] == "UP" else "📉"
        print(trend_color +
              f"{trend_arrow} TREND: {trend['direction']} "
              f"{trend['move_pts']}pts over {trend['duration_minutes']}min"
              + Style.RESET_ALL)

    print(Fore.GREEN + f"ACTION: {action}" + Style.RESET_ALL)

    # ML
    if state.last_ml_result is not None:
        lr = state.last_ml_result
        ml_color = (Fore.GREEN if lr["signal"] == 1
                    else Fore.RED if lr["signal"] == -1
        else Fore.WHITE)
        mins_left = 15 - (datetime.now().minute % 15)
        agree_tag = ""
        if lr["signal"] != 0:
            if _ml_bias_to_str(lr["signal"]) == bias:
                agree_tag = f"  {Fore.GREEN}✅ AGREES{Style.RESET_ALL}"
            else:
                agree_tag = f"  {Fore.YELLOW}⚠ CONFLICTS{Style.RESET_ALL}"
        print(ml_color +
              f"ML: {lr['label']}  ({lr['confidence']:.0%})  ±{lr['x_points']:.0f}pts"
              + Style.RESET_ALL + agree_tag + f"  next:{mins_left}m")

    # Hold the line — active trade OR shadow mode on last suggestion
    htl = None
    htl_source = None   # "active" or "shadow"

    if state.active_trade is not None:
        # ── Active trade: full HTL with exit logic ───────────────────────
        htl_source = "active"
        at = state.active_trade
        wall_distance = compute_next_wall_distance(df, spot, at["option_type"])
        htl = hold_the_line(
            gamma=gamma, momentum_data=momentum_data,
            next_wall_distance=wall_distance,
            trade_direction=at["option_type"],
            oi_signal=oi_signal, prev_gamma=state.previous_gamma,
        )
        verdict = htl["verdict"]
        htl_color = (Fore.GREEN if verdict == "HOLD"
                     else Fore.YELLOW if verdict == "TRAIL"
        else Fore.RED)
        htl_emoji = {"HOLD": "📈", "TRAIL": "⚠", "EXIT": "🚨"}[verdict]
        top_reason = (htl["exit_reasons"] or htl["trail_reasons"] or htl["hold_reasons"])
        reason_str = f"  {top_reason[0][:60]}" if top_reason else ""
        print(htl_color +
              f"{htl_emoji} HTL: {verdict}  Score:{htl['hold_score']}  "
              f"Stop:{htl['stop_suggestion']}{reason_str}" + Style.RESET_ALL)
        print(f"   Active: {at['strike']} {at['option_type']} "
              f"@ ₹{at['entry_price']:.0f}  entered {at.get('entry_time', '?')}")

        if verdict == "EXIT":
            notify(
                f"🚨 EXIT: {at['strike']} {at['option_type']} | "
                f"Score:{htl['hold_score']} | "
                + " | ".join(htl["exit_reasons"][:2])
            )
            state.active_trade = None

    elif state.last_suggestion is not None:
        # ── Shadow HTL: no logged trade, but we have a suggestion ────────
        # Shows what HTL *would* say if you'd entered on the last suggestion.
        # No auto-exit, no Telegram — informational only.
        htl_source = "shadow"
        ls = state.last_suggestion
        shadow_dir = ls["option_type"]   # "CE" or "PE"
        wall_distance = compute_next_wall_distance(df, spot, shadow_dir)
        htl = hold_the_line(
            gamma=gamma, momentum_data=momentum_data,
            next_wall_distance=wall_distance,
            trade_direction=shadow_dir,
            oi_signal=oi_signal, prev_gamma=state.previous_gamma,
        )
        verdict = htl["verdict"]
        htl_color = (Fore.GREEN if verdict == "HOLD"
                     else Fore.YELLOW if verdict == "TRAIL"
        else Fore.RED)
        htl_emoji = {"HOLD": "📈", "TRAIL": "⚠", "EXIT": "🚨"}[verdict]
        top_reason = (htl["exit_reasons"] or htl["trail_reasons"] or htl["hold_reasons"])
        reason_str = f"  {top_reason[0][:60]}" if top_reason else ""
        print(htl_color +
              f"{htl_emoji} SHADOW HTL: {verdict}  Score:{htl['hold_score']}  "
              f"{reason_str}" + Style.RESET_ALL)
        print(Fore.WHITE +
              f"   (tracking {ls['strike']} {ls['option_type']} "
              f"₹{ls['price']:.0f} — not logged, press Meta+I to enter)"
              + Style.RESET_ALL)

    # SNIPER — decides WHETHER to trade and DIRECTION
    sniper_result = sniper_display(
        spot=spot, bias=bias, confidence=confidence,
        gamma=gamma, straddle=straddle, momentum_data=momentum_data,
        move_prob=move_prob, trap=trap, velocity=velocity,
        vacuum=vacuum, wall_break_vac=wall_break_vac,
        flip_breakout=flip_breakout, liq_accel=liq_accel,
        squeeze=squeeze, trend=trend,
        call_wall=call_wall, put_wall=put_wall,
        flip_level=flip_level, regime=regime,
        trade=None, days_to_expiry=days_to_expiry,
        call_oi_speed=call_oi_speed, put_oi_speed=put_oi_speed,
        gamma_shift=gamma_shift, notify_fn=None, debug=DEBUG_MODE
    )

    # TRADE SUGGESTION — only when sniper endorses
    trade = None
    sniper_action = sniper_result.get("action", "") if sniper_result else ""

    if sniper_action in ("TAKE TRADE", "SEND IT") and state.active_trade is None:
        sniper_dir = sniper_result["direction"]   # "LONG" or "SHORT"
        trade_direction = "CALL" if sniper_dir == "LONG" else "PUT"
        trade = suggest_trade(
            spot=spot, straddle=straddle, direction=trade_direction,
            df=df, gamma=gamma, flip_level=flip_level,
            regime=regime, confidence=confidence,
            ml_signal=ml_signal, days_to_expiry=days_to_expiry,
            sniper_setup=sniper_result.get("setup"),
        )

    if trade:
        t = trade
        print(f"💡 {t['strike']} {t['option_type']}  "
              f"₹{t['price']:.0f}  ×{t['lots']}lot  "
              f"Stop:₹{t['stop']:.0f}  Target:₹{t['target']:.0f}  "
              f"Risk:₹{t['risk']:,.0f}")
        print(f"  {Fore.WHITE}Meta+I / Ctrl+T → log entry  │  Meta+X / Ctrl+X → exit{Style.RESET_ALL}")
        state.last_suggestion = trade

        # Telegram with full trade details
        icon = "🎯🎯🎯" if sniper_action == "SEND IT" else "🎯"
        sniper_notify(
            notify,
            action=sniper_action,
            message=(
                f"{icon} SNIPER {sniper_action} | "
                f"{sniper_result['setup']} {sniper_result['direction']} | "
                f"Score:{int(round(sniper_result['score']))}/10 ({sniper_result['confidence']}) | "
                f"Spot:{spot} | "
                f"💡 {t['strike']} {t['option_type']} "
                f"₹{t['price']:.0f} ×{t['lots']} "
                f"Stop:₹{t['stop']:.0f} Target:₹{t['target']:.0f}"
            ),
        )
    elif sniper_action in ("TAKE TRADE", "SEND IT"):
        # Sniper endorsed but no viable strike/sizing — still alert
        icon = "🎯🎯🎯" if sniper_action == "SEND IT" else "🎯"
        sniper_notify(
            notify,
            action=sniper_action,
            message=(
                f"{icon} SNIPER {sniper_action} | "
                f"{sniper_result['setup']} {sniper_result['direction']} | "
                f"Score:{int(round(sniper_result['score']))}/10 ({sniper_result['confidence']}) | "
                f"Spot:{spot} | ⚠ no viable strike"
            ),
        )

    # Clear last_suggestion when sniper rejects (but keep during STALK/LOCKED)
    if sniper_action not in ("TAKE TRADE", "SEND IT", "STALK — WAIT FOR TRIGGER", "LOCKED") \
            and state.active_trade is None:
        state.last_suggestion = None

    print("=" * 63)

    # ==========================================================================
    # DEBUG — gated on Ctrl+D
    # ==========================================================================
    if DEBUG_MODE:
        print_divider(char="-")
        print(f"GEX:{int(gamma):,}  Dealer:{gamma_shift}  OI:{oi_signal}")
        print(f"Magnet:{magnet_strike}({magnet_prob}%)  GammaWall:{gamma_wall}  Mode:{mode}")
        # NEW: trend and wall retreat debug
        if trend and trend["trending"]:
            print(Fore.CYAN + f"TREND: {trend['direction']} {trend['move_pts']}pts "
                  f"over {trend['duration_minutes']}min" + Style.RESET_ALL)
        from detectors import detect_wall_retreat
        cw_ret, cw_cnt = detect_wall_retreat("call")
        pw_ret, pw_cnt = detect_wall_retreat("put")
        if cw_ret or pw_ret:
            print(f"Wall retreat: call={'YES' if cw_ret else 'no'}({cw_cnt})  "
                  f"put={'YES' if pw_ret else 'no'}({pw_cnt})")
        if state.ml_consecutive_wrong >= 3:
            print(Fore.YELLOW + f"ML streak: {state.ml_consecutive_wrong} wrong in a row"
                  + Style.RESET_ALL)
        if pcr_signal == "NEUTRAL":
            print(f"PCR:{pcr_val} (NEUTRAL)")
        if momentum_strikes:
            print(Fore.CYAN + f"Momentum strikes: {momentum_strikes}" + Style.RESET_ALL)
        if momentum_data and momentum_data.get("momentum_15m") is not None:
            print(f"Straddle 15m: {momentum_data['momentum_15m']:.1f}%")
        if gamma_flip:
            print(Fore.RED + f"Gamma flip event: {gamma_flip}" + Style.RESET_ALL)
        if count >= 2:
            print(Fore.YELLOW + f"Breakout countdown: {direction}  cycles:{count}" + Style.RESET_ALL)
        if trap["confidence"] >= 40:
            print(Fore.MAGENTA + f"Trap: {trap['type']} {trap['confidence']}%" + Style.RESET_ALL)
            for r in trap["reason"]:
                print(f"   {r}")
        if vacuum and vacuum["detected"]:
            print("Vacuum:")
            for r in vacuum["reason"]:
                print(f"   {r}")
        if wall_break_vac and wall_break_vac["detected"]:
            print("Wall break:")
            for r in wall_break_vac["reason"]:
                print(f"   {r}")
        if flip_breakout and flip_breakout["detected"]:
            print("Flip breakout:")
            for r in flip_breakout["reason"]:
                print(f"   {r}")
        if liq_accel and liq_accel["detected"]:
            print("Accel:")
            for r in liq_accel["reason"]:
                print(f"   {r}")
        if move_prob["probability"] > 0:
            print(f"Move prob: {move_prob['probability']}%  {move_prob['direction']}")
            for r in move_prob["reasons"]:
                print(f"   {r}")
        if velocity:
            spd = (f"  [C:{call_oi_speed:+,.0f}  P:{put_oi_speed:+,.0f}]"
                   if call_oi_speed is not None else "")
            print(f"OI velocity: {velocity}{spd}")
        if trade:
            print("Trade reasoning:")
            for r in trade["reasoning"]:
                print(f"   {r}")
        if htl is not None:
            print("HTL detail:")
            for r in htl.get("exit_reasons", []):
                print(f"   ❌ {r}")
            for r in htl.get("trail_reasons", []):
                print(f"   ⚠ {r}")
            for r in htl.get("hold_reasons", []):
                print(f"   ✅ {r}")
        atm_row_d = df[df["strike"] == atm]
        if not atm_row_d.empty:
            c_iv_d = atm_row_d["call_iv"].values[0]
            p_iv_d = atm_row_d["put_iv"].values[0]
            if c_iv_d is not None and p_iv_d is not None:
                print(f"ATM IV  C:{c_iv_d * 100:.1f}%  P:{p_iv_d * 100:.1f}%")
        print(f"Best call: {best_call} CE  │  Best put: {best_put} PE")
        print_divider(char="-")

    # ── Persist all derived signals for post-session debrief ──────────────
    _ml_conf = None
    if state.last_ml_result is not None and state.last_ml_result["signal"] != 0:
        _ml_conf = round(state.last_ml_result["confidence"], 3)

    save_signals(
        spot=spot, atm=atm, gamma=gamma, straddle=straddle,
        bias=bias, confidence=confidence, regime=regime, action=action,
        call_wall=call_wall, put_wall=put_wall,
        flip_level=flip_level, gravity=gravity, pcr_val=pcr_val,
        momentum_5m=momentum_5m,
        momentum_15m=momentum_data["momentum_15m"] if momentum_data and momentum_data.get("momentum_15m") is not None else None,
        trap_type=trap["type"], trap_conf=trap_prob,
        velocity=velocity,
        move_prob_score=move_prob["probability"],
        move_prob_dir=move_prob["direction"],
        move_prob_conv=move_prob["conviction"],
        vacuum_status=vacuum["status"] if vacuum else "NONE",
        vacuum_dir=vacuum["direction"] if vacuum else None,
        vacuum_score=vacuum["score"] if vacuum else 0,
        flip_breakout_detected=flip_breakout["detected"] if flip_breakout else False,
        flip_breakout_dir=flip_breakout["direction"] if flip_breakout and flip_breakout["detected"] else None,
        liq_accel_detected=liq_accel["detected"] if liq_accel else False,
        liq_accel_dir=liq_accel["direction"] if liq_accel and liq_accel["detected"] else None,
        liq_accel_score=liq_accel["score"] if liq_accel else 0,
        wall_break_detected=wall_break_vac["detected"] if wall_break_vac else False,
        wall_break_dir=wall_break_vac["direction"] if wall_break_vac and wall_break_vac["detected"] else None,
        squeeze=squeeze,
        trend_dir=trend["direction"] if trend and trend["trending"] else None,
        trend_pts=trend["move_pts"] if trend else 0,
        trend_duration=trend["duration_minutes"] if trend else 0,
        trade_suggested=trade is not None,
        trade_strike=trade["strike"] if trade else None,
        trade_type=trade["trade_type"] if trade else None,
        trade_dir=trade["direction"] if trade else None,
        htl_verdict=htl["verdict"] if htl else None,
        htl_score=htl["hold_score"] if htl else None,
        htl_source=htl_source,
        ml_signal=ml_signal,
        ml_confidence=_ml_conf,
        days_to_expiry=days_to_expiry,
    )

    return gamma, straddle, bias, trap_prob, count


# =============================================================================
# ML HELPERS
# =============================================================================
try:
    from ml_engine import MLSignal

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠ ml_engine.py not found — running without ML predictions")


def _ml_bias_to_str(signal):
    return {1: "BULLISH", -1: "BEARISH", 0: "RANGE"}.get(signal, "RANGE")


def _print_feedback_verdicts(resolved):
    print_divider()
    print("📬 ML OUTCOME REPORT")
    for r in resolved:
        sig_label = {1: "BULLISH", -1: "BEARISH"}.get(r["signal"], "?")
        color = Fore.GREEN if r["outcome"] == "correct" else Fore.RED
        move_str = f"+{r['actual_move']:.1f}" if r["actual_move"] >= 0 else f"{r['actual_move']:.1f}"
        print(color + f"  {r['verdict']}  │  Called {sig_label} at {r['spot_at_signal']:.0f}  "
                      f"│  Moved {move_str}pts  │  Target was ±{r['x_points']:.0f}pts")
        notify(
            f"{r['verdict']} ML: called {sig_label} @ {r['spot_at_signal']:.0f} | "
            f"moved {move_str}pts | ±{r['x_points']:.0f}pts → {r['outcome'].upper()}"
        )

        # FIX: Track consecutive wrong for ML kill switch
        if r["outcome"] == "wrong":
            state.ml_consecutive_wrong += 1
        else:
            state.ml_consecutive_wrong = 0

        from config import ML_CONSECUTIVE_WRONG_KILL
        if state.ml_consecutive_wrong >= ML_CONSECUTIVE_WRONG_KILL:
            print(Fore.YELLOW +
                  f"  ⚠ ML KILLED: {state.ml_consecutive_wrong} consecutive wrong — "
                  f"suppressing to neutral until correct"
                  + Style.RESET_ALL)


# =============================================================================
# TIMING
# =============================================================================
def wait_until_next_minute():
    now = datetime.now()
    next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    wait_sec = (next_run - now).total_seconds()
    print(f"Waiting {int(wait_sec)}s until next candle...")
    time.sleep(wait_sec)


def is_market_open():
    return MARKET_OPEN <= datetime.now().time() <= MARKET_CLOSE


# =============================================================================
# SIGNALS LOG — one row per minute with all derived signals for debrief
# =============================================================================
_SIGNALS_CSV = None   # set at first call based on instrument + date

def save_signals(spot, atm, gamma, straddle, bias, confidence, regime, action,
                 call_wall, put_wall, flip_level, gravity, pcr_val,
                 momentum_5m, momentum_15m, trap_type, trap_conf,
                 velocity, move_prob_score, move_prob_dir, move_prob_conv,
                 vacuum_status, vacuum_dir, vacuum_score,
                 flip_breakout_detected, flip_breakout_dir,
                 liq_accel_detected, liq_accel_dir, liq_accel_score,
                 wall_break_detected, wall_break_dir,
                 squeeze, trend_dir, trend_pts, trend_duration,
                 trade_suggested, trade_strike, trade_type, trade_dir,
                 htl_verdict, htl_score, htl_source,
                 ml_signal, ml_confidence, days_to_expiry):
    """
    Append one row of derived signals to the daily signals log.
    Separate from the raw options CSV — this is for post-session analysis.
    """
    global _SIGNALS_CSV
    now = datetime.now()
    date_str = now.strftime("%d%m%Y")

    if _SIGNALS_CSV is None:
        _SIGNALS_CSV = f"data/signals_log_{_instrument_arg.lower()}_{date_str}.csv"

    file_exists = os.path.exists(_SIGNALS_CSV)

    with open(_SIGNALS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "spot", "atm", "gamma_pressure", "straddle",
                "bias", "confidence", "regime", "action",
                "call_wall", "put_wall", "flip_level", "gravity", "pcr",
                "straddle_mom_5m", "straddle_mom_15m",
                "trap_type", "trap_confidence",
                "oi_velocity", "move_prob", "move_prob_dir", "move_prob_conv",
                "vacuum_status", "vacuum_dir", "vacuum_score",
                "flip_breakout", "flip_breakout_dir",
                "liq_accel", "liq_accel_dir", "liq_accel_score",
                "wall_break", "wall_break_dir",
                "squeeze", "trend_dir", "trend_pts", "trend_duration_min",
                "trade_suggested", "trade_strike", "trade_type", "trade_dir",
                "htl_verdict", "htl_score", "htl_source",
                "ml_signal", "ml_confidence", "days_to_expiry",
            ])
        writer.writerow([
            now.strftime("%H:%M:%S"), spot, atm, int(gamma), round(straddle, 1),
            bias, confidence, regime, action,
            call_wall, put_wall, flip_level, gravity, pcr_val,
            momentum_5m, momentum_15m,
            trap_type, trap_conf,
            velocity or "", move_prob_score, move_prob_dir, move_prob_conv,
            vacuum_status, vacuum_dir or "", vacuum_score,
            flip_breakout_detected, flip_breakout_dir or "",
            liq_accel_detected, liq_accel_dir or "", liq_accel_score,
            wall_break_detected, wall_break_dir or "",
            squeeze or "", trend_dir or "", trend_pts, trend_duration,
            trade_suggested, trade_strike or "", trade_type or "", trade_dir or "",
            htl_verdict or "", htl_score or "", htl_source or "",
            ml_signal, ml_confidence or "", days_to_expiry,
        ])


# =============================================================================
# CSV LOGGER
# =============================================================================
def save_rows(rows, spot, atm, expiry, gamma, straddle, bias, trap_prob, counter):
    now = datetime.now().replace(second=0, microsecond=0)
    days_to_exp = (expiry - datetime.now().date()).days
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "spot", "atm_strike", "strike",
                "option_type", "ltp", "oi", "volume", "expiry",
                "days_to_expiry", "gamma_pressure", "straddle",
                "market_bias", "trap_probability", "breakout_cycles"
            ])
        for r in rows:
            sym = r["symbol"]
            strike = int(sym[-7:-2])
            opt_type = sym[-2:]
            writer.writerow([
                now, sym, spot, atm, strike, opt_type,
                r["ltp"], r["oi"], r["volume"], expiry, days_to_exp,
                gamma, straddle, bias, trap_prob, counter
            ])


# =============================================================================
# END-OF-DAY ARCHIVE
# =============================================================================
def archive_daily_log():
    if not os.path.exists(CSV_FILE):
        print("No log file to archive.")
        return

    date_str = datetime.now().strftime("%d%m%Y")
    archived = f"data/options_log_1min_{_instrument_arg.lower()}_{date_str}.csv"

    if os.path.exists(archived):
        print(f"Archive already exists: {archived} — skipping.")
        return

    os.rename(CSV_FILE, archived)
    print(f"📁 Log archived → {archived}")
    notify(f"📁 Day complete. Log saved as `{archived}`")


# =============================================================================
# MAIN LOOP
# =============================================================================
def run_logger():
    instruments = load_instruments(
        exchange=PROFILE["exchange"],
        cache_file=PROFILE["cache_file"],
    )
    consecutive_errors = 0
    MAX_ERRORS = 5
    archived_today = False

    ml = MLSignal(instrument=_instrument_arg.lower()) if ML_AVAILABLE else None

    print("✅ Options Intelligence Terminal Started")

    while True:
        now = datetime.now()

        if now.time() >= MARKET_CLOSE and not archived_today:
            archive_daily_log()
            archived_today = True
            global _SIGNALS_CSV
            _SIGNALS_CSV = None   # reset so next day gets a fresh file
            state.reset_session()
            if ml is not None:
                try:
                    print("🔄 Retraining ML model...")
                    ml.engine.rolling_retrain(lookback_days=30)
                    ml.ready = True
                    print("✅ ML retrained")
                except Exception as e:
                    print(f"⚠ ML retrain failed: {e}")
            print("State reset. Waiting for next market open...")
            time.sleep(60)
            continue

        if now.time() < MARKET_OPEN:
            archived_today = False

        if not is_market_open():
            print(f"Market closed. Waiting... ({now.strftime('%H:%M:%S')})")
            time.sleep(60)
            continue

        try:
            print(f"\nSnapshot: {now.strftime('%H:%M:%S')}")

            expiry = get_nearest_expiry(instruments, name=PROFILE["name"])
            spot = get_spot(spot_symbol=PROFILE["spot_symbol"])
            atm, strikes = get_strikes(spot, strike_step=PROFILE["strike_step"])
            symbols = build_symbol_list(
                instruments, expiry, strikes,
                name=PROFILE["name"], exchange=PROFILE["exchange"],
            )
            rows = fetch_quotes(symbols)
            df = build_option_dataframe(rows, spot)
            momentum_strikes = detect_momentum(df, state.previous_snapshot)

            gamma, straddle, bias, trap_prob, counter = print_dashboard(
                df, spot, atm, momentum_strikes, expiry
            )

            state.previous_snapshot = df.copy()
            save_rows(rows, spot, atm, expiry, gamma, straddle, bias, trap_prob, counter)

            if ml is not None:
                minutes_since_open = (now.hour * 60 + now.minute) - (9 * 60 + 15)
                days_to_expiry = (expiry - now.date()).days
                ml_result = ml.on_tick(
                    df=df, spot=spot, atm=atm, gamma=gamma,
                    straddle=straddle, trap_prob=trap_prob,
                    breakout_cycles=counter, bias=bias,
                    minutes_since_open=minutes_since_open,
                    days_to_expiry=days_to_expiry,
                )

                if isinstance(ml_result, dict) and "resolved" in ml_result:
                    _print_feedback_verdicts(ml_result["resolved"])

                if ml_result is not None and "resolved" not in ml_result:
                    state.last_ml_result = ml_result
                    # Inline ML display — compressed for clean layout
                    lr = ml_result
                    ml_color = (Fore.GREEN if lr["signal"] == 1
                                else Fore.RED if lr["signal"] == -1
                    else Fore.WHITE)
                    agree_tag = ""
                    if lr["signal"] != 0:
                        agree_tag = (f"  {Fore.GREEN}✅" if _ml_bias_to_str(lr["signal"]) == bias
                                     else f"  {Fore.YELLOW}⚠")
                    print(ml_color +
                          f"ML: {lr['label']}  ({lr['confidence']:.0%})  "
                          f"±{lr['x_points']:.0f}pts  🟢{lr['p_bullish']:.0%}  "
                          f"🔴{lr['p_bearish']:.0%}"
                          + Style.RESET_ALL + agree_tag)

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            print(f"⚠ Error ({consecutive_errors}/{MAX_ERRORS}): {e}")
            if consecutive_errors >= MAX_ERRORS:
                notify(
                    f"🚨 WAR ROOM DOWN — {consecutive_errors} errors. Last: {e}"
                )
                consecutive_errors = 0

        wait_until_next_minute()


# =============================================================================
# HOTKEYS
# =============================================================================
def start_hotkey_listener():
    """
    Hotkeys — supports BOTH Meta and Ctrl variants:
      Ctrl+D / Meta+D  → toggle debug
      Ctrl+T / Meta+I  → log trade entry
      Ctrl+X / Meta+X  → exit trade

    Meta+D sends ^D (EOF) to macOS terminal — we catch and ignore it.
    The Ctrl variants are cleaner on macOS.
    """
    try:
        from pynput import keyboard

        _held = set()

        META_KEYS = set()
        for _k in ["cmd", "cmd_l", "cmd_r", "meta", "meta_l", "meta_r"]:
            try:
                META_KEYS.add(getattr(keyboard.Key, _k))
            except AttributeError:
                pass

        CTRL_KEYS = set()
        for _k in ["ctrl", "ctrl_l", "ctrl_r"]:
            try:
                CTRL_KEYS.add(getattr(keyboard.Key, _k))
            except AttributeError:
                pass

        def on_press(key):
            global DEBUG_MODE
            _held.add(key)
            meta_held = bool(_held & META_KEYS)
            ctrl_held = bool(_held & CTRL_KEYS)
            try:
                char = key.char if hasattr(key, "char") else None
            except Exception:
                char = None

            # Debug toggle: Meta+D or Ctrl+D
            if (meta_held and char == "d") or (ctrl_held and char == "\x04"):
                DEBUG_MODE = not DEBUG_MODE
                status = Fore.CYAN + "🔍 DEBUG ON" if DEBUG_MODE else Fore.WHITE + "🔕 DEBUG OFF"
                print(f"\n{'─' * 40}\n  {status}{Style.RESET_ALL}  (takes effect next print)\n{'─' * 40}")

            # Trade entry: Meta+I or Ctrl+T
            elif (meta_held and char == "i") or (ctrl_held and char == "\x14"):
                _register_trade_entry()

            # Trade exit: Meta+X or Ctrl+X
            elif (meta_held and char == "x") or (ctrl_held and char == "\x18"):
                _register_trade_exit()

        def on_release(key):
            _held.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release,
                                     suppress=False)
        listener.daemon = True
        listener.start()
        print("⌨  Meta+I / Ctrl+T: log entry  │  Meta+X / Ctrl+X: exit  │  Meta+D / Ctrl+D: debug")

    except ImportError:
        print("⚠  pynput not installed — pip install pynput")
    except Exception as e:
        print(f"⚠  Hotkey failed: {e}")


def _register_trade_entry():
    now = datetime.now().strftime("%H:%M")
    if state.last_suggestion is not None:
        s = state.last_suggestion
        state.active_trade = {
            "strike": s["strike"],
            "option_type": s["option_type"],
            "entry_price": s["price"],
            "entry_time": now,
            "lots": s["lots"],
            "trade_type": s.get("trade_type", "directional"),
        }
        msg = (f"✅ ENTERED: {s['strike']} {s['option_type']} "
               f"₹{s['price']:.0f} ×{s['lots']} | {now}")
        print(f"\n{'─' * 50}\n  {Fore.GREEN}{msg}{Style.RESET_ALL}\n{'─' * 50}")
        notify(
            f"🟢 ENTRY: {s['strike']} {s['option_type']} @ ₹{s['price']:.0f} "
            f"| ×{s['lots']} | Stop ₹{s['stop']:.0f} | Target ₹{s['target']:.0f} | {now}"
        )
    else:
        # No suggestion cached — offer manual entry via /dev/tty
        # (avoids EOF from Meta+key poisoning stdin)
        import threading
        import sys

        def _manual():
            tty = None
            try:
                try:
                    tty = open("/dev/tty", "r")
                    read_line = tty.readline
                except OSError:
                    tty = None
                    read_line = sys.stdin.readline

                def prompt(msg):
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    line = read_line()
                    if not line:
                        raise EOFError("stdin closed")
                    return line.strip()

                print(f"\n{'─' * 50}")
                print(f"  {Fore.YELLOW}No suggestion cached.{Style.RESET_ALL} Enter details:")
                strike      = int(prompt("  Strike: "))
                opt_type    = prompt("  CE/PE: ").upper()
                entry_price = float(prompt("  Entry price: "))
                lots        = int(prompt("  Lots: "))
                state.active_trade = {
                    "strike": strike, "option_type": opt_type,
                    "entry_price": entry_price, "entry_time": now,
                    "lots": lots, "trade_type": "manual",
                }
                msg = f"✅ MANUAL: {strike} {opt_type} ₹{entry_price:.0f} ×{lots} | {now}"
                print(f"  {Fore.GREEN}{msg}{Style.RESET_ALL}\n{'─' * 50}")
                notify(
                    f"🟢 MANUAL ENTRY: {strike} {opt_type} @ ₹{entry_price:.0f} | ×{lots} | {now}"
                )
            except (EOFError, ValueError) as e:
                print(f"  ⚠ Entry cancelled: {e}")
            finally:
                if tty:
                    tty.close()

        threading.Thread(target=_manual, daemon=True).start()


def _prompt_tty(tty, prompt):
    """Print prompt to stdout, read from /dev/tty."""
    import sys
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = tty.readline()
    if not line:
        raise EOFError("tty closed")
    return line.strip()


def _register_trade_exit():
    if state.active_trade is None:
        print(f"\n{'─' * 40}\n  {Fore.YELLOW}No active trade{Style.RESET_ALL}\n{'─' * 40}")
        return
    at = state.active_trade
    now = datetime.now().strftime("%H:%M")
    print(f"\n{'─' * 50}\n  {Fore.RED}🚨 EXITED MANUALLY{Style.RESET_ALL}")
    print(f"  {at['strike']} {at['option_type']} | "
          f"Entry ₹{at['entry_price']:.0f} @ {at['entry_time']} | Exited {now}")
    print(f"{'─' * 50}")
    notify(
        f"🔴 EXIT (manual): {at['strike']} {at['option_type']} | "
        f"Entry ₹{at['entry_price']:.0f} @ {at['entry_time']} | Exited {now}"
    )
    state.active_trade = None


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    _start_caffeinate()
    start_hotkey_listener()
    restore_state_from_csv(csv_file=CSV_FILE)
    run_logger()