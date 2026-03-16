"""
detectors.py — Signal detectors, formatters, and Telegram builders.
Trap classifier, vacuums, wall break, flip breakout, liquidity acceleration,
hold the line, move probability.
"""

import numpy as np
import pandas as pd
from colorama import Fore

from config import (
    OI_DESERT_THRESHOLD, WALL_DISSOLVE_RATE, VACUUM_MIN_WIDTH,
    WALL_BREAK_OI_DROP, FLIP_BREAKOUT_PROXIMITY, FLIP_BREAKOUT_IV_MIN,
    ACCEL_PRICE_SPEED_MIN, ACCEL_PRICE_SPEED_HIGH, ACCEL_IV_MIN,
    ACCEL_IV_HIGH, ACCEL_SCORE_THRESHOLD,
    HTL_IV_HOLD_MIN, HTL_IV_EXIT_MIN, HTL_WALL_TRAIL_PTS,
    HTL_WALL_EXIT_PTS, HTL_GAMMA_TRAIL,
    MPM_WEIGHTS, MPM_GAMMA_AMP, MPM_GAMMA_DAMP, MPM_CONFLICT_PEN,
)
from state import state


# =============================================================================
# TRAP CLASSIFIER
# =============================================================================

def classify_option_trap(spot, prev_spot, call_wall, put_wall, gamma,
                         gamma_wall, oi_signal, straddle, straddle_momentum,
                         days_to_expiry, atm):
    """
    Classifies market condition into BULL TRAP / BEAR TRAP / PIN TRAP / NONE.
    Gamma acts as a score multiplier — the causal engine behind all traps.
    Negative gamma suppresses confidence by 60% (trap = breakout in neg gamma).
    """
    expected_move = straddle / 2
    result = {
        "type": "NONE", "confidence": 0,
        "fade_strike": None, "fade_type": None,
        "reversal_lvl": None, "reason": []
    }

    spot_rising  = prev_spot is not None and spot > prev_spot
    spot_falling = prev_spot is not None and spot < prev_spot
    dist_call    = abs(spot - call_wall)
    dist_put     = abs(spot - put_wall)

    call_writing  = "Call Writing"  in oi_signal
    call_covering = "Call Covering" in oi_signal
    put_writing   = "Put Writing"   in oi_signal
    put_covering  = "Put Covering"  in oi_signal or "Put Unwinding" in oi_signal

    iv_compressing = straddle_momentum is not None and straddle_momentum < -0.5

    if   gamma > 0:  gamma_mult = 1.5
    elif gamma == 0: gamma_mult = 1.0
    else:            gamma_mult = 0.6

    # BULL TRAP
    bull_score, bull_reason = 0, []
    if spot_rising and dist_call < 80:
        bull_score += 25
        bull_reason.append(f"Spot {spot} rising into Call Wall {call_wall}")
    if dist_call < 40:
        bull_score += 15
        bull_reason.append(f"Within {dist_call:.0f}pts of Call Wall — supply zone")
    if call_writing:
        bull_score += 20
        bull_reason.append("Call writers adding boldly — not scared of breakout")
    if put_covering:
        bull_score += 15
        bull_reason.append("Put writers covering — no fear below")
    if iv_compressing:
        bull_score += 15
        bull_reason.append(f"Straddle compressing {straddle_momentum:.1f}% — no energy behind rally")
    if abs(spot - gamma_wall) < 50:
        bull_score += 10
        bull_reason.append(f"Gamma Wall {gamma_wall} nearby — dealer resistance concentrated here")
    bull_score = min(int(bull_score * gamma_mult), 100)

    # BEAR TRAP
    bear_score, bear_reason = 0, []
    if spot_falling and dist_put < 80:
        bear_score += 25
        bear_reason.append(f"Spot {spot} falling into Put Wall {put_wall}")
    if dist_put < 40:
        bear_score += 15
        bear_reason.append(f"Within {dist_put:.0f}pts of Put Wall — support zone")
    if put_writing:
        bear_score += 20
        bear_reason.append("Put writers adding boldly — not scared of breakdown")
    if call_covering:
        bear_score += 15
        bear_reason.append("Call writers covering — no fear above")
    if iv_compressing:
        bear_score += 15
        bear_reason.append(f"Straddle compressing {straddle_momentum:.1f}% — no energy behind selloff")
    if abs(spot - gamma_wall) < 50:
        bear_score += 10
        bear_reason.append(f"Gamma Wall {gamma_wall} nearby — dealer support concentrated here")
    bear_score = min(int(bear_score * gamma_mult), 100)

    # PIN TRAP
    pin_score, pin_reason = 0, []
    if days_to_expiry <= 1:
        pin_score += 15
        pin_reason.append("Expiry day — pin forces are strongest")
    if gamma > 0 and abs(spot - atm) < 30:
        pin_score += 35
        pin_reason.append(f"Positive gamma pinning spot near ATM {atm}")
    if iv_compressing:
        pin_score += 25
        pin_reason.append("Premium decaying — time pin active")
    if call_writing and put_writing:
        pin_score += 15
        pin_reason.append("Both sides writing — market makers pinning the strike")
    if abs(spot - atm) < 15:
        pin_score += 10
        pin_reason.append(f"Spot within 15pts of ATM — deep in pin zone")
    pin_score = min(pin_score, 100)

    scores     = {"BULL TRAP": bull_score, "BEAR TRAP": bear_score, "PIN TRAP": pin_score}
    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]

    if gamma < 0:
        best_score = int(best_score * 0.4)

    if best_score < 40:
        return result

    result["type"]       = best_type
    result["confidence"] = best_score

    if best_type == "BULL TRAP":
        result["reason"]       = bull_reason
        result["fade_strike"]  = call_wall
        result["fade_type"]    = "CE"
        result["reversal_lvl"] = round(call_wall - expected_move)
    elif best_type == "BEAR TRAP":
        result["reason"]       = bear_reason
        result["fade_strike"]  = put_wall
        result["fade_type"]    = "PE"
        result["reversal_lvl"] = round(put_wall + expected_move)
    elif best_type == "PIN TRAP":
        result["reason"]       = pin_reason
        result["fade_strike"]  = atm
        result["fade_type"]    = "STRADDLE"
        result["reversal_lvl"] = None

    return result


def build_trap_telegram(trap, spot):
    if trap["type"] == "NONE":
        return None
    t, conf, fade, rev, ft = (trap["type"], trap["confidence"],
                               trap["fade_strike"], trap["reversal_lvl"], trap["fade_type"])
    emoji = "🔥" if conf >= 80 else "🪤"
    msg   = f"{emoji} {t} @ {spot} | Conf: {conf}% | Fade: {fade} {ft}"
    if rev:
        direction = "below" if t == "BULL TRAP" else "above"
        msg += f" | Watch reversal {direction} {rev}"
    msg += f" | {'Gamma pinning' if t == 'PIN TRAP' else 'Gamma forcing dealers to resist move'}"
    return msg


# =============================================================================
# BREAKOUT COUNTDOWN + STRIKE WAR
# =============================================================================

def breakout_countdown(spot, call_wall, put_wall, momentum_strikes, gamma):
    threshold = 30
    if abs(call_wall - spot) < threshold:
        if state.breakout_direction == "UPSIDE":
            state.breakout_counter += 1
        else:
            state.breakout_direction = "UPSIDE"
            state.breakout_strike    = call_wall
            state.breakout_counter   = 1
    elif abs(spot - put_wall) < threshold:
        if state.breakout_direction == "DOWNSIDE":
            state.breakout_counter += 1
        else:
            state.breakout_direction = "DOWNSIDE"
            state.breakout_strike    = put_wall
            state.breakout_counter   = 1
    else:
        state.breakout_counter   = 0
        state.breakout_direction = None
        state.breakout_strike    = None
    return state.breakout_counter, state.breakout_direction, state.breakout_strike


def detect_strike_war(df, spot):
    df = df.copy()
    df["total_oi"] = df["call_oi"] + df["put_oi"]
    df["distance"] = abs(df["strike"] - spot)
    near_df = df[df["distance"] < 100]
    if near_df.empty:
        return None, None
    war_row       = near_df.loc[near_df["total_oi"].idxmax()]
    balance_ratio = min(war_row["call_oi"], war_row["put_oi"]) / max(war_row["call_oi"], war_row["put_oi"])
    return (war_row["strike"], "STRIKE WAR") if balance_ratio > 0.7 else (None, None)


def detect_strike_war_break(df, prev_df, war_strike, spot):
    if prev_df is None or war_strike is None:
        return None
    row      = df[df["strike"] == war_strike]
    prev_row = prev_df[prev_df["strike"] == war_strike]
    if row.empty or prev_row.empty:
        return None
    call_oi     = row["call_oi"].values[0]
    put_oi      = row["put_oi"].values[0]
    call_change = call_oi - prev_row["call_oi"].values[0]
    put_change  = put_oi  - prev_row["put_oi"].values[0]
    imbalance   = abs(call_oi - put_oi) / (call_oi + put_oi + 1)
    if imbalance > 0.35:
        if call_change < -50_000: return "UPSIDE WAR BREAK"
        if put_change  < -50_000: return "DOWNSIDE WAR BREAK"
    return None


# =============================================================================
# LIQUIDITY VACUUM
# =============================================================================

def detect_liquidity_vacuum(df, prev_df, spot, gamma, straddle_momentum):
    result = {
        "detected": False, "status": "NONE", "score": 0,
        "desert_start": None, "desert_end": None, "desert_width": 0,
        "target_wall": None, "wall_dissolving": False, "dissolve_pct": 0.0,
        "direction": None, "reason": [],
    }

    df = df.sort_values("strike").copy()
    df["total_oi"] = df["call_oi"] + df["put_oi"]
    max_oi = df["total_oi"].max()
    if max_oi == 0:
        return result

    df["oi_pct"] = df["total_oi"] / max_oi
    above_spot   = df[df["strike"] > spot].copy()
    below_spot   = df[df["strike"] < spot].copy()

    def find_desert_ahead(side_df, direction):
        if side_df.empty:
            return None, None, None
        desert_strikes = []
        for _, row in side_df.iterrows():
            if row["oi_pct"] < OI_DESERT_THRESHOLD:
                desert_strikes.append(row["strike"])
            else:
                break
        if not desert_strikes:
            return None, None, None
        d_start = desert_strikes[0]
        d_end   = desert_strikes[-1]
        beyond  = side_df[side_df["strike"] > d_end] if direction == "UPSIDE" \
                  else side_df[side_df["strike"] < d_start]
        target  = (beyond.iloc[0]["strike"] if direction == "UPSIDE" else beyond.iloc[-1]["strike"]) \
                  if not beyond.empty else None
        return d_start, d_end, target

    up_start, up_end, up_target = find_desert_ahead(above_spot, "UPSIDE")
    dn_start, dn_end, dn_target = find_desert_ahead(
        below_spot.sort_values("strike", ascending=False), "DOWNSIDE"
    )

    up_width = (up_end - up_start) if up_start is not None else 0
    dn_width = (dn_start - dn_end) if dn_start is not None else 0

    if up_width == 0 and dn_width == 0:
        return result

    if up_width >= dn_width:
        direction, desert_start, desert_end, desert_width, target_wall = \
            "UPSIDE", up_start, up_end, up_width, up_target
    else:
        direction, desert_start, desert_end, desert_width, target_wall = \
            "DOWNSIDE", dn_end, dn_start, dn_width, dn_target

    if desert_width < VACUUM_MIN_WIDTH:
        return result

    score, reason = 30, []
    reason.append(f"OI desert: {desert_start}–{desert_end} ({desert_width}pts of near-zero OI)")

    if desert_width >= 150:
        score += 20
        reason.append(f"Wide vacuum ({desert_width}pts) — extended free-run zone")
    elif desert_width >= 100:
        score += 10
        reason.append(f"Moderate vacuum ({desert_width}pts)")

    wall_dissolving, dissolve_pct = False, 0.0
    if prev_df is not None:
        behind = df[df["strike"] <= spot] if direction == "UPSIDE" else df[df["strike"] >= spot]
        if not behind.empty:
            wall_candidate = behind.loc[behind["total_oi"].idxmax(), "strike"]
            prev_row = prev_df[prev_df["strike"] == wall_candidate]
            curr_row = df[df["strike"] == wall_candidate]
            if not prev_row.empty and not curr_row.empty:
                prev_oi = prev_row["call_oi"].values[0] + prev_row["put_oi"].values[0]
                curr_oi = curr_row["call_oi"].values[0] + curr_row["put_oi"].values[0]
                if prev_oi > 0:
                    dissolve_pct = (prev_oi - curr_oi) / prev_oi
                    if dissolve_pct >= WALL_DISSOLVE_RATE:
                        wall_dissolving = True
                        score += 25
                        reason.append(
                            f"Wall at {wall_candidate} dissolving "
                            f"({dissolve_pct*100:.0f}% OI lost) — support gone"
                        )

    if straddle_momentum is not None and straddle_momentum > 1.5:
        score += 15
        reason.append(f"Straddle expanding {straddle_momentum:.1f}% — IV confirming move")

    if gamma < 0:
        score  = min(int(score * 1.4), 100)
        status = "CONFIRMED" if score >= 60 else "WARNING"
        reason.append("Negative gamma — dealers AMPLIFYING move, no absorption")
    elif gamma > 0:
        score  = max(int(score * 0.6), 0)
        status = "WARNING"
        reason.append("Positive gamma — dealers absorbing move, vacuum dampened")
    else:
        status = "WARNING"

    if score < 40:
        return result

    result.update({
        "detected": True, "status": status, "score": score,
        "desert_start": desert_start, "desert_end": desert_end,
        "desert_width": desert_width, "target_wall": target_wall,
        "wall_dissolving": wall_dissolving, "dissolve_pct": dissolve_pct,
        "direction": direction, "reason": reason,
    })
    return result


def build_vacuum_telegram(vac, spot):
    if not vac["detected"] or vac["status"] != "CONFIRMED":
        return None
    emoji = "🌪" if vac["score"] >= 80 else "⚡"
    msg   = (f"{emoji} LIQUIDITY VACUUM | {vac['direction']} | "
             f"Desert: {vac['desert_start']}–{vac['desert_end']} ({vac['desert_width']}pts) | "
             f"Neg Gamma confirmed — dealers amplifying")
    if vac["target_wall"]:
        msg += f" | Target wall: {vac['target_wall']}"
    if vac["wall_dissolving"]:
        msg += f" | Wall dissolving {vac['dissolve_pct']*100:.0f}% — no support behind"
    return msg


# =============================================================================
# WALL BREAK VACUUM
# =============================================================================

def detect_wall_break_vacuum(df, prev_df, spot, gamma):
    result = {
        "detected": False, "direction": None, "broken_wall": None,
        "oi_drop_pct": 0.0, "target_wall": None, "reason": [],
    }

    if prev_df is None:
        return result

    df     = df.sort_values("strike").copy()
    merged = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    if merged.empty:
        return result

    merged["total_oi"]      = merged["call_oi"]     + merged["put_oi"]
    merged["total_oi_prev"] = merged["call_oi_prev"] + merged["put_oi_prev"]
    oi_threshold = merged["total_oi_prev"].quantile(0.70)

    for _, row in merged.iterrows():
        prev_oi = row["total_oi_prev"]
        curr_oi = row["total_oi"]
        strike  = row["strike"]

        if prev_oi < oi_threshold or prev_oi == 0:
            continue
        drop_pct = (prev_oi - curr_oi) / prev_oi
        if drop_pct < WALL_BREAK_OI_DROP:
            continue

        direction = "UPSIDE" if strike <= spot else "DOWNSIDE"

        beyond = df[df["strike"] > strike].copy() if direction == "UPSIDE" \
                 else df[df["strike"] < strike].sort_values("strike", ascending=False).copy()
        beyond["total_oi"] = beyond["call_oi"] + beyond["put_oi"]
        target_wall = None
        if not beyond.empty:
            significant = beyond[beyond["total_oi"] >= oi_threshold]
            if not significant.empty:
                target_wall = significant.iloc[0]["strike"]

        reason = [
            f"Wall at {strike} lost {drop_pct*100:.0f}% OI — institutional defence gone",
            f"Spot {spot} has crossed {strike} {'above' if direction == 'UPSIDE' else 'below'}",
        ]
        if gamma < 0:
            reason.append("Negative gamma — dealers amplifying move through broken wall")
        if target_wall:
            reason.append(f"Next wall: {target_wall} — expect acceleration to here")

        result.update({
            "detected": True, "direction": direction, "broken_wall": strike,
            "oi_drop_pct": drop_pct, "target_wall": target_wall, "reason": reason,
        })
        return result

    return result


def build_wall_break_telegram(wb, spot):
    if not wb["detected"]:
        return None
    return (f"🌪 WALL BREAK VACUUM | {wb['direction']} | "
            f"Wall {wb['broken_wall']} lost {wb['oi_drop_pct']*100:.0f}% OI | "
            f"Spot {spot} crossed — fast trend active"
            + (f" | Target: {wb['target_wall']}" if wb["target_wall"] else ""))


# =============================================================================
# GAMMA FLIP BREAKOUT
# =============================================================================

def detect_gamma_flip_breakout(spot, prev_spot, flip_level, straddle_momentum):
    result = {"detected": False, "direction": None,
              "flip_level": flip_level, "reason": []}

    if flip_level is None or prev_spot is None or straddle_momentum is None:
        return result

    if abs(spot - flip_level) > FLIP_BREAKOUT_PROXIMITY:
        return result
    if straddle_momentum < FLIP_BREAKOUT_IV_MIN:
        return result

    crossed_up   = prev_spot < flip_level <= spot
    crossed_down = prev_spot > flip_level >= spot

    if not crossed_up and not crossed_down:
        return result

    direction = "UPSIDE" if crossed_up else "DOWNSIDE"
    result.update({
        "detected": True, "direction": direction, "flip_level": flip_level,
        "reason": [
            f"Spot crossed gamma flip {flip_level} {'upward' if crossed_up else 'downward'}",
            f"Straddle expanding {straddle_momentum:.1f}% — IV confirming regime change",
            f"Dealer behaviour flipping: {'now absorbing moves' if crossed_up else 'now amplifying moves'}",
        ],
    })
    return result


# =============================================================================
# LIQUIDITY ACCELERATION
# =============================================================================

def detect_liquidity_acceleration(spot, prev_spot, momentum_data,
                                   call_oi_speed, put_oi_speed):
    result = {
        "detected": False, "score": 0, "direction": None,
        "conviction": None, "reason": [],
    }

    if prev_spot is None or momentum_data is None:
        return result

    score, reason = 0, []
    price_speed   = spot - prev_spot

    if abs(price_speed) >= ACCEL_PRICE_SPEED_HIGH:
        score += 35
        reason.append(f"Price accelerating {price_speed:+.1f}pts — strong velocity")
    elif abs(price_speed) >= ACCEL_PRICE_SPEED_MIN:
        score += 20
        reason.append(f"Price moving {price_speed:+.1f}pts — building momentum")
    else:
        return result

    price_direction = "UPSIDE" if price_speed > 0 else "DOWNSIDE"
    iv_mom = momentum_data["momentum_5m"]

    if iv_mom >= ACCEL_IV_HIGH:
        score += 35
        reason.append(f"Straddle expanding {iv_mom:.1f}% — institutions actively positioning")
    elif iv_mom >= ACCEL_IV_MIN:
        score += 20
        reason.append(f"Straddle expanding {iv_mom:.1f}% — premium confirming move")

    if call_oi_speed is not None and put_oi_speed is not None:
        hist = np.array(state.oi_velocity_history) if len(state.oi_velocity_history) >= 5 else None
        oi_threshold = (hist.mean() + 1.5 * hist.std()) if hist is not None else 150_000

        if price_direction == "UPSIDE" and call_oi_speed < -oi_threshold:
            score += 30
            reason.append(
                f"Call covering surge [{call_oi_speed:+,.0f}] — supply above spot evaporating"
            )
        elif price_direction == "DOWNSIDE" and put_oi_speed < -oi_threshold:
            score += 30
            reason.append(
                f"Put covering surge [{put_oi_speed:+,.0f}] — support below spot evaporating"
            )
        elif abs(call_oi_speed) > oi_threshold * 0.6 or abs(put_oi_speed) > oi_threshold * 0.6:
            score += 15
            reason.append("OI flow building — partial confirmation")

    if score < ACCEL_SCORE_THRESHOLD:
        return result

    conviction = "HIGH" if score >= 75 else "MODERATE"
    result.update({
        "detected": True, "score": min(score, 100),
        "direction": price_direction, "conviction": conviction, "reason": reason,
    })
    return result


# =============================================================================
# HOLD THE LINE
# =============================================================================

def compute_next_wall_distance(df, spot, trade_direction):
    """Distance in pts to nearest significant OI wall in trade direction."""
    df = df.sort_values("strike").copy()
    df["total_oi"]  = df["call_oi"] + df["put_oi"]
    oi_threshold    = df["total_oi"].quantile(0.70)
    significant     = df[df["total_oi"] >= oi_threshold]

    if trade_direction == "CALL":
        ahead = significant[significant["strike"] > spot]
        return float(ahead["strike"].iloc[0] - spot) if not ahead.empty else 999
    else:
        ahead = significant[significant["strike"] < spot]
        return float(spot - ahead["strike"].iloc[-1]) if not ahead.empty else 999


def hold_the_line(gamma, momentum_data, next_wall_distance,
                  trade_direction, oi_signal, prev_gamma=None):
    result = {
        "verdict": "HOLD", "hold_score": 100,
        "exit_reasons": [], "trail_reasons": [], "hold_reasons": [],
        "stop_suggestion": "ORIGINAL",
    }

    if momentum_data is None:
        return result

    iv_mom     = momentum_data["momentum_5m"]
    exit_flags = []
    trail_flags = []
    hold_flags  = []
    deductions  = 0

    # EXIT conditions
    if iv_mom < HTL_IV_EXIT_MIN:
        exit_flags.append(f"IV collapsing {iv_mom:.1f}% — move is done")
        deductions += 40

    if gamma > 0:
        just_flipped = prev_gamma is not None and prev_gamma <= 0
        if just_flipped:
            exit_flags.append("Gamma just flipped positive — dealers switched to absorbing")
            deductions += 35
        else:
            trail_flags.append("Gamma positive — dealers absorbing moves, reduce conviction")
            deductions += 20

    if next_wall_distance <= HTL_WALL_EXIT_PTS:
        exit_flags.append(
            f"Next wall only {next_wall_distance:.0f}pts away — take profit"
        )
        deductions += 35

    if trade_direction == "CALL":
        if "Call Writing" in oi_signal and "Put Unwinding" in oi_signal:
            exit_flags.append("Call writers re-entering — smart money fading the move")
            deductions += 30
    else:
        if "Put Writing" in oi_signal and "Call Unwinding" in oi_signal:
            exit_flags.append("Put writers re-entering — smart money fading the move")
            deductions += 30

    # TRAIL conditions
    if HTL_IV_EXIT_MIN <= iv_mom < HTL_IV_HOLD_MIN:
        trail_flags.append(f"IV slowing to {iv_mom:.1f}% — move aging, tighten stop")
        deductions += 20

    if HTL_WALL_EXIT_PTS < next_wall_distance <= HTL_WALL_TRAIL_PTS:
        trail_flags.append(
            f"Next wall {next_wall_distance:.0f}pts away — trail stop to breakeven"
        )
        deductions += 15

    if HTL_GAMMA_TRAIL <= gamma <= 0.0 and gamma != 0:
        trail_flags.append("Gamma near zero — regime transitioning")
        deductions += 10

    # HOLD conditions
    if iv_mom >= HTL_IV_HOLD_MIN:
        hold_flags.append(f"IV expanding {iv_mom:.1f}% — move still has energy")
    if gamma < 0:
        hold_flags.append("Negative gamma — dealers amplifying, let it run")
    if next_wall_distance > HTL_WALL_TRAIL_PTS:
        hold_flags.append(f"Next wall {next_wall_distance:.0f}pts away — clear runway")

    hold_score = max(0, 100 - deductions)

    if exit_flags:
        verdict, stop_suggestion = "EXIT", "CLOSE"
    elif trail_flags and hold_score < 60:
        verdict = "TRAIL"
        stop_suggestion = "TRAIL_TIGHT" if hold_score < 40 else "BREAKEVEN"
    elif trail_flags:
        verdict, stop_suggestion = "TRAIL", "BREAKEVEN"
    else:
        verdict, stop_suggestion = "HOLD", "ORIGINAL"

    result.update({
        "verdict": verdict, "hold_score": hold_score,
        "exit_reasons": exit_flags, "trail_reasons": trail_flags,
        "hold_reasons": hold_flags, "stop_suggestion": stop_suggestion,
    })
    return result


# =============================================================================
# MOVE PROBABILITY METER
# =============================================================================

def compute_move_probability(gamma, momentum_data, velocity, vacuum,
                              wall_break, flip_breakout, acceleration,
                              momentum_strikes):
    result = {
        "probability": 0, "direction": "UNCLEAR", "conviction": "LOW",
        "active_signals": [], "reasons": [], "conflicted": False,
    }

    signals, reasons = [], []

    if flip_breakout and flip_breakout["detected"]:
        signals.append(("flip_breakout", MPM_WEIGHTS["flip_breakout"], flip_breakout["direction"]))
        reasons.append(f"Gamma flip breakout {flip_breakout['direction']} (+{MPM_WEIGHTS['flip_breakout']})")

    if acceleration and acceleration["detected"]:
        signals.append(("liq_accel", MPM_WEIGHTS["liq_accel"], acceleration["direction"]))
        reasons.append(
            f"Liquidity acceleration {acceleration['direction']} "
            f"(+{MPM_WEIGHTS['liq_accel']}) [{acceleration['conviction']}]"
        )

    vac_active = vacuum and vacuum["detected"]
    wb_active  = wall_break and wall_break["detected"]

    if vac_active and wb_active and vacuum["direction"] == wall_break["direction"]:
        signals.append(("vacuum",     MPM_WEIGHTS["vacuum"],                vacuum["direction"]))
        signals.append(("wall_break", MPM_WEIGHTS["wall_break"] // 2,       wall_break["direction"]))
        reasons.append(
            f"Vacuum + wall break {vacuum['direction']} "
            f"(+{MPM_WEIGHTS['vacuum'] + MPM_WEIGHTS['wall_break'] // 2}) [overlapping]"
        )
    else:
        if vac_active:
            signals.append(("vacuum", MPM_WEIGHTS["vacuum"], vacuum["direction"]))
            reasons.append(f"Liquidity vacuum {vacuum['direction']} (+{MPM_WEIGHTS['vacuum']})")
        if wb_active:
            signals.append(("wall_break", MPM_WEIGHTS["wall_break"], wall_break["direction"]))
            reasons.append(f"Wall break {wall_break['direction']} (+{MPM_WEIGHTS['wall_break']})")

    if momentum_data and momentum_data["momentum_5m"] > 2.0:
        signals.append(("iv_expansion", MPM_WEIGHTS["iv_expansion"], None))
        reasons.append(f"IV expanding {momentum_data['momentum_5m']:.1f}% (+{MPM_WEIGHTS['iv_expansion']})")

    if velocity and "SURGE" in velocity and "CONFLICTED" not in velocity:
        vel_dir = "UPSIDE" if "BULLISH" in velocity else "DOWNSIDE"
        signals.append(("oi_surge", MPM_WEIGHTS["oi_surge"], vel_dir))
        reasons.append(f"OI velocity surge {vel_dir} (+{MPM_WEIGHTS['oi_surge']})")

    if momentum_strikes:
        signals.append(("momentum_strike", MPM_WEIGHTS["momentum_strike"], None))
        reasons.append(f"Momentum at {len(momentum_strikes)} strike(s) (+{MPM_WEIGHTS['momentum_strike']})")

    if not signals:
        return result

    dir_scores = {"UPSIDE": 0, "DOWNSIDE": 0}
    for _, w, d in signals:
        if d in dir_scores:
            dir_scores[d] += w

    if dir_scores["UPSIDE"] == dir_scores["DOWNSIDE"] == 0:
        dominant_dir, conflicted = "UNCLEAR", False
    elif dir_scores["UPSIDE"] == dir_scores["DOWNSIDE"]:
        dominant_dir, conflicted = "UNCLEAR", True
    else:
        dominant_dir = max(dir_scores, key=dir_scores.get)
        minority     = min(dir_scores, key=dir_scores.get)
        total_dir    = dir_scores["UPSIDE"] + dir_scores["DOWNSIDE"]
        conflicted   = total_dir > 0 and dir_scores[minority] / total_dir > 0.30

    raw_score = sum(w for _, w, _ in signals)

    if conflicted:
        raw_score = int(raw_score * MPM_CONFLICT_PEN)
        reasons.append(f"⚠ Direction conflict (penalty ×{MPM_CONFLICT_PEN})")

    if gamma < 0:
        raw_score = int(raw_score * MPM_GAMMA_AMP)
        reasons.append(f"Negative gamma amplifier ×{MPM_GAMMA_AMP}")
    elif gamma > 0:
        raw_score = int(raw_score * MPM_GAMMA_DAMP)
        reasons.append(f"Positive gamma dampener ×{MPM_GAMMA_DAMP}")

    probability = min(raw_score, 100)

    if   probability >= 80: conviction = "VERY HIGH"
    elif probability >= 60: conviction = "HIGH"
    elif probability >= 40: conviction = "MODERATE"
    else:                   conviction = "LOW"

    result.update({
        "probability": probability, "direction": dominant_dir,
        "conviction": conviction, "active_signals": signals,
        "reasons": reasons, "conflicted": conflicted,
    })
    return result