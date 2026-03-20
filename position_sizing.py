"""
position_sizing.py — Trade logic, regime interpretation, position sizing.
interpret_market, suggest_trade, compute_position_size, choose_otm_strike.
"""

from config import (
    REGIME_RISK, BASE_RISK_PCT, OPTION_STOP_PCT,
    EXPIRY_STOP_PCT, EXPIRY_LOT_CAP, EXPIRY_RISK_SCALAR,
    MIN_OTM_PCT, MAX_OTM_PCT, MAX_CAPITAL_PCT,
    ACCOUNT_SIZE, LOT_SIZE, STRIKE_STEP,
)


# =============================================================================
# HELPERS
# =============================================================================

def _get_regime_risk(regime):
    return REGIME_RISK.get(regime, BASE_RISK_PCT)


def _get_expiry_params(days_to_expiry):
    stop_pct    = EXPIRY_STOP_PCT.get(days_to_expiry, OPTION_STOP_PCT)
    lot_cap     = EXPIRY_LOT_CAP.get(days_to_expiry, 999)
    risk_scalar = EXPIRY_RISK_SCALAR.get(days_to_expiry, 1.0)
    return stop_pct, lot_cap, risk_scalar


# =============================================================================
# TRADE MODE
# =============================================================================

def compute_trade_mode(bias, trap_prob, breakout_cycles, momentum_strikes, gamma):
    if trap_prob > 60 and breakout_cycles >= 2: return "BREAKOUT WATCH"
    if momentum_strikes and gamma < 0:          return "MOMENTUM TRADE"
    if bias == "RANGE":                         return "RANGE SCALP"
    if bias == "BULLISH":                       return "BUY CALLS ON DIP"
    if bias == "BEARISH":                       return "BUY PUTS ON RISE"
    return "WAIT"


# =============================================================================
# REGIME INTERPRETATION
# =============================================================================

def interpret_market(spot, atm, bias, confidence, gamma, gamma_flip,
                     trap_prob, breakout_count, momentum_data, oi_signal,
                     vacuum=None, velocity=None):
    regime = "UNCLEAR"
    action = "WAIT"

    if vacuum and vacuum["status"] == "CONFIRMED" and vacuum["score"] >= 60:
        d      = vacuum["direction"]
        target = vacuum["target_wall"]
        action = f"FOLLOW {d} MOVE — TARGET {target}" if target else f"FOLLOW {d} MOVE"
        return "LIQUIDITY VACUUM", action

    # gamma sign determines regime — not flip event
    if gamma > 0 and abs(spot - atm) < 40:
        return "GAMMA PINNING", "RANGE SCALP — AVOID BREAKOUTS"
    if gamma < 0:
        return "VOLATILITY EXPANSION", "TREND MODE — FOLLOW BREAKOUT"

    # OI Velocity SURGE overrides RANGE
    if velocity and "CONFLICTED" not in velocity:
        if "BULLISH SURGE" in velocity:
            return "VOLATILITY EXPANSION", "TREND FOLLOW — BULLISH SURGE DETECTED"
        if "BEARISH SURGE" in velocity:
            return "VOLATILITY EXPANSION", "TREND FOLLOW — BEARISH SURGE DETECTED"

    if momentum_data:
        if momentum_data["momentum_5m"] > 5:
            return "VOLATILITY EXPANSION", "BREAKOUT RISK — PREPARE FOR TREND"
        if momentum_data["momentum_5m"] < -3:
            return "VOL COMPRESSION", "RANGE MARKET — SCALP ONLY"

    if trap_prob >= 70:
        return "OPTION TRAP", f"EXPECT REVERSAL — FADE THE {trap_prob}% CONFIDENCE TRAP"

    if breakout_count >= 3:
        return "BREAKOUT PRESSURE", f"WATCH FOR {bias} BREAKOUT"

    if   confidence == 0:  regime, action = "NO EDGE",              "STAY OUT"
    elif confidence <= 16: regime, action = "WEAK STRUCTURE",       "LIGHT SCALPS ONLY"
    elif confidence <= 50: regime, action = f"MODERATE {bias}",     f"FAVOR {bias} SETUPS"
    elif confidence <= 80: regime, action = f"STRONG {bias}",       f"LOOK FOR {bias} MOMENTUM"
    else:                  regime, action = f"INSTITUTIONAL {bias}", f"HIGH CONVICTION {bias}"

    if   "Call Covering" in oi_signal and "Put Writing"   in oi_signal: action += " | BULLISH OI FLOW"
    elif "Call Writing"  in oi_signal and "Put Unwinding" in oi_signal: action += " | BEARISH OI FLOW"
    elif "Call Writing"  in oi_signal and "Put Writing"   in oi_signal: action += " | RANGE BUILDING"

    return regime, action


# =============================================================================
# STRIKE SELECTION
# =============================================================================

def choose_otm_strike(spot, expected_move, direction, gamma, flip_level, regime):
    base_pct = (MIN_OTM_PCT + MAX_OTM_PCT) / 2
    distance = expected_move * base_pct

    if gamma < 0:
        distance *= 1.15
    elif gamma > 0:
        distance *= 0.85

    if regime == "LIQUIDITY VACUUM":
        distance *= 1.20
    elif regime == "GAMMA PINNING":
        distance *= 0.70

    if direction == "CALL":
        strike = round((spot + distance) / STRIKE_STEP) * STRIKE_STEP
        if flip_level and strike > flip_level and spot < flip_level:
            strike = round(flip_level / STRIKE_STEP) * STRIKE_STEP
    else:
        strike = round((spot - distance) / STRIKE_STEP) * STRIKE_STEP
        if flip_level and strike < flip_level and spot > flip_level:
            strike = round(flip_level / STRIKE_STEP) * STRIKE_STEP

    return int(strike)


# =============================================================================
# POSITION SIZING
# =============================================================================

def compute_position_size(option_price, regime, confidence,
                          trap_confidence, ml_signal, days_to_expiry=5):
    risk_pct = _get_regime_risk(regime)
    if risk_pct == 0:
        return 0

    stop_pct, lot_cap, risk_scalar = _get_expiry_params(days_to_expiry)
    account_risk  = ACCOUNT_SIZE * risk_pct * risk_scalar
    stop_distance = option_price * stop_pct
    risk_per_lot  = stop_distance * LOT_SIZE

    if risk_per_lot == 0:
        return 0

    base_lots = account_risk / risk_per_lot
    base_lots *= max(confidence / 100, 0.25)

    if trap_confidence >= 80:
        return 0
    elif trap_confidence >= 60:
        base_lots *= 0.4
    elif trap_confidence >= 40:
        base_lots *= 0.7

    if ml_signal == "agree":
        base_lots *= 1.30
    elif ml_signal == "conflict":
        base_lots *= 0.50

    base_lots = min(base_lots, lot_cap)
    max_from_capital = (ACCOUNT_SIZE * MAX_CAPITAL_PCT) / (option_price * LOT_SIZE)
    base_lots        = min(base_lots, max_from_capital)

    if base_lots < 0.5:
        return 0

    return max(1, int(base_lots))


# =============================================================================
# SUGGEST TRADE — sniper decides direction, this computes the numbers
# =============================================================================

def suggest_trade(spot, straddle, direction, df, gamma, flip_level,
                  regime, confidence, ml_signal, days_to_expiry,
                  sniper_setup=None):
    """
    Compute trade mechanics for a sniper-endorsed direction.

    Sniper decides WHETHER to trade and the DIRECTION.
    This function computes WHAT to trade: strike, price, lots, stop, target.

    Args:
        direction: "CALL" or "PUT" (mapped from sniper's LONG/SHORT)
        sniper_setup: setup name from sniper for reasoning display
    """
    expected_move = straddle / 2

    strike = choose_otm_strike(spot, expected_move, direction, gamma, flip_level, regime)

    col = "call_ltp" if direction == "CALL" else "put_ltp"
    try:
        price = df.loc[df["strike"] == strike, col].values[0]
    except IndexError:
        nearest = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
        strike  = nearest["strike"].values[0]
        price   = nearest[col].values[0]

    if price <= 0:
        return None

    lots = compute_position_size(
        option_price    = price,
        regime          = regime,
        confidence      = confidence,
        trap_confidence = 0,   # sniper already factored trap into its score
        ml_signal       = ml_signal,
        days_to_expiry  = days_to_expiry,
    )
    if lots == 0:
        return None

    stop_pct, lot_cap, risk_scalar = _get_expiry_params(days_to_expiry)
    stop       = round(price * (1 - stop_pct), 2)
    target     = round(price * 2.0, 2)
    risk_amt   = round((price - stop) * LOT_SIZE * lots, 2)
    reward_amt = round((target - price) * LOT_SIZE * lots, 2)
    otm_pct    = round(abs(strike - spot) / expected_move * 100)
    capital    = round(price * LOT_SIZE * lots, 2)

    reasoning = []
    if sniper_setup:
        reasoning.append(f"Sniper setup: {sniper_setup}")
    reasoning.append(
        f"Regime: {regime} → risk {_get_regime_risk(regime)*100:.1f}% "
        f"× expiry scalar {risk_scalar:.2f}"
    )
    reasoning.append(f"Confidence {confidence}% applied as scalar")
    reasoning.append(
        f"Strike {abs(strike-spot):.0f}pts OTM = {otm_pct}% of ±{expected_move:.0f}pt move"
    )
    if days_to_expiry <= 2:
        reasoning.append(
            f"⚠ Expiry in {days_to_expiry}d — stop {stop_pct*100:.0f}%, "
            f"risk cut {(1-risk_scalar)*100:.0f}%, max {lot_cap} lots"
        )
    if ml_signal == "agree":
        reasoning.append("ML agrees → +30%")
    elif ml_signal == "conflict":
        reasoning.append("ML conflicts → -50%")

    return {
        "direction":   direction,
        "strike":      strike,
        "option_type": "CE" if direction == "CALL" else "PE",
        "price":       price,
        "lots":        lots,
        "stop":        stop,
        "stop_pct":    stop_pct,
        "target":      target,
        "risk":        risk_amt,
        "reward":      reward_amt,
        "otm_pct":     otm_pct,
        "capital":     capital,
        "trade_type":  trade_type,
        "reasoning":   reasoning,
    }