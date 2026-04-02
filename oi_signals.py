"""
oi_signals.py — OI-based market signals.
Walls, velocity, PCR, straddle, gravity, magnet, bias.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from state import state


def compute_oi_walls(df):
    """
    Gamma-weighted wall computation with 3-strike cluster smoothing.
    Falls back to raw OI if BS gamma columns are absent.
    """
    df = df.sort_values("strike").copy()

    if "call_gamma_bs" in df.columns and "put_gamma_bs" in df.columns:
        df["call_wall_strength"] = df["call_gamma_bs"] * df["call_oi"]
        df["put_wall_strength"]  = df["put_gamma_bs"]  * df["put_oi"]
    else:
        df["call_wall_strength"] = df["call_oi"].astype(float)
        df["put_wall_strength"]  = df["put_oi"].astype(float)

    df["call_cluster"] = df["call_wall_strength"].rolling(3, center=True, min_periods=1).sum()
    df["put_cluster"]  = df["put_wall_strength"].rolling(3, center=True,  min_periods=1).sum()

    call_wall = df.loc[df["call_cluster"].idxmax(), "strike"]
    put_wall  = df.loc[df["put_cluster"].idxmax(),  "strike"]
    return call_wall, put_wall


def compute_oi_change(df, prev_df):
    if prev_df is None:
        return "No OI history yet"

    merged      = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    call_change = (merged["call_oi"] - merged["call_oi_prev"]).sum()
    put_change  = (merged["put_oi"]  - merged["put_oi_prev"]).sum()

    signal  = ""
    signal += "Call Writing | "  if call_change > 0 else "Call Covering | "
    signal += "Put Writing"      if put_change  > 0 else "Put Unwinding"
    return signal


def compute_oi_imbalance(df):
    total_call = df["call_oi"].sum()
    total_put  = df["put_oi"].sum()
    imbalance  = (total_call - total_put) / (total_call + total_put)
    if imbalance > 0.25:
        return "CALL DOMINANCE (Bearish positioning)"
    if imbalance < -0.25:
        return "PUT DOMINANCE (Bullish positioning)"
    return None


def compute_pcr(df):
    total_call_oi = df["call_oi"].sum()
    total_put_oi  = df["put_oi"].sum()
    if total_call_oi == 0:
        return "NEUTRAL", 1.0
    pcr = total_put_oi / total_call_oi
    if pcr > 1.3:
        return "BULLISH", round(pcr, 2)
    elif pcr < 0.7:
        return "BEARISH", round(pcr, 2)
    return "NEUTRAL", round(pcr, 2)


def compute_oi_velocity(df, prev_df, straddle_momentum=None, spot_change=None):
    if prev_df is None:
        return None, None, None

    merged     = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    call_speed = (merged["call_oi"] - merged["call_oi_prev"]).sum()
    put_speed  = (merged["put_oi"]  - merged["put_oi_prev"]).sum()

    state.oi_velocity_history.append(abs(call_speed) + abs(put_speed))

    if len(state.oi_velocity_history) >= 5:
        hist      = np.array(state.oi_velocity_history)
        threshold = hist.mean() + 1.5 * hist.std()
    else:
        threshold = 150_000

    signal = None
    if   call_speed < -threshold: signal = "BULLISH SURGE (Call Covering)"
    elif put_speed  >  threshold: signal = "BULLISH BUILDUP (Put Writing)"
    elif put_speed  < -threshold: signal = "BEARISH SURGE (Put Covering)"
    elif call_speed >  threshold: signal = "BEARISH BUILDUP (Call Writing)"

    if signal is not None and straddle_momentum is not None:
        if "BULLISH" in signal and straddle_momentum >  2.0:
            signal = f"⚠ CONFLICTED ({signal} but straddle expanding)"
        elif "BEARISH" in signal and straddle_momentum < -2.0:
            signal = f"⚠ CONFLICTED ({signal} but straddle compressing)"

    return signal, call_speed, put_speed


def detect_momentum(df, prev_df):
    if prev_df is None:
        return None

    merged     = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    signals    = []
    all_vol    = merged["call_vol"] + merged["put_vol"]
    vol_thresh = all_vol.quantile(0.90) if len(all_vol) > 0 else 1

    for _, row in merged.iterrows():
        vol_now        = row["call_vol"] + row["put_vol"]
        vol_prev       = row["call_vol_prev"] + row["put_vol_prev"] + 1
        vol_ratio      = vol_now / vol_prev
        oi_change      = (row["call_oi"] + row["put_oi"]) - (row["call_oi_prev"] + row["put_oi_prev"])
        premium_change = (row["call_ltp"] + row["put_ltp"]) - (row["call_ltp_prev"] + row["put_ltp_prev"])

        if vol_ratio > 2 and vol_now > vol_thresh and oi_change > 50_000 and abs(premium_change) > 3:
            signals.append(row["strike"])

    return signals if signals else None


def compute_straddle(df, atm):
    row = df[df["strike"] == atm]
    return row["call_ltp"].values[0] + row["put_ltp"].values[0]


def update_straddle(straddle_price):
    state.straddle_history.append((datetime.now(), straddle_price))

    if len(state.straddle_history) < 6:
        return None

    current      = state.straddle_history[-1][1]
    five_min_ago = state.straddle_history[-6][1]
    momentum_5m  = (current - five_min_ago) / (five_min_ago + 1e-9) * 100

    momentum_15m = None
    if len(state.straddle_history) >= 16:
        fifteen_min_ago = state.straddle_history[-16][1]
        momentum_15m    = (current - fifteen_min_ago) / (fifteen_min_ago + 1e-9) * 100

    status = "STABLE"
    if momentum_5m > 2:
        status = "EXPANDING ⚡"
    elif momentum_5m < -2:
        status = "COMPRESSING 💤"

    acceleration = None
    if len(state.straddle_history) >= 11:
        ten_min_ago  = state.straddle_history[-11][1]
        momentum_10m = (current - ten_min_ago) / (ten_min_ago + 1e-9) * 100
        acceleration = momentum_10m > momentum_5m * 1.5

    return {
        "current":      current,
        "momentum_5m":  round(momentum_5m, 2),
        "momentum_15m": round(momentum_15m, 2) if momentum_15m is not None else None,
        "status":       status,
        "acceleration": acceleration,
    }


def compute_premium_gravity(df):
    df = df.copy()
    df["straddle"]     = df["call_ltp"] + df["put_ltp"]
    df["premium_mass"] = df["straddle"] * (df["call_oi"] + df["put_oi"])
    total_mass = df["premium_mass"].sum()
    if total_mass == 0:
        return int(df["strike"].median())
    return round((df["strike"] * df["premium_mass"]).sum() / total_mass)


def compute_dealer_magnet(df, spot, profile=None):
    df = df.copy()
    df["total_oi"]      = df["call_oi"] + df["put_oi"]
    df["total_premium"] = df["call_ltp"] + df["put_ltp"]
    df["distance"]      = abs(df["strike"] - spot)

    def norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0

    df["magnet_score"] = (
        norm(df["total_oi"])      * 0.5 +
        norm(df["total_premium"]) * 0.3 -
        norm(df["distance"])      * 0.2
    )
    row      = df.loc[df["magnet_score"].idxmax()]
    distance = abs(spot - row["strike"])
    p = profile or {}
    mag_near = p.get("magnet_near", 20)
    mag_mid  = p.get("magnet_mid", 40)
    mag_far  = p.get("magnet_far", 80)
    if   distance < mag_near: probability = 80
    elif distance < mag_mid:  probability = 60
    elif distance < mag_far:  probability = 40
    else:                     probability = 20
    return row["strike"], probability


def best_option_to_buy(df, spot):
    df = df.copy()

    def norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else pd.Series(0.5, index=s.index)

    dist_score       = 1 - norm(abs(df["strike"] - spot))
    df["call_score"] = norm(df["call_vol"]) * 0.4 + norm(df["call_oi"]) * 0.3 + dist_score * 0.3
    df["put_score"]  = norm(df["put_vol"])  * 0.4 + norm(df["put_oi"])  * 0.3 + dist_score * 0.3

    return df.loc[df["call_score"].idxmax(), "strike"], df.loc[df["put_score"].idxmax(), "strike"]


def compute_market_bias(spot, gravity, call_wall, put_wall, oi_signal, pcr_signal,
                        trend=None, session_range=None, spot_position=None):
    score = 0
    score += 1 if spot > gravity else -1
    score += 1 if abs(spot - put_wall) < abs(call_wall - spot) else -1

    if   "Call Covering" in oi_signal and "Put Writing"   in oi_signal: score += 2
    elif "Call Writing"  in oi_signal and "Put Unwinding" in oi_signal: score -= 2

    if   pcr_signal == "BULLISH": score += 2
    elif pcr_signal == "BEARISH": score -= 2

    # Price-action trend boost — when spot is trending hard, OI may lag
    # but bias should reflect reality.  Scaled by move size.
    # Session range 100+ = trending session (Nifty does 180-400pts daily;
    # 100pts is reached by ~09:45 on most days, well before the main move).
    trending_session = session_range is not None and session_range >= 100
    if trend and trend["trending"]:
        move = trend["move_pts"]
        if trending_session and move >= 60:
            # Trending session + active trend = strong directional bias
            # Don't wait for OI to confirm what price is screaming
            boost = 4
        elif move >= 100:
            boost = 3
        elif move >= 60:
            boost = 2
        elif move >= 30:
            boost = 1
        else:
            boost = 0
        score += boost if trend["direction"] == "UP" else -boost
    elif trending_session and spot_position is not None:
        # No active trend but session is clearly trending —
        # spot position within session range tells the story
        # (spot_position: 0.0 = at session low, 1.0 = at session high)
        if spot_position < 0.25:
            score -= 2   # near session low on a big range day = bearish
        elif spot_position > 0.75:
            score += 2   # near session high on a big range day = bullish

    if   score >= 3:  bias = "BULLISH"
    elif score <= -3: bias = "BEARISH"
    else:             bias = "RANGE"

    return bias, min(abs(score) * 16, 100)