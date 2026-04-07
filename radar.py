# =============================================================================
# RADAR — Anticipatory Setup Scanner
#
# Runs every 3 minutes. Evaluates PREDICTIVE signals that fire BEFORE a move:
#   1. Gamma flip proximity — spot approaching flip = explosive move brewing
#   2. Vacuum ahead — OI desert in likely direction = no resistance
#   3. Wall asymmetry — one side dissolving, other firm = pressure building
#   4. Straddle compression → expansion — coiled spring about to release
#   5. OI buildup — fresh positions stacking near spot = institutional intent
#
# Output: "SETUP FORMING" alerts with direction, strength, and key levels.
# The trader watches the chart — when price confirms, they enter with conviction.
# =============================================================================

from config import INSTRUMENT_PROFILES, OI_DESERT_THRESHOLD, WALL_DISSOLVE_RATE


def _gamma_flip_proximity(spot, flip_level, profile, spot_history=None):
    """
    How close is spot to the gamma flip level?
    Near flip = dealers about to flip hedging direction = explosive move.

    Only fires if spot is APPROACHING the flip (closing the gap), not
    oscillating around it. This prevents noise when spot chops at flip level.

    Returns: dict with direction, distance, score (0-2), narrative
    """
    if flip_level is None or spot is None:
        return None

    dist = spot - flip_level
    flip_near = profile.get("sniper_flip_near", 15)
    flip_far = profile.get("sniper_flip_far", 40)

    abs_dist = abs(dist)
    if abs_dist > flip_far:
        return None

    # Check if spot is approaching flip (gap is closing), not just sitting there
    if spot_history and len(spot_history) >= 5:
        old_dist = abs(spot_history[-5] - flip_level)
        # Spot must be getting closer — at least 3pts closer over last 5 candles
        if abs_dist >= old_dist - 3:
            return None  # not approaching, just hovering

    # Direction: if spot is BELOW flip, crossing UP = bullish breakout
    # If spot is ABOVE flip, crossing DOWN = bearish breakdown
    if dist < 0:
        direction = "LONG"
        narrative = f"Spot {abs_dist:.0f}pts below gamma flip ({flip_level:.0f})"
    else:
        direction = "SHORT"
        narrative = f"Spot {abs_dist:.0f}pts above gamma flip ({flip_level:.0f})"

    if abs_dist <= flip_near:
        score = 2
        narrative += " — approaching flip, explosive move imminent"
    else:
        score = 1
        narrative += " — approaching flip zone"

    return {
        "signal": "GAMMA FLIP",
        "direction": direction,
        "score": score,
        "level": flip_level,
        "distance": abs_dist,
        "narrative": narrative,
    }


def _vacuum_ahead(df, spot, profile):
    """
    Is there a liquidity desert ahead of spot in either direction?
    No OI = no resistance = fast move once it starts.

    Returns: dict with direction, target, score (0-2), narrative
    """
    if df is None or df.empty:
        return None

    df = df.sort_values("strike").copy()
    df["total_oi"] = df["call_oi"] + df["put_oi"]
    max_oi = df["total_oi"].max()
    if max_oi == 0:
        return None

    df["oi_pct"] = df["total_oi"] / max_oi
    vacuum_min = profile.get("vacuum_min_width", 50)
    vacuum_mod = profile.get("vacuum_moderate", 100)

    results = []
    for direction, side_df in [("LONG", df[df["strike"] > spot]),
                                ("SHORT", df[df["strike"] < spot].sort_values("strike", ascending=False))]:
        if side_df.empty:
            continue
        desert_strikes = []
        for _, row in side_df.iterrows():
            if row["oi_pct"] < OI_DESERT_THRESHOLD:
                desert_strikes.append(row["strike"])
            else:
                break
        if not desert_strikes:
            continue

        width = abs(desert_strikes[-1] - desert_strikes[0]) + profile.get("strike_step", 50)
        if width < vacuum_min:
            continue

        # Find target wall on the other side of the desert
        if direction == "LONG":
            beyond = side_df[side_df["strike"] > desert_strikes[-1]]
        else:
            beyond = side_df[side_df["strike"] < desert_strikes[-1]]
        target = beyond.iloc[0]["strike"] if not beyond.empty else None

        score = 2 if width >= vacuum_mod else 1
        narrative = f"OI vacuum {direction} — {width:.0f}pt desert"
        if target:
            narrative += f", target wall at {target:.0f}"

        results.append({
            "signal": "VACUUM",
            "direction": direction,
            "score": score,
            "level": target,
            "width": width,
            "narrative": narrative,
        })

    # Return the best vacuum
    if not results:
        return None
    return max(results, key=lambda r: r["score"])


def _wall_asymmetry(spot, call_wall, put_wall, df, prev_df, profile):
    """
    Is one wall dissolving while the other holds firm?
    Dissolving call wall + firm put wall = bullish pressure building.

    Returns: dict with direction, score (0-2), narrative
    """
    if df is None or prev_df is None or call_wall is None or put_wall is None:
        return None

    merged = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    if merged.empty:
        return None

    # Check OI change at wall strikes
    call_wall_row = merged[merged["strike"] == call_wall]
    put_wall_row = merged[merged["strike"] == put_wall]

    if call_wall_row.empty or put_wall_row.empty:
        return None

    call_wall_oi = call_wall_row.iloc[0]["call_oi"]
    call_wall_prev = call_wall_row.iloc[0].get("call_oi_prev", call_wall_oi)
    put_wall_oi = put_wall_row.iloc[0]["put_oi"]
    put_wall_prev = put_wall_row.iloc[0].get("put_oi_prev", put_wall_oi)

    # Avoid division by zero
    if call_wall_prev == 0 or put_wall_prev == 0:
        return None

    call_change = (call_wall_oi - call_wall_prev) / call_wall_prev
    put_change = (put_wall_oi - put_wall_prev) / put_wall_prev

    # Distance from spot to walls
    dist_to_call = call_wall - spot
    dist_to_put = spot - put_wall

    # Call wall dissolving + put wall firm = bullish
    # Put wall dissolving + call wall firm = bearish
    dissolve_threshold = -WALL_DISSOLVE_RATE  # -30%

    direction = None
    narrative = ""
    score = 0

    if call_change < dissolve_threshold and put_change >= 0:
        direction = "LONG"
        score = 2 if call_change < dissolve_threshold * 1.5 else 1
        narrative = (f"Call wall ({call_wall:.0f}) dissolving {call_change*100:+.0f}%, "
                     f"put wall ({put_wall:.0f}) firm {put_change*100:+.0f}%")
    elif put_change < dissolve_threshold and call_change >= 0:
        direction = "SHORT"
        score = 2 if put_change < dissolve_threshold * 1.5 else 1
        narrative = (f"Put wall ({put_wall:.0f}) dissolving {put_change*100:+.0f}%, "
                     f"call wall ({call_wall:.0f}) firm {call_change*100:+.0f}%")

    if direction is None:
        return None

    return {
        "signal": "WALL ASYMMETRY",
        "direction": direction,
        "score": score,
        "level": call_wall if direction == "LONG" else put_wall,
        "narrative": narrative,
    }


def _straddle_coil(momentum_data, straddle_history):
    """
    Is IV compressed and starting to expand? Coiled spring pattern.
    Low straddle momentum for a while, then a tick up = expansion coming.

    Returns: dict with score (0-2), narrative (direction-neutral)
    """
    if momentum_data is None:
        return None

    mom_5m = momentum_data.get("momentum_5m", 0)
    mom_15m = momentum_data.get("momentum_15m")
    status = momentum_data.get("status", "")

    # We want: was compressed, now ticking up
    # Or: very low 15m momentum (compressed), 5m starting to expand
    if mom_15m is None:
        return None

    # Pattern: 15m compressed (low abs value), 5m expanding
    compressed_15m = abs(mom_15m) < 1.0  # less than 1% over 15 min = quiet
    expanding_5m = abs(mom_5m) > 1.5     # now starting to move

    if compressed_15m and expanding_5m:
        score = 2
        narrative = f"Straddle coiled — quiet 15m ({mom_15m:+.1f}%), now expanding ({mom_5m:+.1f}%)"
    elif compressed_15m and abs(mom_5m) > 0.8:
        score = 1
        narrative = f"Straddle compressed ({mom_15m:+.1f}%), early signs of expansion ({mom_5m:+.1f}%)"
    else:
        return None

    return {
        "signal": "STRADDLE COIL",
        "direction": None,  # direction-neutral — amplifies other signals
        "score": score,
        "narrative": narrative,
    }


def _oi_buildup(df, prev_df, spot, profile):
    """
    Is fresh OI stacking at strikes near spot? Institutions positioning.
    Heavy put writing near spot = bullish floor being built.
    Heavy call writing near spot = bearish ceiling being built.

    Returns: dict with direction, score (0-2), narrative
    """
    if df is None or prev_df is None:
        return None

    strike_step = profile.get("strike_step", 50)
    near_range = strike_step * 3  # look at 3 strikes around spot

    merged = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    near = merged[(merged["strike"] >= spot - near_range) &
                  (merged["strike"] <= spot + near_range)].copy()

    if near.empty:
        return None

    near["call_oi_delta"] = near["call_oi"] - near["call_oi_prev"]
    near["put_oi_delta"] = near["put_oi"] - near["put_oi_prev"]

    total_call_add = near.loc[near["call_oi_delta"] > 0, "call_oi_delta"].sum()
    total_put_add = near.loc[near["put_oi_delta"] > 0, "put_oi_delta"].sum()

    # Need meaningful buildup — at least 5% of existing OI at these strikes
    existing_oi = near["call_oi"].sum() + near["put_oi"].sum()
    if existing_oi == 0:
        return None

    call_pct = total_call_add / (existing_oi + 1e-9) * 100
    put_pct = total_put_add / (existing_oi + 1e-9) * 100

    # Thresholds: >3% = notable, >6% = heavy
    direction = None
    narrative = ""
    score = 0

    # Heavy put writing near ATM = bullish (writers expect support)
    # Heavy call writing near ATM = bearish (writers expect resistance)
    if put_pct > 3 and put_pct > call_pct * 1.5:
        direction = "LONG"
        score = 2 if put_pct > 6 else 1
        top_strike = near.loc[near["put_oi_delta"].idxmax(), "strike"]
        narrative = f"Fresh put OI buildup ({put_pct:.1f}%) near spot, heaviest at {top_strike:.0f}"
    elif call_pct > 3 and call_pct > put_pct * 1.5:
        direction = "SHORT"
        score = 2 if call_pct > 6 else 1
        top_strike = near.loc[near["call_oi_delta"].idxmax(), "strike"]
        narrative = f"Fresh call OI buildup ({call_pct:.1f}%) near spot, heaviest at {top_strike:.0f}"

    if direction is None:
        return None

    return {
        "signal": "OI BUILDUP",
        "direction": direction,
        "score": score,
        "level": top_strike,
        "narrative": narrative,
    }


# =============================================================================
# MAIN RADAR SCAN
# =============================================================================

def radar_scan(spot, flip_level, call_wall, put_wall,
               df, prev_df, momentum_data, straddle_history,
               profile=None, spot_history=None):
    """
    Run the radar scan. Call every 3 minutes.

    Returns: dict with:
        - active: bool — is there a setup forming?
        - direction: LONG / SHORT / None
        - strength: number of aligned signals (0-5)
        - signals: list of individual signal results
        - narrative: human-readable summary
        - key_level: the most important price level to watch
        - watch_for: what to look for on the chart
    """
    if profile is None:
        profile = INSTRUMENT_PROFILES.get("NIFTY", {})

    # Run all 5 predictive scans
    results = []
    has_anchor = False  # need at least one anchor signal (gamma flip or vacuum)

    flip = _gamma_flip_proximity(spot, flip_level, profile, spot_history=spot_history)
    if flip:
        results.append(flip)
        has_anchor = True

    vacuum = _vacuum_ahead(df, spot, profile)
    if vacuum:
        results.append(vacuum)
        has_anchor = True

    asymmetry = _wall_asymmetry(spot, call_wall, put_wall, df, prev_df, profile)
    if asymmetry:
        results.append(asymmetry)

    coil = _straddle_coil(momentum_data, straddle_history)
    if coil:
        results.append(coil)

    buildup = _oi_buildup(df, prev_df, spot, profile)
    if buildup:
        results.append(buildup)

    # Must have at least one anchor signal (gamma flip or vacuum)
    if not results or not has_anchor:
        return {
            "active": False,
            "direction": None,
            "strength": 0,
            "signals": [],
            "narrative": "No setup forming",
            "key_level": None,
            "watch_for": None,
        }

    # Count directional alignment
    long_score = sum(r["score"] for r in results if r["direction"] == "LONG")
    short_score = sum(r["score"] for r in results if r["direction"] == "SHORT")
    neutral_score = sum(r["score"] for r in results if r["direction"] is None)

    # Neutral signals (straddle coil) amplify the dominant direction
    if long_score > short_score:
        direction = "LONG"
        strength = long_score + neutral_score
        aligned = [r for r in results if r["direction"] in ("LONG", None)]
    elif short_score > long_score:
        direction = "SHORT"
        strength = short_score + neutral_score
        aligned = [r for r in results if r["direction"] in ("SHORT", None)]
    else:
        # Tie or all neutral — no clear setup
        if neutral_score > 0 and len(results) >= 2:
            direction = None
            strength = neutral_score
            aligned = results
        else:
            return {
                "active": False,
                "direction": None,
                "strength": 0,
                "signals": [_signal_summary(r) for r in results],
                "narrative": "Mixed signals — no clear direction",
                "key_level": None,
                "watch_for": None,
            }

    # Need strength >= 2 to fire (at least one strong signal or two weak ones)
    if strength < 2:
        return {
            "active": False,
            "direction": direction,
            "strength": strength,
            "signals": [_signal_summary(r) for r in results],
            "narrative": f"Weak {direction or 'neutral'} setup — not enough confluence",
            "key_level": None,
            "watch_for": None,
        }

    # Build narrative
    signal_names = [r["signal"] for r in aligned]
    narratives = [r["narrative"] for r in aligned]

    # Find key level — prioritise flip, then wall, then vacuum target
    key_level = None
    for r in aligned:
        if r.get("level") is not None:
            key_level = r["level"]
            break

    # What to watch for on chart
    if direction == "LONG":
        watch_for = "Watch for bullish price action — higher lows, breakout candle, volume spike"
    elif direction == "SHORT":
        watch_for = "Watch for bearish price action — lower highs, breakdown candle, volume spike"
    else:
        watch_for = "Watch for directional breakout — IV expanding, big candle either way"

    if key_level:
        watch_for += f" near {key_level:.0f}"

    return {
        "active": True,
        "direction": direction,
        "strength": strength,
        "signals": [_signal_summary(r) for r in results],
        "narrative": " | ".join(narratives),
        "key_level": key_level,
        "watch_for": watch_for,
    }


def _signal_summary(r):
    """Compact summary for storage/display."""
    return {
        "signal": r["signal"],
        "direction": r["direction"],
        "score": r["score"],
        "narrative": r["narrative"],
    }
