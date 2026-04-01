"""
sniper_mode.py — 20-25pt Nifty Scalp Scorer

Plugs into print_dashboard(). Reads your already-computed signals,
scores them on a 0-10 scale, classifies the setup, and prints a
single actionable box.

Usage — add to the end of print_dashboard(), before the final "=" line:

    from sniper_mode import sniper_display

    sniper_display(
        spot=spot, bias=bias, confidence=confidence,
        gamma=gamma, straddle=straddle, momentum_data=momentum_data,
        move_prob=move_prob, trap=trap, velocity=velocity,
        vacuum=vacuum, wall_break_vac=wall_break_vac,
        flip_breakout=flip_breakout, liq_accel=liq_accel,
        squeeze=squeeze, trend=trend,
        call_wall=call_wall, put_wall=put_wall,
        flip_level=flip_level, regime=regime,
        trade=trade, days_to_expiry=days_to_expiry,
        active_trade=state.active_trade,
        gamma_shift=gamma_shift,
        notify_fn=notify,
        debug=DEBUG_MODE,
    )

TODO
----
  1. THRESHOLD CALIBRATION
     Run for 1 week with current thresholds (7 = TAKE TRADE, 8.5 = SEND IT).
     Track how often each tier fires and win rate per tier.
     If TAKE TRADE fires 3-4x/day with good win rate → keep.
     If SEND IT never fires or you're missing obvious setups →
     drop SEND IT to 8.0 and TAKE TRADE to 6.5.
     The chop killer matters more than threshold tuning — focus there first.

  2. TRADE LOCK INTEGRATION WITH WAR ROOM
     The SniperLock class below tracks lock state internally.
     To fully integrate with your existing Meta+I / Meta+X hotkey flow:
       - On Meta+I (trade entry):  lock engages automatically via active_trade
       - On Meta+X (manual exit):  lock releases (state.active_trade = None)
       - On HTL EXIT verdict:      lock releases (war room already clears active_trade)
       - On gamma_shift change:    lock releases (structural reset)
     Currently the lock checks active_trade + gamma_shift. If you want
     target/SL-based unlock, pipe entry_price into sniper_display and
     compare against current option LTP.

  3. PER-SETUP WIN RATE TRACKING
     After 30+ trades, log which setup type (BREAKOUT, VACUUM DRIVE, etc.)
     produced the sniper TAKE TRADE / SEND IT action. Compute win rate per
     setup. Use this to add a setup_reliability multiplier to the score —
     setups that historically win get a bonus, losers get penalized.

  4. COOLDOWN AFTER LOSS
     Consider adding a 5-10 minute cooldown after a stopped-out trade
     before sniper can fire again. Prevents revenge entries on the same
     failed structure. The lock currently blocks during active trades but
     doesn't enforce post-exit cooldown.
"""

from colorama import Fore, Style
from config import (CHOP_KILLER_GAMMA_MIN, SNIPER_TAKE_TRADE,
                    SNIPER_SEND_IT, SNIPER_STALK,
                    GAMMA_MOMENTUM_LOOKBACK, GAMMA_MOMENTUM_CHOP_EASE)


# ─────────────────────────────────────────────────────────────
# TRADE LOCK — one trade at a time, unlock on exit or structure reset
# ─────────────────────────────────────────────────────────────

class SniperLock:
    """
    State machine for trade locking.

    Locked when:
      - active_trade is not None (you're in a position)

    Unlocked when:
      - Trade hits target or SL (active_trade goes None via HTL EXIT or Meta+X)
      - Structure resets: gamma_shift changes (new dealer regime = new setup)

    Usage:
      _lock = SniperLock()

      # In sniper_display(), before _decide_action():
      locked, lock_reason = _lock.evaluate(active_trade, gamma_shift)
    """

    def __init__(self):
        self._locked = False
        self._lock_gamma_shift = None    # gamma_shift at time of lock
        self._lock_reason = None

    def evaluate(self, active_trade, gamma_shift=None):
        """
        Returns (is_locked: bool, reason: str or None).

        Call every tick. Handles lock/unlock transitions.
        """
        # ── Unlock conditions ─────────────────────────────────
        if self._locked:
            # Trade exited (target, SL, manual, or HTL EXIT)
            if active_trade is None:
                self._locked = False
                self._lock_reason = None
                self._lock_gamma_shift = None
                return False, None

            # Structure reset: gamma regime shifted since entry
            if (gamma_shift is not None
                    and self._lock_gamma_shift is not None
                    and gamma_shift != self._lock_gamma_shift):
                self._locked = False
                self._lock_reason = None
                self._lock_gamma_shift = None
                return False, None

            # Still locked
            return True, self._lock_reason

        # ── Lock condition ────────────────────────────────────
        if active_trade is not None:
            self._locked = True
            self._lock_gamma_shift = gamma_shift
            self._lock_reason = (
                f"{active_trade.get('strike', '?')} "
                f"{active_trade.get('option_type', '?')} "
                f"@ {active_trade.get('entry_time', '?')}"
            )
            return True, self._lock_reason

        return False, None


# Module-level lock instance — persists across ticks
_sniper_lock = SniperLock()


# ─────────────────────────────────────────────────────────────
# TELEGRAM ALERT — anti-spam: one alert per setup, resets on action change
# ─────────────────────────────────────────────────────────────
_last_sniper_alert = None   # tracks "TAKE TRADE|BREAKOUT|LONG" to avoid repeat


def sniper_notify(notify_fn, action, message):
    """
    Send Telegram only if this is a NEW signal.
    Same action + setup + direction = don't re-send every tick.
    Resets when action changes (e.g. TAKE TRADE → STAND DOWN → TAKE TRADE fires again).
    """
    global _last_sniper_alert
    alert_key = action  # simplest: dedupe on action text

    if alert_key == _last_sniper_alert:
        return  # already sent this one

    _last_sniper_alert = alert_key
    try:
        notify_fn(message)
    except Exception as e:
        print(f"  ⚠ Sniper Telegram failed: {e}")

# ─────────────────────────────────────────────────────────────
# SCORING WEIGHTS — tune these to match your edge
# ─────────────────────────────────────────────────────────────
# Each signal contributes 0-N points to the total.
# Max possible = 10. You need 6+ to fire.

W = {
    "gamma_structure":   1.25,  # gamma sign + flip proximity + wall proximity
    "gamma_momentum":    1.50,  # gamma declining OR sustained negative = trending
    "straddle_momentum": 1.25,  # straddle expanding = fuel for the move
    "spot_vs_walls":     1.25,  # near/past wall = setup, mid-range = noise
    "oi_velocity":       0.50,  # OI SURGE = real flow — rare, low weight
    "iv_premium":        0.50,  # REDUCED — overlaps with straddle & MPM
    "move_prob":         0.75,  # your existing MPM — already synthesised
    "structural_event":  1.25,  # vacuum / flip breakout / liq accel / squeeze
    "trend":             1.75,  # price-action trend — fallback when OI signals lag
}
# Sum = 10.0


# ─────────────────────────────────────────────────────────────
# SCORING FUNCTIONS — each returns 0.0 to 1.0 (scaled by weight)
# ─────────────────────────────────────────────────────────────

def _score_gamma(gamma, flip_level, spot, call_wall, put_wall):
    """
    Negative gamma + near flip = explosive.
    Positive gamma + far from flip = pinned = bad for scalp.
    """
    score = 0.0

    # Gamma sign: negative = dealer short gamma = amplified moves
    if gamma < 0:
        score += 0.5
    elif gamma < 500_000:
        score += 0.2

    # Flip proximity: close to flip = regime change zone
    if flip_level:
        dist = abs(spot - flip_level)
        if dist <= 15:
            score += 0.3     # right at flip — maximum potential
        elif dist <= 40:
            score += 0.15

    # Wall proximity: near a wall = catalyst
    to_upper = abs(call_wall - spot)
    to_lower = abs(spot - put_wall)
    nearest = min(to_upper, to_lower)
    if nearest <= 25:
        score += 0.2
    elif nearest <= 50:
        score += 0.1

    return min(1.0, score)


def _score_straddle(momentum_data):
    """
    Expanding straddle = the move has premium backing.
    Compressing = decay, bad for directional scalps.
    """
    if not momentum_data:
        return 0.0

    m5 = momentum_data.get("momentum_5m") or 0
    status = momentum_data.get("status", "")

    score = 0.0

    # 5-min straddle momentum
    abs_m5 = abs(m5)
    if abs_m5 >= 3.0:
        score += 0.5      # strong expansion
    elif abs_m5 >= 1.5:
        score += 0.3
    elif abs_m5 >= 0.5:
        score += 0.1

    # Status tag from your straddle tracker
    if "EXPANDING" in status.upper():
        score += 0.3
    elif "FAST" in status.upper():
        score += 0.5

    # 15-min confirmation
    m15 = momentum_data.get("momentum_15m") or 0
    if abs(m15) >= 2.0:
        score += 0.2

    return min(1.0, score)


def _score_spot_vs_walls(spot, call_wall, put_wall, gamma):
    """
    Near a wall with negative gamma = breakout potential.
    Near a wall with positive gamma = wall defense / bounce potential.
    Mid-range with positive gamma = pinned = low score.
    Past a wall = breakout underway = high score.
    """
    total_range = call_wall - put_wall
    if total_range <= 0:
        return 0.0

    score = 0.0

    # Spot has broken PAST a wall — strong breakout signal
    if spot > call_wall:
        score += 0.7
        if gamma < 0:
            score += 0.3  # dealers short gamma = accelerating
        return min(1.0, score)
    if spot < put_wall:
        score += 0.7
        if gamma < 0:
            score += 0.3
        return min(1.0, score)

    # Position in range: 0 = at put wall, 1 = at call wall
    position = (spot - put_wall) / total_range
    # Distance from nearest wall as fraction of range
    wall_frac = min(position, 1 - position)

    # Near wall (within 15% of range from either side)
    if wall_frac <= 0.15:
        if gamma < 0:
            # Breakout scenario — wall likely to break
            score += 0.6
            score += 0.2  # gamma amplifies
        else:
            # Defense scenario — wall likely to hold, bounce opportunity
            score += 0.5
            score += 0.15  # positive gamma = dealers defend the level
    elif wall_frac <= 0.25:
        score += 0.3
    else:
        # Mid-range — only good if gamma negative (not pinned)
        if gamma < 0:
            score += 0.15

    # Bonus: extreme positioning (within 10pts of wall)
    nearest = min(abs(call_wall - spot), abs(spot - put_wall))
    if nearest <= 10:
        score += 0.2

    return min(1.0, score)


def _score_oi_velocity(velocity, call_oi_speed, put_oi_speed):
    """
    SURGE = institutional flow. CONFLICTED = noise.
    """
    if not velocity:
        return 0.0

    score = 0.0
    v = velocity.upper()

    if "SURGE" in v and "CONFLICTED" not in v:
        score += 0.7
        # One-sided flow is stronger
        if call_oi_speed is not None and put_oi_speed is not None:
            if abs(call_oi_speed) > abs(put_oi_speed) * 3:
                score += 0.3
            elif abs(put_oi_speed) > abs(call_oi_speed) * 3:
                score += 0.3
    elif "FAST" in v:
        score += 0.3

    return min(1.0, score)


def _score_iv(momentum_data, straddle, days_to_expiry):
    """
    IV expanding = premium supports the move.
    ANTI-DOUBLE-COUNT: if straddle momentum is already strong (>= 2%),
    IV score is capped — that expansion is already captured by _score_straddle.
    Only adds value when IV is moving but straddle hasn't shown it yet
    (e.g. skew shift, term structure move).
    """
    score = 0.0
    straddle_already_strong = False

    if momentum_data:
        m5 = abs(momentum_data.get("momentum_5m") or 0)

        # If straddle momentum is already >= 2%, cap IV contribution
        if m5 >= 2.0:
            straddle_already_strong = True

        if m5 >= 2.0:
            score += 0.3    # reduced from 0.5 — straddle scorer already has this
        elif m5 >= 1.0:
            score += 0.25

        status = (momentum_data.get("status") or "").upper()
        if "EXPANDING" in status or "FAST" in status:
            score += 0.2

    # Expiry penalty: 0 DTE with low straddle = death by theta
    if days_to_expiry == 0 and straddle < 80:
        score *= 0.5

    # Hard cap when straddle momentum already strong
    if straddle_already_strong:
        score = min(score, 0.5)

    return min(1.0, score)


def _score_move_prob(move_prob):
    """
    Your MPM already synthesises multiple signals. Trust it.
    """
    if not move_prob:
        return 0.0

    prob = move_prob.get("probability", 0)
    conv = move_prob.get("conviction", "LOW")
    conflicted = move_prob.get("conflicted", False)

    if conflicted:
        return 0.1  # conflicted = unreliable

    score = 0.0
    if prob >= 80:
        score = 1.0
    elif prob >= 70:
        score = 0.7
    elif prob >= 60:
        score = 0.5
    elif prob >= 40:
        score = 0.3
    elif prob >= 30:
        score = 0.15

    # Conviction bonus
    if conv == "VERY HIGH":
        score = min(1.0, score + 0.2)
    elif conv == "HIGH":
        score = min(1.0, score + 0.1)

    return score


def _score_structural(vacuum, wall_break_vac, flip_breakout, liq_accel, squeeze):
    """
    Structural events are the highest-conviction setups.
    Any one of these firing = strong signal.
    """
    score = 0.0

    # Vacuum confirmed
    if vacuum and vacuum.get("status") == "CONFIRMED":
        sc = vacuum.get("score", 0)
        if sc >= 70:
            score = max(score, 1.0)
        elif sc >= 50:
            score = max(score, 0.7)

    # Wall break
    if wall_break_vac and wall_break_vac.get("detected"):
        score = max(score, 0.8)

    # Gamma flip breakout
    if flip_breakout and flip_breakout.get("detected"):
        score = max(score, 0.9)

    # Liquidity acceleration — score by conviction AND raw score
    if liq_accel and liq_accel.get("detected"):
        conv = liq_accel.get("conviction", "LOW")
        accel_score = liq_accel.get("score", 0)
        if conv == "HIGH":
            score = max(score, 1.0)
        elif accel_score >= 60:
            score = max(score, 0.8)
        else:
            score = max(score, 0.6)

    # Squeeze
    if squeeze:
        score = max(score, 0.8)

    return min(1.0, score)


def _score_trend(trend, spot_history=None, spot=None,
                 session_high=None, session_low=None):
    """
    Pure price-action trend score — fallback when OI-derived signals lag.
    Uses detect_persistent_trend() output + live ROC from spot_history.

    Key insight: roc1 > 10pts (last candle moved 10+ pts in trade direction)
    separates winners (75%) from losers with zero missed winners in backtest.
    """
    if not trend or not trend.get("trending"):
        return 0.0

    # Pullback = retrace within a larger move — don't score the short window
    if trend.get("pullback"):
        return 0.0

    pts = trend.get("move_pts", 0)
    duration = trend.get("duration_minutes", 0)
    trend_dir = trend.get("direction")  # "UP" or "DOWN"

    # (no score-level exhaustion guard — direction resolver handles extremes)

    # ── Live momentum check (roc1) ──────────────────────────
    # If spot moved < 3pts in trend direction in the last candle,
    # the move is stalling — suppress the score.
    if spot_history and len(spot_history) >= 2 and trend_dir:
        roc1 = spot_history[-1] - spot_history[-2]
        if trend_dir == "DOWN":
            roc1 = -roc1  # make positive when moving in trend direction
        if roc1 < 3:
            return 0.0  # stale trend — don't score

    score = 0.0

    # Move magnitude
    if pts >= 80:
        score += 0.5
    elif pts >= 50:
        score += 0.35
    elif pts >= 30:
        score += 0.2

    # Duration — sustained trend is more reliable
    if duration >= 20:
        score += 0.3
    elif duration >= 10:
        score += 0.2
    elif duration >= 5:
        score += 0.1

    # Speed (pts per minute) — fast moves = momentum
    if duration > 0:
        speed = pts / duration
        if speed >= 5:
            score += 0.3     # 5+ pts/min = very fast
        elif speed >= 2:
            score += 0.2
        elif speed >= 1:
            score += 0.1

    return min(1.0, score)


def compute_gamma_momentum(gamma_history):
    """
    Compute gamma rate-of-change from history.
    Returns dict: {
        "declining": bool,      — gamma is dropping (toward flip)
        "drop_pct": float,      — how much gamma dropped (0-1)
        "flipping": bool,       — gamma crossed zero or about to
        "direction": "DOWN"/"UP"/"FLAT"
    }
    """
    if not gamma_history or len(gamma_history) < 3:
        return {"declining": False, "drop_pct": 0.0, "flipping": False, "direction": "FLAT"}

    lookback = min(len(gamma_history), GAMMA_MOMENTUM_LOOKBACK)
    recent = list(gamma_history)[-lookback:]
    first, last = recent[0], recent[-1]

    # Drop percentage (only meaningful when gamma was positive)
    if first > 0 and first > 1e10:
        drop_pct = max(0.0, (first - last) / first)
    elif first < 0 and first < -1e10:
        # Gamma getting more negative = strengthening downside
        drop_pct = max(0.0, (first - last) / abs(first))
    else:
        drop_pct = 0.0

    # Detect sign change or near-zero crossing
    flipping = (first > 0 and last <= 0) or (first >= 0 and last < 0)
    near_flip = first > 0 and last > 0 and last < first * 0.3  # dropped to <30% of original

    # Direction of gamma movement
    if last < first:
        direction = "DOWN"  # gamma declining (moving toward negative)
    elif last > first:
        direction = "UP"    # gamma increasing (moving toward positive)
    else:
        direction = "FLAT"

    declining = direction == "DOWN" and drop_pct >= 0.15

    # Sustained negative gamma — dealers short, trending environment
    sustained_negative = all(g < 0 for g in recent)
    # How negative is it (magnitude relative to typical gamma)
    neg_strength = 0.0
    if last < 0:
        neg_strength = min(1.0, abs(last) / 5e13)  # normalize: 5e13 = strong negative

    return {
        "declining": declining,
        "drop_pct": round(drop_pct, 3),
        "flipping": flipping or near_flip,
        "direction": direction,
        "sustained_negative": sustained_negative,
        "neg_strength": round(neg_strength, 3),
    }


def _score_gamma_momentum(gamma_mom):
    """
    Score gamma rate-of-change AND sustained negative gamma.
    Declining gamma = imminent regime change.
    Sustained negative gamma = dealers short = trending environment.
    """
    if not gamma_mom:
        return 0.0

    score = 0.0

    # --- Sustained negative gamma: dealers short, market can trend ---
    if gamma_mom.get("sustained_negative"):
        neg_str = gamma_mom.get("neg_strength", 0)
        if neg_str >= 0.8:
            score += 0.5   # strongly negative — full trending environment
        elif neg_str >= 0.4:
            score += 0.35  # moderately negative
        else:
            score += 0.2   # mildly negative but sustained
    elif gamma_mom.get("neg_strength", 0) > 0:
        # Currently negative but not sustained (recently flipped)
        score += 0.15

    # --- Declining gamma: rate-of-change signals ---
    if gamma_mom.get("declining"):
        drop_pct = gamma_mom.get("drop_pct", 0)

        # Gamma flipping (crossed zero or about to)
        if gamma_mom.get("flipping"):
            score += 0.3

        # Drop magnitude
        if drop_pct >= 0.7:
            score += 0.3     # gamma collapsed
        elif drop_pct >= 0.5:
            score += 0.2
        elif drop_pct >= 0.3:
            score += 0.1

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# SETUP CLASSIFIER
# ─────────────────────────────────────────────────────────────

def _classify_setup(vacuum, wall_break_vac, flip_breakout, liq_accel,
                    squeeze, trap, velocity, gamma, momentum_data,
                    spot, call_wall, put_wall, trend, gamma_mom=None):
    """
    Returns a human-readable setup name.
    Priority: structural events > flow > position.
    """
    # Structural events (highest conviction)
    if flip_breakout and flip_breakout.get("detected"):
        return "FLIP BREAKOUT"

    if liq_accel and liq_accel.get("detected") and liq_accel.get("conviction") == "HIGH":
        return "LIQ ACCELERATION"

    if vacuum and vacuum.get("status") == "CONFIRMED" and vacuum.get("score", 0) >= 60:
        return "VACUUM DRIVE"

    if wall_break_vac and wall_break_vac.get("detected"):
        return "WALL BREAK"

    if squeeze:
        return "GAMMA SQUEEZE"

    # Trap fade (counter-trend scalp)
    if trap and trap.get("confidence", 0) >= 70 and gamma >= 0:
        return "TRAP FADE"

    # Trend continuation — but NOT pullbacks
    if trend and trend.get("trending") and not trend.get("pullback"):
        pts = trend.get("move_pts", 0)
        if pts >= 15:
            return "TREND CONTINUATION"

    # Gamma regime shift — gamma collapsing or flipping
    if gamma_mom and (gamma_mom.get("flipping") or gamma_mom.get("drop_pct", 0) >= 0.5):
        return "GAMMA SHIFT"

    # Flow-driven
    vel = (velocity or "").upper()
    if "SURGE" in vel and "CONFLICTED" not in vel:
        return "OI SURGE"

    # Near-wall setups
    nearest = min(abs(call_wall - spot), abs(spot - put_wall))
    if nearest <= 20:
        if gamma < 0:
            return "BREAKOUT"
        else:
            return "WALL BOUNCE"

    # Straddle expansion with direction
    if momentum_data and abs(momentum_data.get("momentum_5m") or 0) >= 3.0:
        return "IV EXPANSION"

    return "DEVELOPING"


# ─────────────────────────────────────────────────────────────
# DIRECTION — which side of the scalp
# ─────────────────────────────────────────────────────────────

def _resolve_direction(bias, move_prob, flip_breakout, vacuum,
                       wall_break_vac, liq_accel, trend, velocity,
                       spot=None, session_high=None, session_low=None):
    """
    Returns (direction_str, direction_color).
    Structural events override bias.
    Trend is demoted when it's a pullback within a larger move.
    Exhaustion guard suppresses direction at session extremes — but only
    for weak sources (bias, move_prob). Structural events and strong trends
    (100+ pts) are exempt: on a real trend day every new low is 0% by
    definition, and suppressing direction there costs us the entire move.
    """
    direction = None
    structural = False   # True = direction from a high-conviction source

    # Structural events carry their own direction (highest priority)
    if flip_breakout and flip_breakout.get("detected"):
        direction = flip_breakout["direction"]
        structural = True
    if not direction and liq_accel and liq_accel.get("detected") and liq_accel.get("conviction") in ("HIGH", "MODERATE"):
        direction = liq_accel["direction"]
        if liq_accel.get("conviction") == "HIGH":
            structural = True
    if not direction and vacuum and vacuum.get("status") == "CONFIRMED":
        direction = vacuum.get("direction")
        structural = True
    if not direction and wall_break_vac and wall_break_vac.get("detected"):
        direction = wall_break_vac.get("direction")
        structural = True

    # Trend — only if NOT a pullback
    if not direction and trend and trend.get("trending"):
        if trend.get("pullback"):
            broader = trend.get("broader_direction")
            if broader:
                direction = broader
        else:
            direction = trend["direction"]
        # Strong trend (100+ pts) = structural-grade conviction
        if trend.get("move_pts", 0) >= 100:
            structural = True

    # Structural upgrade: MODERATE liq_accel + confirming trend = structural
    if (not structural and liq_accel and liq_accel.get("detected")
            and trend and trend.get("trending")
            and trend.get("move_pts", 0) >= 75):
        structural = True

    # Fall back to move_prob direction
    if not direction and move_prob:
        mp_dir = move_prob.get("direction", "")
        if mp_dir in ("BULLISH", "UP"):
            direction = "UP"
        elif mp_dir in ("BEARISH", "DOWN"):
            direction = "DOWN"

    # Fall back to bias
    if not direction:
        if bias == "BULLISH":
            direction = "UP"
        elif bias == "BEARISH":
            direction = "DOWN"

    # Exhaustion guard — suppress direction at session extremes
    # Skip for structural sources: on a trend day, new extremes are the trade,
    # not exhaustion. Only gate weak sources (bias, move_prob, weak trend).
    if (direction and not structural
            and spot is not None
            and session_high is not None and session_low is not None):
        session_range = session_high - session_low
        if session_range >= 50:
            position = (spot - session_low) / session_range
            if direction in ("DOWN", "DOWNSIDE", "BEARISH") and position < 0.15:
                direction = None
            elif direction in ("UP", "UPSIDE", "BULLISH") and position > 0.85:
                direction = None

    # Map to display
    if direction in ("UP", "UPSIDE", "BULLISH"):
        return "LONG", Fore.GREEN
    elif direction in ("DOWN", "DOWNSIDE", "BEARISH"):
        return "SHORT", Fore.RED
    else:
        return "NEUTRAL", Fore.YELLOW


# ─────────────────────────────────────────────────────────────
# ACTION DECISION
# ─────────────────────────────────────────────────────────────

def _decide_action(total_score, setup, confidence, trap, bias, days_to_expiry,
                   regime=None, active_trade=None, direction=None,
                   is_locked=False, lock_reason=None, gamma=0,
                   gamma_mom=None):
    """
    Returns (action_text, action_color, action_icon).
    """
    trap_conf = trap.get("confidence", 0) if trap else 0

    # ── Trade lock — block new entries while in a position ────
    # Unlocks only when: trade exits (target/SL/manual) OR structure resets
    if is_locked:
        reason_tag = f" ({lock_reason})" if lock_reason else ""
        return f"LOCKED{reason_tag}", Fore.CYAN, "🔒"

    # ── Chop killer — hard block on chop regimes ─────────────────────
    # Eased when gamma is declining fast (regime about to change)
    gamma_declining = (gamma_mom and gamma_mom.get("declining")
                       and gamma_mom.get("drop_pct", 0) >= GAMMA_MOMENTUM_CHOP_EASE)
    chop_regimes = ("CHOP", "RANGE LOCK")
    gamma_pinning_strong = "GAMMA PINNING" in (regime or "").upper() and gamma > CHOP_KILLER_GAMMA_MIN
    if regime and (any(r in regime.upper() for r in chop_regimes) or gamma_pinning_strong):
        if not gamma_declining and total_score < SNIPER_TAKE_TRADE:
            return "CHOP — NO TRADE", Fore.RED, "🧱"

    # Hard blocks
    if total_score < 3:
        return "STAND DOWN", Fore.RED, "🚫"
    if setup == "DEVELOPING" and total_score < SNIPER_TAKE_TRADE:
        return "NO EDGE — WAIT", Fore.RED, "🚫"

    # Trap warning (but don't block structural setups, trends, or gamma shifts)
    if trap_conf >= 80 and setup not in (
        "TRAP FADE", "FLIP BREAKOUT", "VACUUM DRIVE",
        "LIQ ACCELERATION", "GAMMA SQUEEZE", "TREND CONTINUATION",
        "GAMMA SHIFT"
    ):
        return "TRAP RISK — SKIP", Fore.RED, "🪤"

    # ── Thresholds ────────────────────────────────────────────────────
    # Direction must be resolved to issue a trade signal — can't size without it
    if direction == "NEUTRAL":
        if total_score >= SNIPER_TAKE_TRADE:
            return "STALK — NO DIRECTION", Fore.YELLOW, "👁 "
        if total_score >= SNIPER_STALK:
            return "STALK — WAIT FOR TRIGGER", Fore.YELLOW, "👁 "
        return "WAIT", Fore.YELLOW, "⏸ "
    if total_score >= SNIPER_SEND_IT:
        return "SEND IT", Fore.GREEN, "🎯"
    if total_score >= SNIPER_TAKE_TRADE:
        return "TAKE TRADE", Fore.GREEN, "👉"
    if total_score >= SNIPER_STALK:
        return "STALK — WAIT FOR TRIGGER", Fore.YELLOW, "👁 "

    return "WAIT", Fore.YELLOW, "⏸ "


# ─────────────────────────────────────────────────────────────
# CONFIDENCE LABEL
# ─────────────────────────────────────────────────────────────

def _conf_label(score):
    if score >= SNIPER_SEND_IT:
        return "VERY HIGH", Fore.GREEN
    if score >= SNIPER_TAKE_TRADE:
        return "HIGH", Fore.GREEN
    if score >= SNIPER_STALK:
        return "MEDIUM", Fore.YELLOW
    return "LOW", Fore.RED


# ─────────────────────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────────────────────

def sniper_display(
    spot, bias, confidence, gamma, straddle,
    momentum_data, move_prob, trap, velocity,
    vacuum, wall_break_vac, flip_breakout, liq_accel,
    squeeze, trend,
    call_wall, put_wall, flip_level, regime,
    trade, days_to_expiry,
    call_oi_speed=None, put_oi_speed=None,
    active_trade=None,
    gamma_shift=None,
    notify_fn=None,
    debug=False,
    gamma_history=None,
    spot_history=None,
):
    """
    Call this at the end of print_dashboard().
    Prints the sniper box with score, setup, bias, confidence, and action.

    Pass active_trade=state.active_trade for trade lock behavior.
    Pass gamma_shift=gamma_shift for structural reset unlock.
    Pass notify_fn=notify for Telegram alerts on high-conviction signals.
    """

    # ── Evaluate trade lock ────────────────────────────────────
    is_locked, lock_reason = _sniper_lock.evaluate(active_trade, gamma_shift)

    # ── Compute gamma momentum ─────────────────────────────────
    gamma_mom = compute_gamma_momentum(gamma_history)

    from state import state as _st

    # ── Score each signal ──────────────────────────────────────
    scores = {
        "gamma_structure":   _score_gamma(gamma, flip_level, spot, call_wall, put_wall),
        "gamma_momentum":    _score_gamma_momentum(gamma_mom),
        "straddle_momentum": _score_straddle(momentum_data),
        "spot_vs_walls":     _score_spot_vs_walls(spot, call_wall, put_wall, gamma),
        "oi_velocity":       _score_oi_velocity(velocity, call_oi_speed, put_oi_speed),
        "iv_premium":        _score_iv(momentum_data, straddle, days_to_expiry),
        "move_prob":         _score_move_prob(move_prob),
        "structural_event":  _score_structural(vacuum, wall_break_vac, flip_breakout, liq_accel, squeeze),
        "trend":             _score_trend(trend, spot_history=spot_history,
                                         spot=spot, session_high=_st.session_high,
                                         session_low=_st.session_low),
    }

    # Weighted total (0-10)
    total = sum(scores[k] * W[k] for k in scores)

    # ── Classify ───────────────────────────────────────────────
    setup = _classify_setup(
        vacuum, wall_break_vac, flip_breakout, liq_accel,
        squeeze, trap, velocity, gamma, momentum_data,
        spot, call_wall, put_wall, trend, gamma_mom=gamma_mom,
    )
    direction, dir_color = _resolve_direction(
        bias, move_prob, flip_breakout, vacuum,
        wall_break_vac, liq_accel, trend, velocity,
        spot=spot, session_high=_st.session_high, session_low=_st.session_low,
    )

    # ── Fix 4: Direction conflict penalty ──────────────────────
    # Structural event says GO but bias says RANGE = less conviction
    if bias == "RANGE" and direction != "NEUTRAL":
        total -= 0.5
    # Structural direction opposes bias direction = red flag
    if (bias == "BULLISH" and direction == "SHORT") or \
       (bias == "BEARISH" and direction == "LONG"):
        total -= 1.0

    total = round(max(0.0, min(10.0, total)), 1)
    total_int = int(round(total))

    conf_text, conf_color = _conf_label(total)
    action_text, action_color, action_icon = _decide_action(
        total, setup, confidence, trap, bias, days_to_expiry,
        regime=regime, active_trade=active_trade, direction=direction,
        is_locked=is_locked, lock_reason=lock_reason, gamma=gamma,
        gamma_mom=gamma_mom,
    )

    # ── Score bar ──────────────────────────────────────────────
    filled = total_int
    empty = 10 - filled
    if total >= SNIPER_TAKE_TRADE:
        bar_color = Fore.GREEN
    elif total >= SNIPER_STALK:
        bar_color = Fore.YELLOW
    else:
        bar_color = Fore.RED

    score_check = ""
    if total >= SNIPER_TAKE_TRADE:
        score_check = f" {Fore.GREEN}✅{Style.RESET_ALL}"
    elif total >= SNIPER_STALK:
        score_check = f" {Fore.YELLOW}⚠️{Style.RESET_ALL}"
    else:
        score_check = f" {Fore.RED}❌{Style.RESET_ALL}"

    bar = bar_color + "█" * filled + Fore.WHITE + "░" * empty + Style.RESET_ALL

    # ── Print — compact, matches war room density ──────────────
    print(Fore.CYAN + f"{'─' * 63}" + Style.RESET_ALL)
    print(f"🎯 SNIPER  [{bar}] {total_int}/10{score_check}"
          f"  {Fore.WHITE}{Style.BRIGHT}{setup}{Style.RESET_ALL}"
          f"  {dir_color}{direction}{Style.RESET_ALL}"
          f"  {conf_color}{conf_text}{Style.RESET_ALL}")
    print(f"{action_icon}  {action_color}{Style.BRIGHT}{action_text}{Style.RESET_ALL}")
    print(Fore.CYAN + f"{'─' * 63}" + Style.RESET_ALL)

    # Reset alert key when action changes so next TAKE TRADE / SEND IT fires fresh
    # Telegram is now sent by the war room after computing trade details
    if action_text not in ("TAKE TRADE", "SEND IT"):
        global _last_sniper_alert
        _last_sniper_alert = None

    # ── Signal breakdown — debug only ────────────────────────────
    if debug:
        parts = []
        labels = {
            "gamma_structure":   "Γ",
            "gamma_momentum":    "ΓΔ",
            "straddle_momentum": "Str",
            "spot_vs_walls":     "Spt",
            "oi_velocity":       "OI",
            "iv_premium":        "IV",
            "move_prob":         "MPM",
            "structural_event":  "Evt",
            "trend":             "Trd",
        }
        for key in scores:
            raw = scores[key]
            weighted = raw * W[key]
            if weighted >= W[key] * 0.7:
                c = Fore.GREEN
            elif weighted >= W[key] * 0.3:
                c = Fore.YELLOW
            else:
                c = Fore.RED
            parts.append(f"{c}{labels[key]}:{weighted:.1f}{Style.RESET_ALL}")

        print(f"  {' │ '.join(parts)}")

    return {
        "score": total,
        "setup": setup,
        "direction": direction,
        "confidence": conf_text,
        "action": action_text,
        "signal_scores": scores,
    }