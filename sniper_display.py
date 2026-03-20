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
    "gamma_structure":   1.75,  # gamma sign + flip proximity + wall proximity
    "straddle_momentum": 1.5,   # straddle expanding = fuel for the move
    "spot_vs_walls":     1.75,  # near wall = setup, mid-range = noise
    "oi_velocity":       1.5,   # OI SURGE = real flow, not just noise
    "iv_premium":        0.5,   # REDUCED — overlaps with straddle & MPM
    "move_prob":         1.5,   # your existing MPM — already synthesised
    "structural_event":  1.5,   # vacuum / flip breakout / liq accel / squeeze
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
    Mid-range with positive gamma = pinned = low score.
    """
    total_range = call_wall - put_wall
    if total_range <= 0:
        return 0.0

    # Position in range: 0 = at put wall, 1 = at call wall
    position = (spot - put_wall) / total_range
    # Distance from nearest wall as fraction of range
    wall_frac = min(position, 1 - position)

    score = 0.0

    # Near wall (within 15% of range from either side)
    if wall_frac <= 0.15:
        score += 0.6
        # Extra if gamma negative (walls more likely to break)
        if gamma < 0:
            score += 0.2
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
    elif prob >= 50:
        score = 0.3

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

    # Liquidity acceleration
    if liq_accel and liq_accel.get("detected"):
        conv = liq_accel.get("conviction", "LOW")
        if conv == "HIGH":
            score = max(score, 1.0)
        else:
            score = max(score, 0.6)

    # Squeeze
    if squeeze:
        score = max(score, 0.8)

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────
# SETUP CLASSIFIER
# ─────────────────────────────────────────────────────────────

def _classify_setup(vacuum, wall_break_vac, flip_breakout, liq_accel,
                    squeeze, trap, velocity, gamma, momentum_data,
                    spot, call_wall, put_wall, trend):
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

    # Trend continuation
    if trend and trend.get("trending"):
        pts = trend.get("move_pts", 0)
        if pts >= 15:
            return "TREND CONTINUATION"

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
                       wall_break_vac, liq_accel, trend, velocity):
    """
    Returns (direction_str, direction_color).
    Structural events override bias.
    """
    direction = None

    # Structural events carry their own direction
    if flip_breakout and flip_breakout.get("detected"):
        direction = flip_breakout["direction"]
    elif liq_accel and liq_accel.get("detected") and liq_accel.get("conviction") == "HIGH":
        direction = liq_accel["direction"]
    elif vacuum and vacuum.get("status") == "CONFIRMED":
        direction = vacuum.get("direction")
    elif wall_break_vac and wall_break_vac.get("detected"):
        direction = wall_break_vac.get("direction")
    elif trend and trend.get("trending"):
        direction = trend["direction"]

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

    # Map to display
    if direction in ("UP", "BULLISH"):
        return "LONG", Fore.GREEN
    elif direction in ("DOWN", "BEARISH"):
        return "SHORT", Fore.RED
    else:
        return "NEUTRAL", Fore.YELLOW


# ─────────────────────────────────────────────────────────────
# ACTION DECISION
# ─────────────────────────────────────────────────────────────

def _decide_action(total_score, setup, confidence, trap, bias, days_to_expiry,
                   regime=None, active_trade=None, direction=None,
                   is_locked=False, lock_reason=None):
    """
    Returns (action_text, action_color, action_icon).
    """
    trap_conf = trap.get("confidence", 0) if trap else 0

    # ── Trade lock — block new entries while in a position ────
    # Unlocks only when: trade exits (target/SL/manual) OR structure resets
    if is_locked:
        reason_tag = f" ({lock_reason})" if lock_reason else ""
        return f"LOCKED{reason_tag}", Fore.CYAN, "🔒"

    # ── Fix 2: Chop killer — hard block on chop regimes ──────────────
    chop_regimes = ("CHOP", "GAMMA PINNING", "RANGE LOCK")
    if regime and any(r in regime.upper() for r in chop_regimes):
        if total_score < 7:
            return "CHOP — NO TRADE", Fore.RED, "🧱"

    # Hard blocks
    if total_score < 4:
        return "STAND DOWN", Fore.RED, "🚫"
    if setup == "DEVELOPING" and total_score < 7:
        return "NO EDGE — WAIT", Fore.RED, "🚫"
    if bias == "RANGE" and total_score < 7 and setup not in (
        "TRAP FADE", "FLIP BREAKOUT", "VACUUM DRIVE", "WALL BREAK",
        "LIQ ACCELERATION", "GAMMA SQUEEZE"
    ):
        return "RANGE — SIT OUT", Fore.YELLOW, "⏸ "

    # Trap warning (but don't block structural setups)
    if trap_conf >= 80 and setup not in (
        "TRAP FADE", "FLIP BREAKOUT", "VACUUM DRIVE",
        "LIQ ACCELERATION", "GAMMA SQUEEZE"
    ):
        return "TRAP RISK — SKIP", Fore.RED, "🪤"

    # ── Fix 5: Tighter thresholds ────────────────────────────────────
    if total_score >= 8.5:
        return "SEND IT", Fore.GREEN, "🎯"
    if total_score >= 7:
        return "TAKE TRADE", Fore.GREEN, "👉"
    if total_score >= 5.5:
        return "STALK — WAIT FOR TRIGGER", Fore.YELLOW, "👁 "

    return "WAIT", Fore.YELLOW, "⏸ "


# ─────────────────────────────────────────────────────────────
# CONFIDENCE LABEL
# ─────────────────────────────────────────────────────────────

def _conf_label(score):
    if score >= 8:
        return "VERY HIGH", Fore.GREEN
    if score >= 6:
        return "HIGH", Fore.GREEN
    if score >= 4:
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

    # ── Score each signal ──────────────────────────────────────
    scores = {
        "gamma_structure":   _score_gamma(gamma, flip_level, spot, call_wall, put_wall),
        "straddle_momentum": _score_straddle(momentum_data),
        "spot_vs_walls":     _score_spot_vs_walls(spot, call_wall, put_wall, gamma),
        "oi_velocity":       _score_oi_velocity(velocity, call_oi_speed, put_oi_speed),
        "iv_premium":        _score_iv(momentum_data, straddle, days_to_expiry),
        "move_prob":         _score_move_prob(move_prob),
        "structural_event":  _score_structural(vacuum, wall_break_vac, flip_breakout, liq_accel, squeeze),
    }

    # Weighted total (0-10)
    total = sum(scores[k] * W[k] for k in scores)

    # ── Classify ───────────────────────────────────────────────
    setup = _classify_setup(
        vacuum, wall_break_vac, flip_breakout, liq_accel,
        squeeze, trap, velocity, gamma, momentum_data,
        spot, call_wall, put_wall, trend,
    )
    direction, dir_color = _resolve_direction(
        bias, move_prob, flip_breakout, vacuum,
        wall_break_vac, liq_accel, trend, velocity,
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
        is_locked=is_locked, lock_reason=lock_reason,
    )

    # ── Score bar ──────────────────────────────────────────────
    filled = total_int
    empty = 10 - filled
    if total >= 7:
        bar_color = Fore.GREEN
    elif total >= 4:
        bar_color = Fore.YELLOW
    else:
        bar_color = Fore.RED

    score_check = ""
    if total >= 7:
        score_check = f" {Fore.GREEN}✅{Style.RESET_ALL}"
    elif total >= 5:
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
            "straddle_momentum": "Str",
            "spot_vs_walls":     "Spt",
            "oi_velocity":       "OI",
            "iv_premium":        "IV",
            "move_prob":         "MPM",
            "structural_event":  "Evt",
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