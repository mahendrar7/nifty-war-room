"""
CoilSniper — SENSEX entry system based on pre-move coil detection.

Replaces the traditional sniper for SENSEX. Fires when the coil_score_v2
crosses SCORE_THRESHOLD, indicating straddle compression + directional
consensus building before a significant spot move.

Returns the same dict contract as NiftySniper/sniper_display so the caller
in options_war_room.py needs zero instrument-specific branching.
"""

from collections import deque
from datetime import datetime, timedelta

from colorama import Fore, Style

LOOKBACK_MIN    = 60
SCORE_THRESHOLD = 4.5
COOLDOWN_MIN    = 15


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class CoilSniper:
    """SENSEX sniper: fires on coil_score_v2 >= 4.5 with 15-min cooldown."""

    def __init__(self):
        self.straddle_history = deque(maxlen=LOOKBACK_MIN)   # float
        self.pcr_history      = deque(maxlen=LOOKBACK_MIN)   # float
        self.mpdir_history    = deque(maxlen=LOOKBACK_MIN)   # str "UPSIDE"|"DOWNSIDE"|""
        self.smom15_history   = deque(maxlen=LOOKBACK_MIN)   # float|None
        self.velocity_history = deque(maxlen=LOOKBACK_MIN)   # str
        self.liq_detected_history = deque(maxlen=LOOKBACK_MIN)  # bool
        self.liq_dir_history  = deque(maxlen=LOOKBACK_MIN)   # str
        self.squeeze_history  = deque(maxlen=LOOKBACK_MIN)   # str
        self.last_alert_time  = None

    # ── public interface ──────────────────────────────────────────────────────

    # SHORT signals are suppressed when day_move is 0–200 pts (11–23% win rate historically)
    SHORT_GATE_DAY_MOVE_LO = 0
    SHORT_GATE_DAY_MOVE_HI = 200

    def compute(self, straddle, momentum_data, move_prob,
                call_oi_speed, put_oi_speed, liq_accel, squeeze,
                pcr_val, velocity, now=None, day_open_spot=None, **kwargs):
        """
        Called every tick.  Returns sniper-compatible dict:
          action      : "TAKE TRADE" | "STALK — WAIT FOR TRIGGER" | ""
          direction   : "LONG" | "SHORT" | "NEUTRAL"
          score       : coil_score_v2  (0–10)
          confidence  : "HIGH" | "MEDIUM" | "LOW"
          setup       : short label string
          coil_score  : alias for score (for logging)
          breakdown   : dict of per-component scores (for debug)
        """
        if now is None:
            now = datetime.now()

        # 1. Update rolling histories
        self.straddle_history.append(_safe_float(straddle))
        self.pcr_history.append(_safe_float(pcr_val, default=1.0))

        mpdir = ""
        if move_prob:
            mpdir = move_prob.get("direction", "")
        if mpdir not in ("UPSIDE", "DOWNSIDE"):
            mpdir = ""
        self.mpdir_history.append(mpdir)

        mom15 = None
        if momentum_data:
            mom15 = momentum_data.get("momentum_15m")
        self.smom15_history.append(mom15)

        vel_str = velocity or ""
        self.velocity_history.append(vel_str)

        liq_det = bool(liq_accel and liq_accel.get("detected"))
        liq_dir = (liq_accel or {}).get("direction") or ""
        self.liq_detected_history.append(liq_det)
        self.liq_dir_history.append(liq_dir)

        sq_str = squeeze or ""
        self.squeeze_history.append(sq_str)

        # 2. Determine coil direction from current move_prob_dir
        if mpdir == "UPSIDE":
            move_dir = "UP"
        elif mpdir == "DOWNSIDE":
            move_dir = "DOWN"
        else:
            score, breakdown = 0.0, {}
            return self._result("", "NEUTRAL", 0.0, "LOW", "COIL", breakdown)

        # 3. Compute features over the full lookback window
        score, breakdown = self._score(move_dir)

        # 4. Determine action
        in_cooldown = (
            self.last_alert_time is not None and
            now - self.last_alert_time < timedelta(minutes=COOLDOWN_MIN)
        )

        direction  = "LONG" if move_dir == "UP" else "SHORT"

        # Day-move gate: SHORT signals on up-trending days (0–200 pts from open) have
        # only 11–23% win rate historically — suppress before cooldown is consumed.
        day_move_gate_blocked = False
        if direction == "SHORT" and day_open_spot is not None:
            spot_now = _safe_float(kwargs.get("spot", 0))
            day_move = spot_now - day_open_spot
            if self.SHORT_GATE_DAY_MOVE_LO <= day_move < self.SHORT_GATE_DAY_MOVE_HI:
                day_move_gate_blocked = True

        if day_move_gate_blocked:
            action = ""
        elif score >= SCORE_THRESHOLD and not in_cooldown:
            action    = "TAKE TRADE"
            self.last_alert_time = now
        elif score >= (SCORE_THRESHOLD - 1.0):
            action = "STALK — WAIT FOR TRIGGER"
        else:
            action = ""

        confidence = "HIGH" if score >= 6.5 else ("MEDIUM" if score >= SCORE_THRESHOLD else "LOW")
        setup      = self._setup_label(breakdown)
        if day_move_gate_blocked:
            setup += "[gate:day↑]"

        self._print_box(score, direction, confidence, action, breakdown, in_cooldown)

        return self._result(action, direction, score, confidence, setup, breakdown)

    # ── internals ─────────────────────────────────────────────────────────────

    def _score(self, move_dir):
        straddles = list(self.straddle_history)
        pcrs      = list(self.pcr_history)
        mpdirs    = list(self.mpdir_history)
        smom15s   = list(self.smom15_history)
        vels      = list(self.velocity_history)
        liq_dets  = list(self.liq_detected_history)
        liq_dirs  = list(self.liq_dir_history)
        squeezes  = list(self.squeeze_history)

        n = len(straddles)
        if n < 2:
            return 0.0, {}

        # Straddle compression (0–4 pts) — non-linear step-wise
        roll_max = max(straddles) if straddles else 1.0
        compression = (min(straddles) / roll_max) if roll_max else 1.0
        if   compression < 0.80: s_pts = 4.0
        elif compression < 0.85: s_pts = 3.0
        elif compression < 0.90: s_pts = 2.0
        elif compression < 0.93: s_pts = 0.5
        else:                    s_pts = 0.0

        # move_prob_dir streak (0–2.5 pts)
        target = "UPSIDE" if move_dir == "UP" else "DOWNSIDE"
        streak = 0
        for d in reversed(mpdirs):
            if d == target:
                streak += 1
            else:
                break
        streak_pts = min(streak / 8.0, 1.0) * 2.5

        # PCR crossing 1.0 (0–2 pts)
        mid       = n // 2
        pcr_first = pcrs[:mid] if mid else pcrs
        pcr_second= pcrs[mid:] if mid else pcrs
        pcr_end   = pcrs[-1]
        if move_dir == "UP":
            pcr_crossed = (any(p <= 1.0 for p in pcr_first) and
                           any(p >  1.0 for p in pcr_second))
            pcr_aligned = pcr_end > 1.0
        else:
            pcr_crossed = (any(p >= 1.0 for p in pcr_first) and
                           any(p <  1.0 for p in pcr_second))
            pcr_aligned = pcr_end < 1.0
        pcr_pts = 2.0 if pcr_crossed else (0.5 if pcr_aligned else 0.0)

        # OI covering (0–1.5 pts)
        covering_kw = "PUT COVERING" if move_dir == "UP" else "CALL COVERING"
        oi_covering = any(covering_kw in v.upper() for v in vels if v)
        oi_pts = 1.5 if oi_covering else 0.0

        # Straddle momentum 15m flip (0–1 pt)
        valid_smom = [v for v in smom15s if v is not None]
        smom_turned = False
        if len(valid_smom) > 1:
            if move_dir == "UP":
                smom_turned = valid_smom[0] < 0 and valid_smom[-1] >= 0
            else:
                smom_turned = valid_smom[0] > 0 and valid_smom[-1] <= 0
        smom_pts = 1.0 if smom_turned else 0.0

        # Liq accel aligned (0–0.5 pts)
        liq_target = "UPSIDE" if move_dir == "UP" else "DOWNSIDE"
        liq_hits   = sum(1 for det, d in zip(liq_dets, liq_dirs) if det and d == liq_target)
        liq_ratio  = liq_hits / n
        liq_pts    = 0.5 if liq_ratio >= 0.2 else 0.0

        # Squeeze (0–0.5 pts)
        sq_kw   = "UPSIDE SQUEEZE" if move_dir == "UP" else "DOWNSIDE SQUEEZE"
        sq_present = any(sq_kw in s.upper() for s in squeezes if s)
        sq_pts  = 0.5 if sq_present else 0.0

        raw = s_pts + streak_pts + pcr_pts + oi_pts + smom_pts + liq_pts + sq_pts
        v2  = round(raw * (10.0 / 12.0), 2)

        breakdown = {
            "compression": round(compression, 3),
            "s_pts":        s_pts,
            "streak":       streak,
            "streak_pts":   streak_pts,
            "pcr_end":      round(pcr_end, 3),
            "pcr_crossed":  pcr_crossed,
            "pcr_aligned":  pcr_aligned,
            "pcr_pts":      pcr_pts,
            "oi_covering":  oi_covering,
            "oi_pts":       oi_pts,
            "smom_turned":  smom_turned,
            "smom_pts":     smom_pts,
            "liq_pts":      liq_pts,
            "sq_pts":       sq_pts,
        }
        return v2, breakdown

    def _setup_label(self, breakdown):
        parts = []
        if breakdown.get("s_pts", 0) >= 2.0:
            parts.append("compress")
        if breakdown.get("streak", 0) >= 5:
            parts.append("streak")
        if breakdown.get("pcr_crossed"):
            parts.append("PCR✗")
        elif breakdown.get("pcr_aligned"):
            parts.append("PCR✓")
        if breakdown.get("oi_covering"):
            parts.append("OI")
        if breakdown.get("smom_turned"):
            parts.append("smom")
        return "COIL" + (f"({','.join(parts)})" if parts else "")

    def _print_box(self, score, direction, confidence, action, breakdown, in_cooldown):
        score_bar = "█" * int(score) + "░" * (10 - int(score))
        dir_color = Fore.GREEN if direction == "LONG" else Fore.RED

        action_color = (Fore.GREEN if action == "TAKE TRADE" else
                        Fore.YELLOW if action == "STALK — WAIT FOR TRIGGER" else
                        Fore.WHITE)
        cooldown_tag = " [cooldown]" if in_cooldown else ""

        comp = breakdown.get("compression", 1.0)
        streak = breakdown.get("streak", 0)
        pcr_end = breakdown.get("pcr_end", 1.0)

        print(Fore.CYAN + "┌─ COIL ALERT ─────────────────────────────────────┐" + Style.RESET_ALL)
        print(f"│ Score : {score:.1f}/10  {score_bar}")
        print(f"│ Dir   : {dir_color}{direction}{Style.RESET_ALL}  |  "
              f"Compress:{comp:.2f}  Streak:{streak}  PCR:{pcr_end:.3f}")
        print(f"│ Action: {action_color}{action or 'QUIET'}{Style.RESET_ALL}{cooldown_tag}")
        print(Fore.CYAN + "└──────────────────────────────────────────────────┘" + Style.RESET_ALL)

    @staticmethod
    def _result(action, direction, score, confidence, setup, breakdown):
        return {
            "action":     action,
            "direction":  direction,
            "score":      score,
            "confidence": confidence,
            "setup":      setup,
            "coil_score": score,
            "breakdown":  breakdown,
            # Fields expected by downstream logging (sniper columns in CSV)
            "htf":        "",
            "htf_adj":    0.0,
            "radar":      "",
            "radar_adj":  0.0,
            "signal_scores": {},
        }
