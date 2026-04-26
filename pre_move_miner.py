"""
Pre-Move Signal Miner
=====================
Mines all signal logs to find what indicators consistently appear
BEFORE a sudden significant move (the "coiling fingerprint").

Output:
  - All significant moves found across every day/instrument
  - For each move: what the data showed at T-15, T-10, T-5, T-0
  - Aggregate stats: which indicator combinations have the best
    lead-time and precision for predicting imminent moves
  - A ranked "coil score" formula you can wire into the sniper
"""

import csv
import glob
import os
from collections import defaultdict
from datetime import datetime, time as dtime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Thresholds ──────────────────────────────────────────────────────────────
# A "significant move" = sustained directional move, not just noise
# 0.25% ≈ 190 pts SENSEX, 58 pts NIFTY — filters out random wiggles
LEAD_BUCKETS       = [5, 10, 15, 20, 30]  # minutes-before-move to score

# Straddle compression: ratio to its own rolling max over this window
STRADDLE_ROLL_MIN  = 60

# Only analyse moves that start after market open volatility settles
MIN_TRADE_TIME_MIN = 60    # ignore moves in first 60 min (09:15-10:15)

# Per-instrument configuration
INSTRUMENT_CONFIG = {
    "sensex": {
        "move_pts_threshold" : None,    # use pct instead
        "move_pct_threshold" : 0.25,    # 0.25% ≈ 180 pts on SENSEX
        "move_window_min"    : 15,      # look-forward window to detect move
        "precision_window_min": 20,
        "lookback_min"       : 60,
        "score_key"          : "coil_score_v2",
        "score_threshold"    : 4.5,
        "cooldown_min"       : 15,
    },
    "nifty": {
        "move_pts_threshold" : 100,     # absolute 100-pt move
        "move_pct_threshold" : None,    # not used
        "move_window_min"    : 30,      # must happen within 30 min
        "precision_window_min": 30,
        "lookback_min"       : 60,
        "score_key"          : "coil_score",  # V1 — NIFTY coil signal not validated
        "score_threshold"    : 4.5,
        "cooldown_min"       : 15,
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def ts_to_min(ts: str) -> int:
    """Convert HH:MM:SS to minutes-since-midnight."""
    h, m, s = ts.split(":")
    return int(h) * 60 + int(m)


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_bool(val):
    return str(val).strip().lower() == "true"


def load_file(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["_min"] = ts_to_min(row["timestamp"])
            row["_spot"] = safe_float(row["spot"])
            rows.append(row)
    return rows


# ── Feature extraction for a single row/window ──────────────────────────────

def window_before(rows: list[dict], idx: int, lookback: int) -> list[dict]:
    """All rows within `lookback` minutes before rows[idx]."""
    cutoff = rows[idx]["_min"] - lookback
    return [r for r in rows[:idx] if r["_min"] >= cutoff]


def compute_features(window: list[dict], move_dir: str) -> dict:
    """
    Compute the pre-move fingerprint features for a window of rows.
    move_dir: 'UP' or 'DOWN'
    """
    if not window:
        return {}

    straddles     = [safe_float(r["straddle"]) for r in window]
    pcrs          = [safe_float(r["pcr"])      for r in window]
    mp_dirs       = [r.get("move_prob_dir","") for r in window]
    biases        = [r.get("bias","")          for r in window]
    smom15        = [safe_float(r.get("straddle_mom_15m","0")) for r in window]
    smom5         = [safe_float(r.get("straddle_mom_5m","0"))  for r in window]
    squeezes      = [r.get("squeeze","")       for r in window]
    oi_vels       = [r.get("oi_velocity","")   for r in window]
    liq_accels    = [r.get("liq_accel","")     for r in window]
    liq_dirs      = [r.get("liq_accel_dir","") for r in window]

    n = len(window)
    last = window[-1]

    # 1. Straddle compression ratio (lower = more compressed)
    roll_max = max(straddles) if straddles else 1
    straddle_compression = (min(straddles) / roll_max) if roll_max else 1.0
    straddle_at_end      = straddles[-1]
    straddle_trend       = straddles[-1] - straddles[0] if n > 1 else 0

    # 2. move_prob_dir consistency toward the eventual move direction
    target_mpdir = "UPSIDE" if move_dir == "UP" else "DOWNSIDE"
    mpdir_hits   = sum(1 for d in mp_dirs if d == target_mpdir)
    mpdir_ratio  = mpdir_hits / n if n else 0

    # How many consecutive rows at end had the right move_prob_dir
    mpdir_streak = 0
    for d in reversed(mp_dirs):
        if d == target_mpdir:
            mpdir_streak += 1
        else:
            break

    # 3. PCR features
    pcr_end   = pcrs[-1] if pcrs else 0
    pcr_above1 = sum(1 for p in pcrs if p > 1.0)
    pcr_trend  = pcrs[-1] - pcrs[0] if len(pcrs) > 1 else 0
    # For UP moves: PCR>1 is bullish; for DOWN moves: PCR<1 is bearish
    pcr_aligned = (pcr_end > 1.0 and move_dir == "UP") or \
                  (pcr_end < 1.0 and move_dir == "DOWN")

    # 4. Bias softening toward the move
    bias_map = {"BULLISH": 1, "RANGE": 0, "BEARISH": -1, "UNCLEAR": 0}
    bias_vals = [bias_map.get(b, 0) for b in biases]
    bias_end  = bias_vals[-1] if bias_vals else 0
    # Did bias shift in the right direction during the window?
    bias_shifted_right = (
        (move_dir == "UP"   and bias_end >= 0 and bias_vals[0] < 0) or
        (move_dir == "DOWN" and bias_end <= 0 and bias_vals[0] > 0)
    )
    bias_aligned = (move_dir == "UP" and bias_end >= 0) or \
                   (move_dir == "DOWN" and bias_end <= 0)

    # 5. Straddle momentum (15m) — positive = expansion, negative = compression
    smom15_end    = smom15[-1] if smom15 else 0
    smom15_turned = False          # did it flip sign in this window?
    if len(smom15) > 1:
        if move_dir == "UP":
            smom15_turned = smom15[0] < 0 and smom15[-1] >= 0
        else:
            smom15_turned = smom15[0] > 0 and smom15[-1] <= 0

    # 6. OI velocity events
    covering_kw = "PUT COVERING" if move_dir == "UP" else "CALL COVERING"
    oi_covering  = any(covering_kw in v.upper() for v in oi_vels if v)
    oi_conflicted= any("CONFLICTED" in v.upper() for v in oi_vels if v)

    # 7. Liquidity acceleration aligned with move
    liq_target  = "UPSIDE" if move_dir == "UP" else "DOWNSIDE"
    liq_hits    = sum(1 for a, d in zip(liq_accels, liq_dirs)
                      if safe_bool(a) and d == liq_target)
    liq_ratio   = liq_hits / n if n else 0

    # 8. Squeeze
    squeeze_kw  = "UPSIDE SQUEEZE" if move_dir == "UP" else "DOWNSIDE SQUEEZE"
    squeeze_present = any(squeeze_kw in s.upper() for s in squeezes if s)

    # 9a. V1 composite coil score (kept for reference)
    v1 = 0.0
    v1 += 3.0 * (1 - straddle_compression)
    v1 += 2.0 * mpdir_ratio
    v1 += 1.5 * int(pcr_aligned)
    v1 += 1.0 * int(bias_aligned)
    v1 += 1.0 * int(oi_covering)
    v1 += 0.5 * int(squeeze_present)
    v1 += 1.0 * liq_ratio

    # 9b. V2 SENSEX-tuned coil score (0–10)
    # Drops noisy bias_aligned; non-linear straddle; streak over ratio;
    # rewards PCR crossing 1.0 as a discrete event; smom flip.

    # Straddle compression — step-wise non-linear (0–4 pts)
    if straddle_compression < 0.80:
        s_pts = 4.0
    elif straddle_compression < 0.85:
        s_pts = 3.0
    elif straddle_compression < 0.90:
        s_pts = 2.0
    elif straddle_compression < 0.93:
        s_pts = 0.5
    else:
        s_pts = 0.0

    # move_prob_dir streak — more valuable than overall ratio (0–2.5 pts)
    streak_pts = min(mpdir_streak / 8.0, 1.0) * 2.5

    # PCR crossing 1.0 in the window — discrete event (0–2 pts)
    # UP: PCR was below 1 at start of window, crossed above = accumulation flip
    # DOWN: PCR was above 1, crossed below = distribution flip
    pcr_mid    = n // 2
    pcr_first  = pcrs[:pcr_mid] if pcr_mid else pcrs
    pcr_second = pcrs[pcr_mid:] if pcr_mid else pcrs
    if move_dir == "UP":
        pcr_crossed = (
            any(p <= 1.0 for p in pcr_first) and
            any(p >  1.0 for p in pcr_second)
        )
    else:
        pcr_crossed = (
            any(p >= 1.0 for p in pcr_first) and
            any(p <  1.0 for p in pcr_second)
        )
    pcr_pts = 2.0 if pcr_crossed else (0.5 if pcr_aligned else 0.0)

    # OI covering event (0–1.5 pts)
    oi_pts = 1.5 if oi_covering else 0.0

    # Straddle momentum flip from compression to expansion (0–1 pt)
    smom_pts = 1.0 if smom15_turned else 0.0

    # Liq accel aligned — only reward when ratio is meaningful (0–0.5 pts)
    liq_pts = 0.5 if liq_ratio >= 0.2 else 0.0

    # Squeeze present (0–0.5 pts)
    sq_pts = 0.5 if squeeze_present else 0.0

    raw_v2 = s_pts + streak_pts + pcr_pts + oi_pts + smom_pts + liq_pts + sq_pts
    v2 = round(raw_v2 * (10.0 / 12.0), 2)   # normalise max ~12 → 10

    # 9c. NIFTY V2 score (0–10)
    # NIFTY moves don't coil — straddle compression barely moves vs baseline.
    # Primary drivers: OI covering (institutional repositioning) + PCR crossing.
    # Secondary: long mpdir streak (catches momentum-driven breakouts).
    # Straddle compression only rewarded at extremes (< 0.85).

    # OI covering — #1 signal for NIFTY (0–4 pts)
    nifty_oi_pts = 4.0 if oi_covering else 0.0

    # PCR crossing 1.0 — discrete institutional flip (0–3 pts)
    nifty_pcr_pts = 3.0 if pcr_crossed else (0.5 if pcr_aligned else 0.0)

    # Straddle momentum flip (0–2 pts) — more weight than SENSEX V2
    nifty_smom_pts = 2.0 if smom15_turned else 0.0

    # Long mpdir streak — catches sustained directional momentum (0–2 pts)
    # Needs ≥15 candles to add meaningful signal; caps at 30
    nifty_streak_pts = min(mpdir_streak / 30.0, 1.0) * 2.0

    # Extreme straddle compression only (0–1 pt)
    nifty_s_pts = 1.0 if straddle_compression < 0.85 else 0.0

    # Liq accel + squeeze (0–0.5 pts each)
    nifty_liq_pts = 0.5 if liq_ratio >= 0.2 else 0.0
    nifty_sq_pts  = 0.5 if squeeze_present else 0.0

    raw_nv2 = (nifty_oi_pts + nifty_pcr_pts + nifty_smom_pts +
               nifty_streak_pts + nifty_s_pts + nifty_liq_pts + nifty_sq_pts)
    nv2 = round(raw_nv2 * (10.0 / 13.0), 2)   # normalise max ~13 → 10

    return {
        "straddle_compression"  : round(straddle_compression, 3),
        "straddle_at_end"       : round(straddle_at_end, 1),
        "straddle_trend"        : round(straddle_trend, 1),
        "mpdir_ratio"           : round(mpdir_ratio, 2),
        "mpdir_streak"          : mpdir_streak,
        "pcr_end"               : round(pcr_end, 3),
        "pcr_above1_count"      : pcr_above1,
        "pcr_aligned"           : pcr_aligned,
        "pcr_crossed"           : pcr_crossed,
        "pcr_trend"             : round(pcr_trend, 3),
        "bias_shifted_right"    : bias_shifted_right,
        "bias_aligned"          : bias_aligned,
        "bias_end"              : bias_end,
        "smom15_end"            : round(smom15_end, 2),
        "smom15_turned"         : smom15_turned,
        "oi_covering"           : oi_covering,
        "oi_conflicted"         : oi_conflicted,
        "liq_ratio"             : round(liq_ratio, 2),
        "squeeze_present"       : squeeze_present,
        "coil_score"            : round(v1, 2),
        "coil_score_v2"         : v2,
        "coil_score_nifty_v2"   : nv2,
        "window_rows"           : n,
    }


# ── Move detection ───────────────────────────────────────────────────────────

def find_moves(rows: list[dict], cfg: dict) -> list[dict]:
    """
    Find all candles that are the START of a significant sudden move.
    cfg keys: move_pts_threshold, move_pct_threshold, move_window_min
    Returns list of dicts with move metadata.
    """
    move_pts_thr = cfg.get("move_pts_threshold")   # absolute pts, or None
    move_pct_thr = cfg.get("move_pct_threshold")   # pct, or None
    move_win     = cfg["move_window_min"]

    moves = []
    used_minutes = set()

    open_min = rows[0]["_min"] if rows else 555

    for i, row in enumerate(rows):
        t0  = row["_min"]
        s0  = row["_spot"]
        if s0 == 0:
            continue

        if t0 - open_min < MIN_TRADE_TIME_MIN:
            continue

        future = [r for r in rows[i+1:]
                  if r["_min"] <= t0 + move_win and r["_spot"] > 0]
        if not future:
            continue

        max_spot = max(r["_spot"] for r in future)
        min_spot = min(r["_spot"] for r in future)
        up_pts   = max_spot - s0
        dn_pts   = s0 - min_spot
        up_pct   = up_pts / s0 * 100
        dn_pct   = dn_pts / s0 * 100

        for pts, pct, direction, extreme in [
            (up_pts, up_pct, "UP",   max_spot),
            (dn_pts, dn_pct, "DOWN", min_spot),
        ]:
            if move_pts_thr is not None:
                qualifies = pts >= move_pts_thr
            else:
                qualifies = pct >= move_pct_thr

            if qualifies:
                key = (t0 // 10, direction)
                if key in used_minutes:
                    continue
                used_minutes.add(key)

                moves.append({
                    "start_ts"    : row["timestamp"],
                    "start_min"   : t0,
                    "start_spot"  : round(s0, 1),
                    "extreme_spot": round(extreme, 1),
                    "move_pct"    : round(pct, 2),
                    "move_dir"    : direction,
                    "move_pts"    : round(abs(extreme - s0), 1),
                    "start_idx"   : i,
                })

    return moves


# ── Per-move analysis ────────────────────────────────────────────────────────

def analyse_move(rows, move, lookback_min=60):
    idx = move["start_idx"]
    result = {"move": move, "lead_features": {}}

    for lead in LEAD_BUCKETS:
        cutoff_min = move["start_min"] - lead
        if cutoff_min < 0:
            continue
        candidates = [r for r in rows if r["_min"] <= move["start_min"] - lead]
        if not candidates:
            continue
        anchor_idx  = rows.index(candidates[-1])
        pre_window  = window_before(rows, anchor_idx + 1, lookback_min)
        if not pre_window:
            continue
        feats = compute_features(pre_window, move["move_dir"])
        result["lead_features"][lead] = feats

    return result


# ── Aggregate stats ──────────────────────────────────────────────────────────

def aggregate(all_analyses, score_key="coil_score"):
    """
    For each lead bucket, compute mean score and hit rates of each feature.
    """
    from statistics import mean, stdev

    stats = {}
    for lead in LEAD_BUCKETS:
        scores, feats_list = [], []
        for a in all_analyses:
            f = a["lead_features"].get(lead)
            if f:
                scores.append(f.get(score_key, f["coil_score"]))
                feats_list.append(f)

        if not scores:
            continue

        bool_keys = [
            "pcr_aligned", "pcr_crossed", "bias_shifted_right",
            "smom15_turned", "oi_covering", "squeeze_present",
        ]
        bool_rates = {}
        for k in bool_keys:
            vals = [f.get(k, False) for f in feats_list]
            bool_rates[k] = round(sum(vals) / len(vals) * 100, 1)

        stats[lead] = {
            "n"                     : len(scores),
            "mean_coil_score"       : round(mean(scores), 2),
            "stdev_coil_score"      : round(stdev(scores), 2) if len(scores)>1 else 0,
            "mean_straddle_compress": round(mean(f["straddle_compression"] for f in feats_list), 3),
            "mean_mpdir_ratio"      : round(mean(f["mpdir_ratio"] for f in feats_list), 2),
            "mean_mpdir_streak"     : round(mean(f["mpdir_streak"] for f in feats_list), 1),
            "mean_pcr_above1_count" : round(mean(f["pcr_above1_count"] for f in feats_list), 1),
            "liq_ratio"             : round(mean(f["liq_ratio"] for f in feats_list), 3),
            **bool_rates,
        }

    return stats


# ── Quiet-window baseline ────────────────────────────────────────────────────

def sample_quiet_windows(rows, moves, n_samples=30, move_window_min=15, lookback_min=60):
    """
    Sample windows that are NOT near any move, to get a baseline coil score.
    This is what the score looks like in ordinary market conditions.
    """
    move_minutes = {m["start_min"] for m in moves}
    quiet_rows = [
        r for r in rows
        if all(abs(r["_min"] - mm) > move_window_min + 5 for mm in move_minutes)
        and r["_min"] > 60
    ]
    if not quiet_rows:
        return []

    import random
    random.seed(42)
    sample_rows = random.sample(quiet_rows, min(n_samples, len(quiet_rows)))
    results = []
    for r in sample_rows:
        idx = rows.index(r)
        w   = window_before(rows, idx, lookback_min)
        if len(w) < 5:
            continue
        # Use UP direction for baseline (direction doesn't matter much for baseline)
        f = compute_features(w, "UP")
        results.append(f)
    return results


# ── Per-instrument analysis ──────────────────────────────────────────────────

SENSEX_EXPIRY_DOW = 3   # Thursday = 3 (Mon=0)
NIFTY_EXPIRY_DOW  = 1   # Tuesday  = 1

def is_expiry(fname: str, instrument: str) -> bool:
    """Return True if the file's date is an expiry day for that instrument."""
    from datetime import datetime
    date_str = fname.split(f"signals_log_{instrument}_")[-1].replace(".csv","")
    try:
        d = datetime.strptime(date_str, "%d%m%Y")
        expiry_dow = SENSEX_EXPIRY_DOW if instrument == "sensex" else NIFTY_EXPIRY_DOW
        return d.weekday() == expiry_dow
    except Exception:
        return False


def run_instrument(instrument: str, files: list[str]):
    """Run full analysis pipeline for one instrument and print results."""
    from statistics import mean, stdev

    cfg = INSTRUMENT_CONFIG[instrument]
    LOOKBACK_MIN         = cfg["lookback_min"]
    PRECISION_WINDOW_MIN = cfg["precision_window_min"]
    score_key            = cfg["score_key"]
    SCORE_THRESHOLD      = cfg["score_threshold"]
    COOLDOWN_MIN         = cfg["cooldown_min"]

    print(f"\n{'#'*70}")
    print(f"  INSTRUMENT: {instrument.upper()}")
    move_desc = (f"{cfg['move_pts_threshold']}pts" if cfg["move_pts_threshold"]
                 else f"{cfg['move_pct_threshold']}%")
    print(f"  Threshold: {move_desc} in {cfg['move_window_min']} min")
    print(f"{'#'*70}\n")

    all_moves_by_file  = {}
    all_analyses       = []
    all_quiet_features = []
    total_moves        = 0

    for path in files:
        fname = os.path.basename(path)
        try:
            rows = load_file(path)
        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")
            continue

        moves    = find_moves(rows, cfg)
        analyses = [analyse_move(rows, m, lookback_min=LOOKBACK_MIN) for m in moves]
        quiet    = sample_quiet_windows(rows, moves, n_samples=20,
                                        move_window_min=cfg["move_window_min"],
                                        lookback_min=LOOKBACK_MIN)

        all_moves_by_file[fname] = moves
        all_analyses.extend(analyses)
        all_quiet_features.extend(quiet)
        total_moves += len(moves)

        if moves:
            print(f"  {fname}: {len(moves)} moves")
            for m in moves:
                print(f"    {m['start_ts']}  {m['move_dir']:4}  "
                      f"{m['move_pts']:6.1f}pts ({m['move_pct']:.2f}%)")

    print(f"\n  TOTAL MOVES: {total_moves} across {len(files)} files\n")

    if not all_analyses:
        print("  No moves found — lower MOVE_PCT_THRESHOLD or check data.")
        return

    # ── Feature hit rates ────────────────────────────────────────────────────
    print("  FEATURE HIT RATES BEFORE MOVES (by lead time)\n")
    print(f"  {'Feature':<26}", end="")
    for lead in LEAD_BUCKETS:
        print(f"  T-{lead:2d}m", end="")
    print()
    print("  " + "-" * 65)

    agg = aggregate(all_analyses, score_key)
    score_label = {"sensex": "V2 (SENSEX)", "nifty": "V2 (NIFTY)"}.get(instrument, "V1")
    feature_rows = [
        ("mean_coil_score",        f"Coil Score {score_label}"),
        ("mean_straddle_compress", "Straddle Compression"),
        ("mean_mpdir_ratio",       "move_prob_dir Hit Rate"),
        ("mean_mpdir_streak",      "move_prob_dir Streak"),
        ("pcr_aligned",            "PCR Aligned %"),
        ("pcr_crossed",            "PCR Crossed 1.0 %"),
        ("bias_shifted_right",     "Bias Shifted Right %"),
        ("smom15_turned",          "Straddle Mom Flip %"),
        ("oi_covering",            "OI Covering Event %"),
        ("squeeze_present",        "Squeeze Present %"),
        ("liq_ratio",              "Liq Accel Ratio"),
        ("mean_pcr_above1_count",  "PCR>1 Candle Count"),
    ]

    for key, label in feature_rows:
        print(f"  {label:<26}", end="")
        for lead in LEAD_BUCKETS:
            val = agg.get(lead, {}).get(key, "-")
            if val == "-":
                print(f"  {'':>5}", end="")
            elif isinstance(val, float):
                print(f"  {val:>5.2f}", end="")
            else:
                print(f"  {val:>5}", end="")
        print()

    # ── Quiet baseline ───────────────────────────────────────────────────────
    if all_quiet_features:
        q_score    = mean(f["coil_score"]           for f in all_quiet_features)
        q_compress = mean(f["straddle_compression"] for f in all_quiet_features)
        q_mpdir    = mean(f["mpdir_ratio"]          for f in all_quiet_features)
        q_pcr      = sum(1 for f in all_quiet_features if f["pcr_aligned"]) / len(all_quiet_features) * 100
        print(f"\n  {'─'*65}")
        print(f"  QUIET BASELINE (n={len(all_quiet_features)} random quiet windows)")
        print(f"  Coil Score: {q_score:.2f}  |  Straddle Compress: {q_compress:.3f}  "
              f"|  mpdir Hit: {q_mpdir:.2f}  |  PCR Aligned: {q_pcr:.1f}%")

    # ── Individual move detail ───────────────────────────────────────────────
    print(f"\n  INDIVIDUAL MOVE DETAIL (T-15 features)\n")
    sv = "V2" if instrument == "sensex" else "V1"
    print(f"  {'Date/Time':<32} {'Dir':4} {'Pts':6} "
          f"{f'Scr{sv}':>6} {'SCompr':>7} {'Streak':>6} {'PCR':>6} "
          f"{'PCRx':>4} {'OICv':>4} {'MomF':>4}")
    print("  " + "-" * 90)

    for a in all_analyses:
        m   = a["move"]
        f15 = a["lead_features"].get(15, {})
        if not f15:
            continue
        fname_short = ""
        for fn, mv_list in all_moves_by_file.items():
            if m in mv_list:
                fname_short = fn.replace(f"signals_log_{instrument}_","").replace(".csv","")
                break
        print(f"  {fname_short:<14} {m['start_ts']:8}  "
              f"{m['move_dir']:4} {m['move_pts']:6.0f}  "
              f"{f15.get(score_key, f15.get('coil_score',0)):6.2f}  "
              f"{f15.get('straddle_compression',0):6.3f}  "
              f"{f15.get('mpdir_streak',0):5}  "
              f"{f15.get('pcr_end',0):5.2f}  "
              f"{'X' if f15.get('pcr_crossed') else '.':3}  "
              f"{'X' if f15.get('oi_covering') else '.':3}  "
              f"{'X' if f15.get('smom15_turned') else '.':3}")

    # ── Precision analysis (deduplicated coil events) ───────────────────────
    print(f"\n  PRECISION — deduplicated coil events (one entry per coil run)\n")

    # Time-of-day buckets (minutes since midnight)
    TOD_BUCKETS = [
        ("Morning  10:15–11:30", 615, 690),
        ("Midday   11:30–13:00", 690, 780),
        ("Afternoon13:00–14:30", 780, 870),
        ("Close    14:30–15:30", 870, 930),
    ]
    precision_results = []
    for path in files:
        fname = os.path.basename(path)
        try:
            rows = load_file(path)
        except Exception:
            continue

        open_min = rows[0]["_min"] if rows else 555

        # Pre-compute score for every eligible row for both directions
        scored = []   # (row_index, direction, score, features)
        for i, row in enumerate(rows):
            t0 = row["_min"]
            if t0 - open_min < MIN_TRADE_TIME_MIN:
                continue
            pre_w = window_before(rows, i, LOOKBACK_MIN)
            if len(pre_w) < 10:
                continue
            for direction in ("UP", "DOWN"):
                f     = compute_features(pre_w, direction)
                score = f.get(score_key, f["coil_score"])
                scored.append((i, direction, score, f, row["_min"], row["timestamp"]))

        # Deduplicate: for each direction, walk through time and fire one event
        # at the PEAK score of each above-threshold run, with cooldown between events
        for direction in ("UP", "DOWN"):
            dir_scored = [(i, sc, f, t0, ts)
                          for i, d, sc, f, t0, ts in scored if d == direction]

            last_event_min = -999
            j = 0
            while j < len(dir_scored):
                i, sc, f, t0, ts = dir_scored[j]
                if sc < SCORE_THRESHOLD or t0 - last_event_min < COOLDOWN_MIN:
                    j += 1
                    continue

                # Scan forward to find the peak score in this contiguous run
                peak_i, peak_sc, peak_f, peak_t0, peak_ts = i, sc, f, t0, ts
                k = j + 1
                while k < len(dir_scored):
                    ni, nsc, nf, nt0, nts = dir_scored[k]
                    if nsc < SCORE_THRESHOLD or nt0 > t0 + 20:
                        break
                    if nsc > peak_sc:
                        peak_i, peak_sc, peak_f, peak_t0, peak_ts = ni, nsc, nf, nt0, nts
                    k += 1

                # Measure over a longer window for MAE/MFE (60 min)
                MAE_MFE_WINDOW = 60
                future_long = [r for r in rows[peak_i:]
                               if peak_t0 <= r["_min"] <= peak_t0 + MAE_MFE_WINDOW
                               and r["_spot"] > 0]
                future_prec = [r for r in future_long
                               if r["_min"] <= peak_t0 + PRECISION_WINDOW_MIN]

                if not future_prec:
                    j = k
                    continue

                s0 = rows[peak_i]["_spot"]

                # MFE: best move in the intended direction (pts)
                if direction == "UP":
                    mfe_pts = max(r["_spot"] for r in future_long) - s0
                    mae_pts = s0 - min(r["_spot"] for r in future_long)   # adverse = down
                else:
                    mfe_pts = s0 - min(r["_spot"] for r in future_long)
                    mae_pts = max(r["_spot"] for r in future_long) - s0   # adverse = up

                mfe_pts = max(mfe_pts, 0)
                mae_pts = max(mae_pts, 0)

                # Precision check over shorter window
                if direction == "UP":
                    extreme_prec = max(r["_spot"] for r in future_prec)
                else:
                    extreme_prec = min(r["_spot"] for r in future_prec)
                prec_pts = abs(extreme_prec - s0)
                pct = prec_pts / s0 * 100
                if cfg["move_pts_threshold"] is not None:
                    moved = prec_pts >= cfg["move_pts_threshold"]
                else:
                    moved = pct >= cfg["move_pct_threshold"]

                # Store price path (spot values, 60-min window) for PnL sim
                price_path = [r["_spot"] for r in future_long if r["_spot"] > 0]

                precision_results.append({
                    "score"       : peak_sc,
                    "moved"       : moved,
                    "move_pct"    : round(pct, 2),
                    "direction"   : direction,
                    "fname"       : fname,
                    "ts"          : peak_ts,
                    "t0"          : peak_t0,
                    "streak"      : peak_f["mpdir_streak"],
                    "pcr_crossed" : peak_f.get("pcr_crossed", False),
                    "oi_covering" : peak_f["oi_covering"],
                    "smom_turned" : peak_f["smom15_turned"],
                    "compress"    : peak_f["straddle_compression"],
                    "mfe_pts"     : round(mfe_pts, 1),
                    "mae_pts"     : round(mae_pts, 1),
                    "entry_spot"  : round(s0, 1),
                    "expiry"      : is_expiry(fname, instrument),
                    "price_path"  : price_path,
                })

                last_event_min = peak_t0
                j = k  # skip past the end of this run

    def precision_row(label, subset, indent=2):
        if not subset:
            print(f"  {'':>{indent}}{label:<44}  (no data)")
            return
        n   = len(subset)
        hit = sum(1 for p in subset if p["moved"])
        pct = hit / n * 100
        avg = mean(p["move_pct"] for p in subset)
        bar = "█" * int(pct / 5)   # visual bar, 1 block per 5%
        print(f"  {label:<44} {n:>5}  {hit:>4}  {pct:>6.1f}%  {avg:>6.2f}%  {bar}")

    if precision_results:
        hdr = f"  {'Filter / Segment':<44} {'N':>5}  {'Hit':>4}  {'Prec':>7}  {'AvgMv':>7}  Chart"
        print(hdr)
        print("  " + "─" * 80)

        # ── Score-band breakdown ─────────────────────────────────────────────
        print("  Score bands (all deduplicated coil events):")
        for lo, hi in [(3.0, 4.5), (4.5, 5.5), (5.5, 7.0), (7.0, 10.0)]:
            sub = [p for p in precision_results if lo <= p["score"] < hi]
            precision_row(f"  score {lo:.1f}–{hi:.1f}", sub)

        print()

        # ── Expiry vs non-expiry split ───────────────────────────────────────
        expiry_all  = [p for p in precision_results if p["expiry"]]
        normal_all  = [p for p in precision_results if not p["expiry"]]
        print(f"  EXPIRY vs NON-EXPIRY (all score ≥ 3.0 deduplicated events):")
        precision_row(f"  Expiry days     ({len(expiry_all)} events)",  expiry_all)
        precision_row(f"  Non-expiry days ({len(normal_all)} events)", normal_all)

        print()
        expiry_b45  = [p for p in expiry_all  if p["score"] >= 4.5]
        normal_b45  = [p for p in normal_all  if p["score"] >= 4.5]
        print(f"  EXPIRY vs NON-EXPIRY (score ≥ 4.5):")
        precision_row(f"  Expiry   score ≥ 4.5 ({len(expiry_b45)} events)",  expiry_b45)
        precision_row(f"  Non-expiry score ≥ 4.5 ({len(normal_b45)} events)", normal_b45)

        print()
        print(f"  Compression profile — expiry vs non-expiry (score ≥ 4.5):")
        if expiry_b45:
            ec = sorted(p["compress"] for p in expiry_b45)
            print(f"  Expiry   compress: avg={mean(ec):.3f}  "
                  f"min={min(ec):.3f}  med={ec[len(ec)//2]:.3f}  max={max(ec):.3f}")
        if normal_b45:
            nc = sorted(p["compress"] for p in normal_b45)
            print(f"  Non-exp  compress: avg={mean(nc):.3f}  "
                  f"min={min(nc):.3f}  med={nc[len(nc)//2]:.3f}  max={max(nc):.3f}")

        print()

        # ── Filter combos on score ≥ 4.5 ────────────────────────────────────
        base = [p for p in precision_results if p["score"] >= 4.5]
        print(f"  Filters on score ≥ {SCORE_THRESHOLD} ({len(base)} events):")
        precision_row("  no extra filter (baseline)", base)
        precision_row("  + streak ≥ 5",
                      [p for p in base if p["streak"] >= 5])
        precision_row("  + streak ≥ 10",
                      [p for p in base if p["streak"] >= 10])
        precision_row("  + PCR crossed",
                      [p for p in base if p["pcr_crossed"]])
        precision_row("  + OI covering",
                      [p for p in base if p["oi_covering"]])
        precision_row("  + compress < 0.85",
                      [p for p in base if p["compress"] < 0.85])
        precision_row("  + PCR crossed OR OI covering",
                      [p for p in base if p["pcr_crossed"] or p["oi_covering"]])
        precision_row("  + PCR crossed AND OI covering",
                      [p for p in base if p["pcr_crossed"] and p["oi_covering"]])

        print()

        # ── Two-filter combos ────────────────────────────────────────────────
        print("  Two-filter combos (score ≥ 4.5):")
        precision_row("  + compress<0.85 + PCR/OI (either)",
                      [p for p in base
                       if p["compress"] < 0.85 and (p["pcr_crossed"] or p["oi_covering"])])
        precision_row("  + compress<0.85 + streak≥5",
                      [p for p in base if p["compress"] < 0.85 and p["streak"] >= 5])
        precision_row("  + streak≥5 + PCR crossed",
                      [p for p in base if p["streak"] >= 5 and p["pcr_crossed"]])
        precision_row("  + streak≥5 + OI covering",
                      [p for p in base if p["streak"] >= 5 and p["oi_covering"]])

        print()

        # ── Three-filter combos — candidate COIL ALERT rules ────────────────
        print("  Three-filter combos — candidate COIL ALERT rules:")
        precision_row("  + compress<0.85 + PCR/OI + streak≥5",
                      [p for p in base
                       if p["compress"] < 0.85
                       and (p["pcr_crossed"] or p["oi_covering"])
                       and p["streak"] >= 5])
        precision_row("  + compress<0.85 + PCR crossed + streak≥5",
                      [p for p in base
                       if p["compress"] < 0.85 and p["pcr_crossed"] and p["streak"] >= 5])
        precision_row("  + compress<0.85 + OI covering + streak≥5",
                      [p for p in base
                       if p["compress"] < 0.85 and p["oi_covering"] and p["streak"] >= 5])
        precision_row("  + PCR crossed + OI covering + streak≥5",
                      [p for p in base
                       if p["pcr_crossed"] and p["oi_covering"] and p["streak"] >= 5])

        print()

        # ── Time-of-day breakdown ────────────────────────────────────────────
        print("  Time-of-day precision (score ≥ 4.5, all events):")
        for label, t_lo, t_hi in TOD_BUCKETS:
            sub = [p for p in base if t_lo <= p["t0"] < t_hi]
            precision_row(f"  {label}", sub)

        print()

        # ── Time-of-day × best filter ────────────────────────────────────────
        best_filter = [p for p in base
                       if p["compress"] < 0.85 and (p["pcr_crossed"] or p["oi_covering"])]
        print(f"  Time-of-day × compress<0.85 + PCR/OI ({len(best_filter)} events):")
        for label, t_lo, t_hi in TOD_BUCKETS:
            sub = [p for p in best_filter if t_lo <= p["t0"] < t_hi]
            precision_row(f"  {label}", sub)

        print()

        # ── Best examples ────────────────────────────────────────────────────
        best = sorted(
            [p for p in precision_results if p["score"] >= 5.0 and p["moved"]],
            key=lambda x: -x["score"]
        )[:12]
        if best:
            print(f"  Top converted alerts (score ≥ 5.0, moved=True):")
            for p in best:
                date = p["fname"].replace(f"signals_log_{instrument}_", "").replace(".csv","")
                print(f"    {date}  {p['ts']:8}  {p['direction']:4}  "
                      f"score={p['score']:.2f}  move={p['move_pct']:.2f}%  "
                      f"compress={p['compress']:.3f}  streak={p['streak']}")

        # ── MAE / MFE analysis ───────────────────────────────────────────────
        print(f"\n  MAE / MFE ANALYSIS (60-min window from alert, score ≥ 4.5)\n")

        def mfe_mae_stats(label, subset):
            if len(subset) < 3:
                return
            mfes = [p["mfe_pts"] for p in subset]
            maes = [p["mae_pts"] for p in subset]
            ratios = [p["mfe_pts"] / p["mae_pts"] if p["mae_pts"] > 0 else p["mfe_pts"]
                      for p in subset]
            moved = [p for p in subset if p["moved"]]
            notmoved = [p for p in subset if not p["moved"]]

            print(f"  ── {label} (n={len(subset)}) ──")
            print(f"     MFE: avg={mean(mfes):6.1f}pt  "
                  f"med={sorted(mfes)[len(mfes)//2]:6.1f}pt  "
                  f"p25={sorted(mfes)[len(mfes)//4]:6.1f}pt  "
                  f"p75={sorted(mfes)[len(mfes)*3//4]:6.1f}pt")
            print(f"     MAE: avg={mean(maes):6.1f}pt  "
                  f"med={sorted(maes)[len(maes)//2]:6.1f}pt  "
                  f"p25={sorted(maes)[len(maes)//4]:6.1f}pt  "
                  f"p75={sorted(maes)[len(maes)*3//4]:6.1f}pt")
            print(f"     MFE/MAE ratio: avg={mean(ratios):5.2f}x  "
                  f"med={sorted(ratios)[len(ratios)//2]:5.2f}x")

            if moved:
                print(f"     CONVERTED ({len(moved)})  → "
                      f"MFE avg={mean(p['mfe_pts'] for p in moved):6.1f}pt  "
                      f"MAE avg={mean(p['mae_pts'] for p in moved):6.1f}pt  "
                      f"ratio={mean(p['mfe_pts']/p['mae_pts'] if p['mae_pts']>0 else p['mfe_pts'] for p in moved):.2f}x")
            if notmoved:
                print(f"     NOT MOVED ({len(notmoved)}) → "
                      f"MFE avg={mean(p['mfe_pts'] for p in notmoved):6.1f}pt  "
                      f"MAE avg={mean(p['mae_pts'] for p in notmoved):6.1f}pt  "
                      f"ratio={mean(p['mfe_pts']/p['mae_pts'] if p['mae_pts']>0 else p['mfe_pts'] for p in notmoved):.2f}x")
            print()

        mfe_mae_stats("All events (score ≥ 4.5)", base)
        mfe_mae_stats("Expiry days only",
                      [p for p in base if p["expiry"]])
        mfe_mae_stats("Non-expiry days only",
                      [p for p in base if not p["expiry"]])
        mfe_mae_stats("Non-expiry + Afternoon 13:00–14:30",
                      [p for p in base if not p["expiry"] and 780 <= p["t0"] < 870])
        mfe_mae_stats("Non-expiry + Afternoon + compress < 0.85",
                      [p for p in base
                       if not p["expiry"] and 780 <= p["t0"] < 870 and p["compress"] < 0.85])

        # Distribution tables — non-expiry only (the clean signal)
        base_ne = [p for p in base if not p["expiry"]]
        if base_ne:
            print(f"  MAE distribution — NON-EXPIRY days only (score ≥ 4.5, n={len(base_ne)}):")
            print(f"  {'Stop (pts)':>12}  {'Within stop':>14}  {'%':>6}")
            print(f"  {'─'*38}")
            maes_ne = sorted(p["mae_pts"] for p in base_ne)
            for stop in [50, 75, 100, 125, 150, 200, 250, 300]:
                n_within = sum(1 for m in maes_ne if m <= stop)
                pct = n_within / len(maes_ne) * 100
                bar = "█" * int(pct / 5)
                print(f"  {stop:>12}  {n_within:>14}  {pct:>5.0f}%  {bar}")

            print()
            print(f"  MFE distribution — NON-EXPIRY days only (score ≥ 4.5, n={len(base_ne)}):")
            print(f"  {'Target (pts)':>12}  {'Reaches tgt':>14}  {'%':>6}")
            print(f"  {'─'*38}")
            mfes_ne = sorted(p["mfe_pts"] for p in base_ne)
            for tgt in [50, 100, 150, 200, 250, 300, 400, 500]:
                n_reached = sum(1 for m in mfes_ne if m >= tgt)
                pct = n_reached / len(mfes_ne) * 100
                bar = "█" * int(pct / 5)
                print(f"  {tgt:>12}  {n_reached:>14}  {pct:>5.0f}%  {bar}")

    # ── PnL simulation ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PnL SIMULATION — 2 lots (40 units), non-expiry events, score ≥ 4.5")
    print(f"  Rule: Stop both lots at -SL pts | Book Lot1 at 1:1 R:R | Trail Lot2 at 30%")
    print(f"  Note: PnL in index pts × units (multiply by option delta for rupee PnL)")
    print(f"{'='*70}\n")

    UNITS_PER_LOT = 20
    TRAIL_PCT     = 0.30    # give back 30% of peak gain on lot 2

    def simulate_trade(price_path, entry, direction, stop_pts):
        """
        Simulate the 2-lot trade on a price path.
        Returns (lot1_pts, lot2_pts, exit_reason_lot1, exit_reason_lot2)
        All values in index points per unit.
        """
        if not price_path or len(price_path) < 2:
            return -stop_pts, -stop_pts, "end-of-data", "end-of-data"

        target_pts = stop_pts * 1         # 1:1 R:R for lot 1
        lot1_open  = True
        lot2_open  = True
        lot1_pts   = -stop_pts            # default: stopped out
        lot2_pts   = -stop_pts
        exit1      = "stop"
        exit2      = "stop"
        peak_fav   = 0.0                  # highest favorable move seen so far

        for spot in price_path:
            if direction == "UP":
                fav = spot - entry
                adv = entry - spot
            else:
                fav = entry - spot
                adv = spot - entry

            # Update peak
            if fav > peak_fav:
                peak_fav = fav

            # Check stop (both lots)
            if adv >= stop_pts:
                if lot1_open:
                    lot1_pts, exit1 = -stop_pts, "stop"
                    lot1_open = False
                if lot2_open:
                    lot2_pts, exit2 = -stop_pts, "stop"
                    lot2_open = False
                break

            # Check lot 1 target (1:5)
            if lot1_open and fav >= target_pts:
                lot1_pts, exit1 = target_pts, "target"
                lot1_open = False

            # Check lot 2 trail (only after lot 1 is booked)
            if not lot1_open and lot2_open and peak_fav > 0:
                trail_stop = peak_fav * (1 - TRAIL_PCT)
                if fav <= trail_stop and peak_fav >= target_pts:
                    lot2_pts  = max(fav, 0)  # exit at current fav move
                    exit2     = f"trail@{peak_fav:.0f}→{fav:.0f}"
                    lot2_open = False

        # End of path — exit whatever is still open at last price
        if price_path:
            last = price_path[-1]
            last_fav = (last - entry) if direction == "UP" else (entry - last)
            if lot1_open:
                lot1_pts = last_fav
                exit1    = "end-of-path"
            if lot2_open:
                lot2_pts = last_fav
                exit2    = "end-of-path"

        return lot1_pts, lot2_pts, exit1, exit2

    non_expiry_events = [p for p in precision_results
                         if not p["expiry"] and p["score"] >= 4.5 and p["price_path"]]

    print(f"  Non-expiry events with price paths: {len(non_expiry_events)}\n")

    for stop_label, stop_pts in [("150-pt stop", 150), ("250-pt stop", 250)]:
        print(f"  ── {stop_label} (target = {stop_pts*1} pts on Lot1, trail 30% on Lot2) ──\n")

        total_lot1 = total_lot2 = 0
        wins = losses = trail_wins = 0
        trade_log = []

        for p in non_expiry_events:
            l1, l2, e1, e2 = simulate_trade(
                p["price_path"], p["entry_spot"], p["direction"], stop_pts
            )
            pnl_lot1_rs = l1 * UNITS_PER_LOT
            pnl_lot2_rs = l2 * UNITS_PER_LOT
            total_rs    = pnl_lot1_rs + pnl_lot2_rs
            total_lot1 += pnl_lot1_rs
            total_lot2 += pnl_lot2_rs

            if total_rs > 0:   wins += 1
            else:              losses += 1
            if "trail" in e2:  trail_wins += 1

            date = p["fname"].replace(f"signals_log_{instrument}_","").replace(".csv","")
            trade_log.append((date, p["ts"], p["direction"], l1, l2, total_rs, e1, e2))

        grand_total = total_lot1 + total_lot2
        n = len(non_expiry_events)

        # Print trade-by-trade
        print(f"  {'Date':<12} {'Time':>8}  {'Dir':>4}  "
              f"{'Lot1 pts':>9}  {'Lot2 pts':>9}  "
              f"{'Total pts':>10}  {'Total rs':>10}  Exit")
        print(f"  {'─'*90}")
        for date, ts, dirn, l1, l2, tot_rs, e1, e2 in trade_log:
            marker = " ✓" if tot_rs > 0 else " ✗"
            print(f"  {date:<12} {ts:>8}  {dirn:>4}  "
                  f"{l1:>+9.0f}  {l2:>+9.0f}  "
                  f"{l1+l2:>+10.0f}  {tot_rs:>+10.0f}{marker}  {e1} / {e2}")

        print(f"\n  {'─'*60}")
        print(f"  Trades: {n}  |  Wins: {wins}  |  Losses: {losses}  "
              f"|  Win rate: {wins/n*100:.0f}%")
        print(f"  Lot 1 total : {total_lot1:>+,.0f} pts-units")
        print(f"  Lot 2 total : {total_lot2:>+,.0f} pts-units")
        print(f"  Grand total : {grand_total:>+,.0f} pts-units")
        print(f"  Avg per trade: {grand_total/n:>+,.0f} pts-units")
        print(f"  Trail exits that were profitable: {trail_wins}\n")

    # ── Threshold recommendation ─────────────────────────────────────────────
    print(f"\n  RECOMMENDED THRESHOLDS\n")
    if all_quiet_features and all_analyses:
        q_scores    = [f.get(score_key, f["coil_score"]) for f in all_quiet_features]
        m_scores_15 = [a["lead_features"][15].get(score_key, a["lead_features"][15]["coil_score"])
                       for a in all_analyses if 15 in a["lead_features"]]

        if m_scores_15 and q_scores:
            q_mean = mean(q_scores)
            q_std  = stdev(q_scores) if len(q_scores) > 1 else 1
            m_mean = mean(m_scores_15)
            thr_lo = round(q_mean + 1.5 * q_std, 1)
            thr_hi = round(q_mean + 2.0 * q_std, 1)

            catch_lo = sum(1 for s in m_scores_15 if s >= thr_lo) / len(m_scores_15) * 100
            catch_hi = sum(1 for s in m_scores_15 if s >= thr_hi) / len(m_scores_15) * 100

            print(f"  Quiet baseline coil score : {q_mean:.2f}  (±{q_std:.2f})")
            print(f"  Pre-move mean (T-15)      : {m_mean:.2f}")
            print()
            print(f"  Alert LOW  (≥{thr_lo}) → catches {catch_lo:.0f}% of moves at T-15")
            print(f"  Alert HIGH (≥{thr_hi}) → catches {catch_hi:.0f}% of moves at T-15")

            if precision_results:
                for thr in [thr_lo, thr_hi]:
                    subset = [p for p in precision_results if p["score"] >= thr]
                    if subset:
                        prec = sum(1 for p in subset if p["moved"]) / len(subset) * 100
                        print(f"  Precision at {thr}: {len(subset)} alerts fired, "
                              f"{prec:.1f}% preceded a real move")

            print()
            print(f"  Rule for {instrument.upper()}:")
            print(f"    IF coil_score >= {thr_lo}")
            print( "    AND move_prob_dir consistent ≥ 10 candles")
            print( "    AND PCR aligned to direction")
            print( "    → COIL ALERT: significant move likely in next ~20 min")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "signals_log_*.csv")))
    print(f"Found {len(all_files)} total signal log files\n")

    for instrument in ("nifty", "sensex"):
        inst_files = [f for f in all_files if f"signals_log_{instrument}_" in f]
        run_instrument(instrument, inst_files)


if __name__ == "__main__":
    main()
