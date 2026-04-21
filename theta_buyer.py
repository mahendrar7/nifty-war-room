"""
theta_buyer.py — Theta engine for options BUYERS only.

Two core questions a buyer asks every tick:
  1. Entry:  "Is theta eating faster than any expected move can pay me back?"
  2. In-trade: "Is my position in a theta-vs-momentum race I'm winning or losing?"

Public API
----------
compute_buyer_theta_context(df, atm, spot, straddle, expiry, momentum_data,
                             days_to_expiry)
    → ThetaBuyerContext  (named dict-like)
    Call once per tick AFTER compute_strike_gammas/thetas.
    Produces every number + a single-line ACTION for the dashboard.

log_theta_to_trade(trade_dict, theta_ctx)
    → dict  (adds theta fields to the trade dict before CSV write)
    Call at entry and on every HTL tick for the trade log.

format_dashboard_line(theta_ctx)
    → str  (coloured, ready to print)

format_trade_log_line(theta_ctx, at)
    → str  (coloured, ready to print alongside HTL verdict)
"""

import math
from datetime import datetime
from colorama import Fore, Style


# =============================================================================
# THRESHOLDS — tweak in config if needed, sensible defaults here
# =============================================================================

# theta_pct = daily theta / straddle_price * 100
THETA_PCT_RICH      = 2.5   # >= this → "very rich" for buyer, danger zone
THETA_PCT_MODERATE  = 1.5   # 1.5–2.5 → moderate drag
# below 1.5 → thin, theta is not the problem

# race ratio: momentum_pts_per_5m / breakeven_pts_per_5m
# > 1.0 = winning, < 0.6 = losing, in between = watch
RACE_WIN_RATIO      = 1.0
RACE_WARN_RATIO     = 0.6

# IV collapse = straddle momentum below this (excess beyond theta)
IV_COLLAPSE_EXCESS  = -1.2   # % per 5m, after subtracting theta

# straddle expansion threshold to confirm vol bid
IV_EXPANSION_MIN    = 1.5    # % per 5m


# =============================================================================
# BREAKEVEN MOVE CALCULATOR
# =============================================================================

def compute_breakeven_move(entry_price, spot, theta_rs_per_day,
                            holding_minutes, lot_size=75):
    """
    How far does spot need to move to break even, given:
      - entry_price   : premium paid (₹)
      - spot          : current Nifty spot
      - theta_rs_per_day : daily ₹ theta burn for this strike (from gamma_engine)
      - holding_minutes  : how long you intend to hold
      - lot_size      : contract lot size

    Returns:
        breakeven_pts        — spot move needed (points)
        theta_cost_rs        — total theta cost over holding period (₹ per lot)
        theta_cost_pct       — theta cost as % of premium paid
        pts_per_5m_needed    — pace needed (pts/5min) to beat theta
    """
    # Fraction of day = holding_minutes / (6.25 * 60)  [6h15m session]
    fraction_of_day  = holding_minutes / 375.0
    theta_cost_rs    = abs(theta_rs_per_day) * fraction_of_day * lot_size
    theta_cost_pct   = (theta_cost_rs / (entry_price * lot_size) * 100
                        if entry_price > 0 else 0.0)

    # Break-even point: entry premium + theta bleed = option value at exit
    # For a rough estimate, 1pt move ≈ 0.5 delta for ATM (conservative)
    # Use delta=0.5 for ATM, adjusts naturally as premium changes
    delta_approx     = 0.5
    breakeven_pts    = (entry_price + abs(theta_rs_per_day) * fraction_of_day) / delta_approx
    pts_per_5m_needed = breakeven_pts / (holding_minutes / 5.0) if holding_minutes > 0 else 0.0

    return {
        "breakeven_pts":      round(breakeven_pts, 1),
        "theta_cost_rs":      round(theta_cost_rs, 0),
        "theta_cost_pct":     round(theta_cost_pct, 1),
        "pts_per_5m_needed":  round(pts_per_5m_needed, 2),
    }


# =============================================================================
# THETA VS MOMENTUM RACE
# =============================================================================

def compute_race(spot_history, theta_rs_per_day, straddle_price,
                 momentum_data, lot_size=75):
    """
    The buyer's core question: is my trade moving faster than theta is decaying?

    spot_history    : list of recent spot prices (from state.spot_history)
    theta_rs_per_day: daily theta in ₹ per unit (from compute_atm_theta_metrics)
    straddle_price  : current ATM straddle (proxy for premium level)
    momentum_data   : from update_straddle() — has momentum_5m

    Returns:
        momentum_pts_per_5m  — actual spot speed (pts per 5min)
        theta_pts_per_5m     — theta decay in equivalent spot pts per 5min
        race_ratio           — momentum / theta (> 1 = winning)
        verdict              — "WINNING" | "LOSING" | "STALLING" | "NO DATA"
        action               — clear action string
        action_color         — Fore color
    """
    # Spot speed — last 5 ticks = 5 min
    momentum_pts_5m = 0.0
    if spot_history and len(spot_history) >= 6:
        momentum_pts_5m = abs(spot_history[-1] - spot_history[-6])

    # Convert theta ₹/day to equivalent spot pts/5min
    # theta/day → theta/5min (as ₹), then ÷ 0.5 delta = pts needed
    theta_per_5m_rs   = abs(theta_rs_per_day) / 75.0   # per 5min tick (75 ticks/day)
    theta_pts_per_5m  = theta_per_5m_rs / 0.5           # approximate pts equivalent

    race_ratio = (momentum_pts_5m / theta_pts_per_5m
                  if theta_pts_per_5m > 0 else 0.0)

    # IV context from straddle momentum
    iv_mom = momentum_data["momentum_5m"] if momentum_data else 0.0

    if momentum_pts_5m == 0 and iv_mom < 0:
        verdict      = "STALLING"
        action       = (f"⚠  STALLING — No spot movement. Theta burning ₹{theta_per_5m_rs:.1f}/5m. "
                        f"EXIT if no move in next 2 ticks.")
        action_color = Fore.YELLOW

    elif race_ratio >= RACE_WIN_RATIO:
        verdict      = "WINNING"
        action       = (f"✅ WINNING RACE — Spot {momentum_pts_5m:.0f}pts/5m vs theta "
                        f"{theta_pts_per_5m:.1f}pts equiv. HOLD.")
        action_color = Fore.GREEN

    elif RACE_WARN_RATIO <= race_ratio < RACE_WIN_RATIO:
        verdict      = "WATCH"
        action       = (f"⚡ THETA CATCHING UP — Spot {momentum_pts_5m:.0f}pts/5m, "
                        f"need >{theta_pts_per_5m:.1f}pts to stay ahead. TRAIL STOP.")
        action_color = Fore.YELLOW

    elif race_ratio < RACE_WARN_RATIO and momentum_pts_5m > 0:
        verdict      = "LOSING"
        action       = (f"🚨 LOSING RACE — Theta eating faster than spot moving. "
                        f"Move >{theta_pts_per_5m:.1f}pts/5m needed, only {momentum_pts_5m:.0f}pts. EXIT.")
        action_color = Fore.RED

    else:
        verdict      = "NO DATA"
        action       = ""
        action_color = Fore.WHITE

    return {
        "momentum_pts_per_5m": round(momentum_pts_5m, 1),
        "theta_pts_per_5m":    round(theta_pts_per_5m, 2),
        "theta_per_5m_rs":     round(theta_per_5m_rs, 2),
        "race_ratio":          round(race_ratio, 2),
        "verdict":             verdict,
        "action":              action,
        "action_color":        action_color,
    }


# =============================================================================
# IV ENTRY FILTER — should a buyer even enter?
# =============================================================================

def compute_iv_entry_filter(straddle_momentum, theta_per_5m_rs,
                             straddle_price, days_to_expiry):
    """
    Before entering: is IV working FOR or AGAINST the buyer?

    Separates theta decay from IV move using real theta_per_5m_rs.
    Returns a clear ENTER / WAIT / AVOID with reason.
    """
    if straddle_momentum is None or straddle_price <= 0:
        return {
            "verdict":      "WAIT",
            "action":       "WAIT — No straddle data yet.",
            "action_color": Fore.WHITE,
            "excess_pct":   0.0,
        }

    # Expected compression from theta alone (as % of straddle per 5min)
    expected_pct = -(theta_per_5m_rs / straddle_price * 100)
    excess_pct   = straddle_momentum - expected_pct   # positive = more compression than theta

    # ── Classify ──────────────────────────────────────────────────────────────

    if straddle_momentum >= IV_EXPANSION_MIN:
        # IV is being BID — good for buyers
        verdict      = "ENTER"
        action       = (f"✅ IV EXPANDING +{straddle_momentum:.1f}% — Vol being bought. "
                        f"Good entry window for buyer. ACT NOW.")
        action_color = Fore.GREEN

    elif excess_pct < IV_COLLAPSE_EXCESS:
        # Straddle compressing beyond what theta explains — IV being crushed
        verdict      = "AVOID"
        action       = (f"🚨 IV COLLAPSING — Excess squeeze {excess_pct:.1f}% beyond theta. "
                        f"DO NOT BUY. Premium headwind will eat your P&L even if right on direction.")
        action_color = Fore.RED

    elif -1.2 <= excess_pct <= 0.3 and straddle_momentum < 0:
        # Compression is just theta — neutral for buyer
        verdict      = "WAIT"
        action       = (f"⏳ THETA DECAY ONLY — Straddle {straddle_momentum:.1f}% "
                        f"({days_to_expiry}DTE). Just the clock ticking. "
                        f"WAIT for IV to expand before buying.")
        action_color = Fore.YELLOW

    elif abs(straddle_momentum) < 0.5:
        # Flat — IV is stable, not compressing or expanding
        verdict      = "WAIT"
        action       = (f"⏳ IV FLAT — Straddle stable. No vol edge either way. "
                        f"Wait for expansion signal before entry.")
        action_color = Fore.YELLOW

    else:
        verdict      = "WAIT"
        action       = f"WAIT — Mixed IV signals. Straddle {straddle_momentum:.1f}%."
        action_color = Fore.WHITE

    return {
        "verdict":      verdict,
        "action":       action,
        "action_color": action_color,
        "excess_pct":   round(excess_pct, 2),
        "expected_pct": round(expected_pct, 2),
    }


# =============================================================================
# MASTER CONTEXT — single call per tick
# =============================================================================

def compute_buyer_theta_context(df, atm, spot, straddle, expiry,
                                 momentum_data, days_to_expiry,
                                 spot_history=None, lot_size=75,
                                 active_trade=None):
    """
    Single entry point. Call once per tick after compute_strike_thetas().

    Returns ThetaBuyerContext dict with:
        theta_rs_per_day     — daily ₹ decay on ATM straddle
        theta_pct            — as % of straddle premium
        theta_per_5m_rs      — ₹ per 5min tick
        iv_filter            — compute_iv_entry_filter() result
        race                 — compute_race() result (None if no active trade)
        breakeven            — compute_breakeven_move() result (None if no active trade)
        dashboard_lines      — list of (color, text) ready to print
        log_fields           — flat dict to merge into signals_log row
    """
    from gamma_engine import compute_atm_theta_metrics

    # ── Theta metrics ────────────────────────────────────────────────────────
    theta_m = compute_atm_theta_metrics(
        df=df, atm=atm, spot=spot,
        straddle_price=straddle, expiry=expiry,
    )
    theta_m["straddle_price"] = straddle   # needed by race / filter

    theta_rs   = theta_m["atm_theta_rs"]       # ₹/day straddle
    theta_pct  = theta_m["theta_pct"]          # %/day
    t_per_5m   = theta_m["theta_per_5m"]       # ₹/5min

    # ── IV entry filter ──────────────────────────────────────────────────────
    mom_5m = momentum_data["momentum_5m"] if momentum_data else None
    iv_filter = compute_iv_entry_filter(
        straddle_momentum=mom_5m,
        theta_per_5m_rs=t_per_5m,
        straddle_price=straddle,
        days_to_expiry=days_to_expiry,
    )

    # ── Race + breakeven (only meaningful in active trade) ───────────────────
    race      = None
    breakeven = None

    if active_trade is not None:
        race = compute_race(
            spot_history=spot_history or [],
            theta_rs_per_day=theta_rs,
            straddle_price=straddle,
            momentum_data=momentum_data,
            lot_size=lot_size,
        )
        entry_price    = active_trade.get("entry_price", straddle / 2)
        entry_time_str = active_trade.get("entry_time", "")
        holding_min    = _minutes_held(entry_time_str)

        breakeven = compute_breakeven_move(
            entry_price=entry_price,
            spot=spot,
            theta_rs_per_day=theta_rs,
            holding_minutes=max(holding_min, 5),
            lot_size=lot_size,
        )

    # ── Build dashboard lines ────────────────────────────────────────────────
    lines = _build_dashboard_lines(
        theta_pct=theta_pct,
        theta_rs=theta_rs,
        t_per_5m=t_per_5m,
        days_to_expiry=days_to_expiry,
        iv_filter=iv_filter,
        race=race,
        breakeven=breakeven,
        active_trade=active_trade,
        atm_iv=theta_m.get("atm_iv", 0),
    )

    # ── Log fields ───────────────────────────────────────────────────────────
    log_fields = {
        "theta_rs_per_day":    theta_rs,
        "theta_pct":           theta_pct,
        "theta_per_5m_rs":     round(t_per_5m, 3),
        "iv_filter_verdict":   iv_filter["verdict"],
        "iv_excess_pct":       iv_filter["excess_pct"],
        "atm_iv":              theta_m.get("atm_iv", 0),
        "race_verdict":        race["verdict"]    if race else "",
        "race_ratio":          race["race_ratio"] if race else "",
        "breakeven_pts":       breakeven["breakeven_pts"]   if breakeven else "",
        "theta_cost_pct":      breakeven["theta_cost_pct"]  if breakeven else "",
        "pts_needed_per_5m":   breakeven["pts_per_5m_needed"] if breakeven else "",
    }

    return {
        "theta_rs_per_day": theta_rs,
        "theta_pct":        theta_pct,
        "theta_per_5m_rs":  t_per_5m,
        "iv_filter":        iv_filter,
        "race":             race,
        "breakeven":        breakeven,
        "dashboard_lines":  lines,
        "log_fields":       log_fields,
        "atm_iv":           theta_m.get("atm_iv", 0),
        "days_to_expiry":   days_to_expiry,
    }


# =============================================================================
# DASHBOARD FORMATTER
# =============================================================================

def _build_dashboard_lines(theta_pct, theta_rs, t_per_5m, days_to_expiry,
                            iv_filter, race, breakeven, active_trade, atm_iv):
    """
    Returns list of (color, text) tuples.
    Caller does:  print(color + text + Style.RESET_ALL)
    """
    lines = []

    # ── Line 1: Theta cost header ────────────────────────────────────────────
    if theta_pct >= THETA_PCT_RICH:
        col = Fore.RED
        tag = "HIGH DECAY"
    elif theta_pct >= THETA_PCT_MODERATE:
        col = Fore.YELLOW
        tag = "MODERATE DECAY"
    else:
        col = Fore.WHITE
        tag = "LOW DECAY"

    lines.append((col,
        f"Θ {tag} — ₹{theta_rs:.0f}/day ({theta_pct:.1f}% of premium)  "
        f"IV:{atm_iv:.1f}%  {days_to_expiry}DTE  "
        f"[₹{t_per_5m:.1f}/5m bleeding]"
    ))

    # ── Line 2: IV entry filter — the main pre-entry action ──────────────────
    lines.append((
        iv_filter["action_color"],
        f"IV SIGNAL: {iv_filter['action']}"
    ))

    # ── Line 3: Race (only in active trade) ──────────────────────────────────
    if race and race["verdict"] != "NO DATA":
        lines.append((
            race["action_color"],
            f"RACE: {race['action']}"
        ))

    # ── Line 4: Breakeven pace (only in active trade) ────────────────────────
    if breakeven and active_trade:
        at     = active_trade
        held   = _minutes_held(at.get("entry_time", ""))
        be_pts = breakeven["breakeven_pts"]
        pace   = breakeven["pts_per_5m_needed"]
        t_cost = breakeven["theta_cost_pct"]

        # Colour by how achievable the breakeven is
        if pace <= 3:
            be_col = Fore.GREEN
        elif pace <= 6:
            be_col = Fore.YELLOW
        else:
            be_col = Fore.RED

        lines.append((be_col,
            f"BREAKEVEN: Need {be_pts:.0f}pts move  "
            f"@ {pace:.1f}pts/5m pace  "
            f"[Theta cost so far: {t_cost:.1f}% of premium]  "
            f"held {held}min"
        ))

    return lines


def format_dashboard_lines(theta_ctx):
    """
    Ready-to-print string block. Use in print_dashboard().
    Returns a single string with embedded newlines and colour codes.
    """
    out = []
    for color, text in theta_ctx["dashboard_lines"]:
        out.append(color + text + Style.RESET_ALL)
    return "\n".join(out)


# =============================================================================
# TRADE LOG INTEGRATION
# =============================================================================

def log_theta_to_trade(trade_dict, theta_ctx):
    """
    Merges theta fields into a trade dict (for active_trade state or signals log).
    Call at entry and again on each HTL tick so the log has fresh theta state.

    Returns the enriched dict (original is not mutated).
    """
    enriched = dict(trade_dict)
    enriched.update({
        "theta_rs_per_day":  theta_ctx["theta_rs_per_day"],
        "theta_pct":         theta_ctx["theta_pct"],
        "atm_iv_at_log":     theta_ctx["atm_iv"],
        "iv_filter_verdict": theta_ctx["iv_filter"]["verdict"],
        "iv_action":         theta_ctx["iv_filter"]["action"],
    })
    if theta_ctx["race"]:
        enriched["race_verdict"]  = theta_ctx["race"]["verdict"]
        enriched["race_ratio"]    = theta_ctx["race"]["race_ratio"]
        enriched["race_action"]   = theta_ctx["race"]["action"]
    if theta_ctx["breakeven"]:
        enriched["breakeven_pts"]      = theta_ctx["breakeven"]["breakeven_pts"]
        enriched["theta_cost_pct"]     = theta_ctx["breakeven"]["theta_cost_pct"]
        enriched["pts_needed_per_5m"]  = theta_ctx["breakeven"]["pts_per_5m_needed"]
    return enriched


def format_trade_log_line(theta_ctx, active_trade):
    """
    One-line coloured string for the trade log printed alongside HTL verdict.
    Shows what theta is doing TO the open trade right now.

    Use in print_dashboard() immediately after HTL output.
    """
    if active_trade is None:
        return ""

    race = theta_ctx.get("race")
    be   = theta_ctx.get("breakeven")

    if race is None:
        return ""

    entry  = active_trade.get("entry_price", 0)
    strike = active_trade.get("strike", "")
    otype  = active_trade.get("option_type", "")
    held   = _minutes_held(active_trade.get("entry_time", ""))

    be_str = ""
    if be:
        be_str = (f"  │  Need {be['breakeven_pts']:.0f}pts "
                  f"@ {be['pts_per_5m_needed']:.1f}pts/5m  "
                  f"Θ-cost:{be['theta_cost_pct']:.1f}%")

    line = (
        f"Θ-RACE [{race['verdict']}]  "
        f"Ratio:{race['race_ratio']:.2f}  "
        f"Move:{race['momentum_pts_per_5m']:.0f}pts/5m  "
        f"Need:{race['theta_pts_per_5m']:.1f}pts/5m"
        f"{be_str}  "
        f"│  {strike}{otype} @₹{entry:.0f}  held:{held}min"
    )

    return race["action_color"] + line + Style.RESET_ALL


# =============================================================================
# SIGNALS LOG EXTENSION — add theta columns to save_signals()
# =============================================================================

THETA_BUYER_CSV_HEADERS = [
    "theta_rs_per_day", "theta_pct", "theta_per_5m_rs",
    "iv_filter_verdict", "iv_excess_pct", "atm_iv",
    "race_verdict", "race_ratio",
    "breakeven_pts", "theta_cost_pct", "pts_needed_per_5m",
]


def theta_log_row(theta_ctx):
    """
    Returns ordered list matching THETA_BUYER_CSV_HEADERS.
    Append to writer.writerow() after the existing signals fields.
    """
    lf = theta_ctx["log_fields"]
    return [lf.get(h, "") for h in THETA_BUYER_CSV_HEADERS]


# =============================================================================
# HELPERS
# =============================================================================

def _minutes_held(entry_time_str):
    """Parse 'HH:MM' entry time string → minutes held since entry."""
    if not entry_time_str:
        return 0
    try:
        now  = datetime.now()
        et   = datetime.strptime(entry_time_str, "%H:%M").replace(
                   year=now.year, month=now.month, day=now.day)
        diff = (now - et).total_seconds() / 60.0
        return max(int(diff), 1)
    except Exception:
        return 0
