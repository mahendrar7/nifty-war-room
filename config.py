# =============================================================================
# CONFIG — shared constants across all modules
# Update this file when SEBI lot sizes, RBI rates, or risk params change.
# =============================================================================

from datetime import time as dtime

# -----------------------------
# MARKET
# -----------------------------
CSV_FILE        = "data/options_log_1min.csv"
CACHE_FILE      = "instrument_cache.pkl"
STRIKE_STEP     = 50
NUM_STRIKES     = 10
MARKET_OPEN     = dtime(9, 15)
MARKET_CLOSE    = dtime(15, 30)

LOT_SIZE        = 65       # Nifty lot size — updated per SEBI revision
RISK_FREE_RATE  = 0.065    # RBI repo rate — update when RBI changes (last: 6.5%)
GAMMA_SIGMA     = 120      # Gaussian kernel width for gamma weighting

# -----------------------------
# INSTRUMENT PROFILES
# -----------------------------
INSTRUMENT_PROFILES = {
    "NIFTY": {
        "spot_symbol":          "NSE:NIFTY 50",
        "exchange":             "NFO",
        "name":                 "NIFTY",
        "strike_step":          50,
        "lot_size":             65,
        "gamma_sigma":          120,
        "gamma_flip_danger_zone": 20,
        "csv_file":             "data/options_log_1min_nifty.csv",
        "cache_file":           "instrument_cache_nifty.pkl",
        # Point-based detector thresholds (NIFTY spot ~22-23k)
        "trap_wall_far":        80,     # pts — spot approaching wall
        "trap_wall_near":       40,     # pts — deep in wall zone
        "trap_atm_far":         30,     # pts — near ATM for pin trap
        "trap_atm_near":        15,     # pts — deep ATM pin zone
        "trap_gamma_wall":      50,     # pts — gamma wall proximity
        "breakout_threshold":   30,     # pts — countdown trigger distance
        "strike_war_range":     100,    # pts — search range for strike war
        "vacuum_min_width":     50,     # pts — minimum desert width
        "vacuum_moderate":      100,    # pts — moderate vacuum
        "vacuum_wide":          150,    # pts — wide vacuum
        "accel_speed_min":      1.5,    # pts/candle — acceleration min
        "accel_speed_high":     6.0,    # pts/candle — acceleration high
        "flip_proximity":       25,     # pts — flip breakout proximity
        "htl_wall_exit":        25,     # pts — wall distance for exit
        "htl_wall_trail":       60,     # pts — wall distance for trail
        "magnet_near":          20,     # pts — magnet probability bands
        "magnet_mid":           40,
        "magnet_far":           80,
        "ml_move_min":          20,     # pts — ML target threshold floor
        "ml_move_max":          80,     # pts — ML target threshold cap
        # Sniper trend scoring thresholds
        "sniper_trend_move":    [30, 50, 80],    # low/med/high move magnitude
        "sniper_trend_speed":   [1, 2, 5],       # low/med/high pts/min
        "sniper_trend_roc":     [-1, 1, 3],      # reverse/stall/slow roc1 boundaries
        "trend_regime_override": 50,              # pts to override GAMMA PINNING regime
        "backtest_min_pts":     50,               # min swing size for backtest
        # Sniper gamma/wall scoring thresholds
        "sniper_flip_near":     15,               # pts — right at flip level
        "sniper_flip_far":      40,               # pts — approaching flip
        "sniper_wall_near":     25,               # pts — near a wall
        "sniper_wall_far":      50,               # pts — approaching a wall
        "sniper_wall_touch":    10,               # pts — essentially at the wall
        "sniper_straddle_low":  80,               # straddle floor for expiry penalty
    },
    "SENSEX": {
        "spot_symbol":          "BSE:SENSEX",
        "exchange":             "BFO",
        "name":                 "SENSEX",
        "strike_step":          100,
        "lot_size":             20,
        "gamma_sigma":          300,
        "gamma_flip_danger_zone": 50,
        "csv_file":             "data/options_log_1min_sensex.csv",
        "cache_file":           "instrument_cache_sensex.pkl",
        # Point-based detector thresholds (SENSEX spot ~72-77k, ~3x NIFTY)
        "trap_wall_far":        200,
        "trap_wall_near":       100,
        "trap_atm_far":         75,
        "trap_atm_near":        40,
        "trap_gamma_wall":      125,
        "breakout_threshold":   75,
        "strike_war_range":     250,
        "vacuum_min_width":     125,
        "vacuum_moderate":      250,
        "vacuum_wide":          375,
        "accel_speed_min":      4.5,
        "accel_speed_high":     18.0,
        "flip_proximity":       65,
        "htl_wall_exit":        60,
        "htl_wall_trail":       150,
        "magnet_near":          50,
        "magnet_mid":           100,
        "magnet_far":           200,
        "ml_move_min":          60,
        "ml_move_max":          240,
        # Sniper trend scoring thresholds (~3x NIFTY)
        "sniper_trend_move":    [90, 150, 240],
        "sniper_trend_speed":   [3, 6, 15],
        "sniper_trend_roc":     [-3, 3, 9],
        "trend_regime_override": 150,
        "backtest_min_pts":     150,
        # Sniper gamma/wall scoring thresholds (~3x NIFTY)
        "sniper_flip_near":     45,
        "sniper_flip_far":      120,
        "sniper_wall_near":     75,
        "sniper_wall_far":      150,
        "sniper_wall_touch":    30,
        "sniper_straddle_low":  240,
    },
}

# -----------------------------
# POSITION SIZING
# -----------------------------
ACCOUNT_SIZE     = 100000   # change to your capital
BASE_RISK_PCT    = 0.01     # 1% base risk — adjusted by regime
OPTION_STOP_PCT  = 0.30     # 30% stop on option premium (non-expiry days)
MIN_OTM_PCT      = 0.35     # minimum OTM distance as % of expected move
MAX_OTM_PCT      = 0.55     # maximum OTM distance as % of expected move
MAX_CAPITAL_PCT  = 0.05     # never deploy > 5% of account in one trade

REGIME_RISK = {
    "GAMMA PINNING":         0.005,
    "VOL COMPRESSION":       0.005,
    "MODERATE RANGE":        0.007,
    "MODERATE BULLISH":      0.008,
    "MODERATE BEARISH":      0.008,
    "BREAKOUT PRESSURE":     0.010,
    "STRONG BULLISH":        0.010,
    "STRONG BEARISH":        0.010,
    "VOLATILITY EXPANSION":  0.012,
    "INSTITUTIONAL BULLISH": 0.012,
    "INSTITUTIONAL BEARISH": 0.012,
    "LIQUIDITY VACUUM":      0.015,
    "OPTION TRAP":           0.000,
    "NO EDGE":               0.000,
    "UNCLEAR":               0.000,
    "WEAK STRUCTURE":        0.005,
}

EXPIRY_STOP_PCT = {0: 0.50, 1: 0.40, 2: 0.35}
EXPIRY_LOT_CAP  = {0: 2,    1: 3,    2: 4}
EXPIRY_RISK_SCALAR = {0: 0.50, 1: 0.70, 2: 0.85}

# -----------------------------
# PRICE TRACKER (option LTP monitoring thread)
# -----------------------------
PT_POLL_INTERVAL     = 10     # seconds between LTP polls
PT_HARD_STOP_PCT     = 0.10   # 10% drop from entry → EXIT (phase 1)
PT_BREAKEVEN_TRIGGER = 0.10   # 10% gain from entry → move stop to breakeven (phase 2)
PT_TRAIL_DRAWDOWN    = 0.40   # 40% of peak gain retraced → EXIT (phase 3)
PT_STALE_PEAK_SEC    = 300    # 5 min with no new peak → tighten trail
PT_STALE_DRAWDOWN    = 0.25   # tightened trail: 25% of gain when peak is stale

# -----------------------------
# DETECTOR THRESHOLDS
# -----------------------------
GAMMA_FLIP_DANGER_ZONE  = 20     # pts — fallback (use PROFILE["gamma_flip_danger_zone"])
FLIP_BREAKOUT_PROXIMITY = 25     # pts — fallback (use PROFILE["flip_proximity"])
FLIP_BREAKOUT_IV_MIN    = 0.5    # % straddle momentum needed to confirm breakout
OI_DESERT_THRESHOLD     = 0.10   # strike OI < 10% of chain max = desert
WALL_DISSOLVE_RATE      = 0.30   # wall lost >30% OI = dissolving
VACUUM_MIN_WIDTH        = 50     # pts — fallback (use PROFILE["vacuum_min_width"])
WALL_BREAK_OI_DROP      = 0.25   # 25% OI loss = wall breaking

ACCEL_PRICE_SPEED_MIN   = 1.5    # pts/candle — fallback (use PROFILE["accel_speed_min"])
ACCEL_PRICE_SPEED_HIGH  = 6.0    # pts/candle — fallback (use PROFILE["accel_speed_high"])
ACCEL_IV_MIN            = 0.8    # % straddle momentum minimum
ACCEL_IV_HIGH           = 2.5    # % straddle momentum = high conviction
ACCEL_SCORE_THRESHOLD   = 40     # minimum score to report acceleration

# HTL IV thresholds — per instrument, with expiry-day overrides
# On DTE 0-1 theta crush makes IV mom naturally negative; widen thresholds
HTL_IV_THRESHOLDS = {
    "NIFTY": {
        "hold_min":   0.5,   # above = HOLD
        "trail_min": -1.0,   # soft TRAIL
        "exit_min":  -3.0,   # hard EXIT (P5 = -3.70)
        # Expiry day: median is -2.22, P5 is -13.96
        "expiry_hold_min":  -0.5,
        "expiry_trail_min": -4.0,
        "expiry_exit_min":  -9.0,
    },
    "SENSEX": {
        "hold_min":   0.5,   # above = HOLD
        "trail_min": -2.0,   # soft TRAIL
        "exit_min":  -5.0,   # hard EXIT (P5 = -4.33)
        # Expiry day: median is -1.23, P5 is -9.81
        "expiry_hold_min":  -0.5,
        "expiry_trail_min": -3.5,
        "expiry_exit_min":  -7.0,
    },
}
HTL_IV_HOLD_MIN    =  0.5   # default fallback
HTL_IV_EXIT_MIN    = -3.0   # default fallback
HTL_IV_TRAIL_MIN   = -1.0   # default fallback
HTL_WALL_TRAIL_PTS =  60    # pts — fallback (use PROFILE["htl_wall_trail"])
HTL_WALL_EXIT_PTS  =  25    # pts — fallback (use PROFILE["htl_wall_exit"])
HTL_GAMMA_TRAIL    =  0.0   # gamma at zero = start trailing

# HEAVYWEIGHT MOMENTUM (for HTL)
HW_ROC_WINDOW        = 15    # candles — broad ROC: is there a real move?
HW_ROC_WINDOW_FAST   = 5     # candles — fast ROC: is the move dying? (used in-trade)
HW_STALL_WINDOW      = 2     # candles — very short ROC to detect flatline
HW_STALL_RATIO       = 0.25  # if short ROC < 25% of broad ROC → stalled
HW_STRONG_THRESHOLD  = 0.15  # weighted ROC ≥ 0.15% = strong move
HW_WEAK_THRESHOLD    = 0.05  # weighted ROC ≥ 0.05% = moderate move
HW_EXHAUSTION_DEDUCT = 25    # HTL deduction when heavyweights decelerate
HW_AGAINST_DEDUCT    = 30    # HTL deduction when heavyweights move against trade
HW_STALL_DEDUCT      = 20    # HTL deduction when heavyweights stall after a move

MPM_WEIGHTS = {
    "flip_breakout":   30,
    "liq_accel":       25,
    "vacuum":          20,
    "wall_break":      20,
    "trend":           18,   # price-action trend — fires when OI signals lag
    "iv_expansion":    12,
    "oi_surge":        12,
    "momentum_strike":  6,
}
MPM_GAMMA_AMP    = 1.20
MPM_GAMMA_DAMP   = 0.85
MPM_CONFLICT_PEN = 0.75

# -----------------------------
# TREND DETECTION
# -----------------------------
TREND_WINDOW_MINUTES  = 30     # look back this far to detect persistent trends
TREND_MIN_MOVE_MULT   = 0.3   # must move ≥ 0.3× expected move to qualify (~70pts)
TREND_TRAP_SUPPRESS   = 0.40  # suppress trap confidence to 40% in trends
TREND_BIAS_BOOST      = 30    # add this to directional confidence in trends

# -----------------------------
# SNIPER TUNING
# -----------------------------
CHOP_KILLER_GAMMA_MIN = 5e13   # only apply chop killer when gamma > this (strong pin)
SNIPER_TAKE_TRADE     = 5.0    # score threshold for TAKE TRADE (was 5.5)
SNIPER_SEND_IT        = 7.0    # score threshold for SEND IT (was 8.5)
SNIPER_STALK          = 4.0    # score threshold for STALK (was 5.5)
GAMMA_MOMENTUM_LOOKBACK   = 5     # candles to compute gamma rate-of-change
GAMMA_MOMENTUM_CHOP_EASE  = 0.30  # if gamma dropped >30% over lookback, ease chop killer

# -----------------------------
# WALL RETREAT DETECTION
# -----------------------------
WALL_HISTORY_LENGTH   = 10    # track last N wall positions
WALL_RETREAT_STRIKES  = 2     # wall retreated ≥ 2 strikes = retreating
WALL_RETREAT_TRAP_PEN = 0.50  # halve trap confidence when wall retreating

# -----------------------------
# EXPIRY THETA NORMALISATION
# -----------------------------
EXPIRY_THETA_NORM_DTE = [0, 1]   # on these DTE values, normalise straddle momentum
THETA_DECAY_RATE_PER_5M = -1.5   # expected ~1.5% decay per 5min on expiry day

# -----------------------------
# ML KILL SWITCH
# -----------------------------
ML_CONSECUTIVE_WRONG_KILL = 5     # after N consecutive wrong, suppress to neutral
ML_RECENT_ACCURACY_KILL   = 0.25  # if last 10 accuracy < 25%, suppress

# -----------------------------
# SIGNAL THROTTLE INTERVALS
# These signals are computed every N minutes instead of every tick.
# Signals not listed here are computed every tick (1 minute).
# -----------------------------
THROTTLE_INTERVALS = {
    "trap":           5,    # trap classifier — 5 min
    "vacuum":         5,    # liquidity vacuum — 5 min
    "wall_break":     5,    # wall break vacuum — 5 min
    "move_prob":      3,    # move probability meter — 3 min
    "strike_war":     5,    # strike war detection — 5 min
    "magnet":         5,    # dealer magnet — 5 min
    "best_option":    5,    # best option to buy — 5 min
    "gravity":        3,    # premium gravity — 3 min
    "pcr":            3,    # put/call ratio — 3 min
    "oi_imbalance":   3,    # OI imbalance — 3 min
    "radar":          3,    # anticipatory setup scanner — 3 min
}
# Computed every tick (1 min): spot, gamma, GEX, straddle, walls,
# flip level, OI change, velocity, bias, regime, acceleration,
# flip breakout, gamma squeeze, HTL