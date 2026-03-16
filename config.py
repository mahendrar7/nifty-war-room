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
# DETECTOR THRESHOLDS
# -----------------------------
GAMMA_FLIP_DANGER_ZONE  = 20     # pts — danger zone around flip level
FLIP_BREAKOUT_PROXIMITY = 10     # pts — how close to flip before watching for cross
FLIP_BREAKOUT_IV_MIN    = 2.0    # % straddle momentum needed to confirm breakout
OI_DESERT_THRESHOLD     = 0.10   # strike OI < 10% of chain max = desert
WALL_DISSOLVE_RATE      = 0.30   # wall lost >30% OI = dissolving
VACUUM_MIN_WIDTH        = 50     # desert must span ≥ 50pts
WALL_BREAK_OI_DROP      = 0.25   # 25% OI loss = wall breaking

ACCEL_PRICE_SPEED_MIN   = 3.0    # pts per candle minimum for acceleration
ACCEL_PRICE_SPEED_HIGH  = 10.0   # pts per candle = high conviction
ACCEL_IV_MIN            = 2.0    # % straddle momentum minimum
ACCEL_IV_HIGH           = 4.0    # % straddle momentum = high conviction
ACCEL_SCORE_THRESHOLD   = 60     # minimum score to report acceleration

HTL_IV_HOLD_MIN    =  1.0   # straddle momentum floor for HOLD
HTL_IV_EXIT_MIN    = -1.0   # straddle momentum floor — below = EXIT
HTL_WALL_TRAIL_PTS =  60    # pts to next wall — start trailing
HTL_WALL_EXIT_PTS  =  25    # pts to next wall — exit
HTL_GAMMA_TRAIL    =  0.0   # gamma at zero = start trailing

MPM_WEIGHTS = {
    "flip_breakout":   30,
    "liq_accel":       25,
    "vacuum":          20,
    "wall_break":      20,
    "iv_expansion":    12,
    "oi_surge":        12,
    "momentum_strike":  6,
}
MPM_GAMMA_AMP    = 1.20
MPM_GAMMA_DAMP   = 0.85
MPM_CONFLICT_PEN = 0.75