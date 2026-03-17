# =============================================================================
# NIFTY OPTIONS WAR ROOM — BS GAMMA + FLIP LEVEL + POSITION SIZING EDITION
# =============================================================================
#
# CHANGELOG
# ---------
#   - True Black-Scholes gamma replaces OI-proxy gamma
#   - IV computed per strike via Newton-Raphson implied vol solver
#   - compute_gamma_pressure and compute_gamma_wall now use real GEX
#   - Graceful fallback to OI proxy if BS solve fails for a strike
#   - LOT_SIZE = 65, RISK_FREE_RATE = 6.5% added to CONFIG
#   - find_gamma_flip_level(): scans ±500pts to find exact zero-GEX price level
#   - MAP line shows gamma flip level in color (yellow / green danger / red crossed)
#   - One-shot Telegram alert when spot enters flip danger zone (no spam)
#   - Position sizing engine: regime-aware risk, expiry scaling, ML agreement
#   - Expiry-aware sizing: lot caps, stop widening, risk scalar on 0-2 DTE
#   - Capital deployment guard: never > MAX_CAPITAL_PCT in one trade
#   - Strike selection: dynamic OTM distance from gamma, regime, flip level
#   - Gamma Flip Breakout trigger: fires when spot crosses flip + straddle > 2%
#   - OI Velocity SURGE overrides RANGE bias → TREND FOLLOW action
#   - Trap confidence *= 0.4 when gamma < 0 — suppresses false traps in squeezes
#   - Wall Break Vacuum: OI drop >25% + spot crosses strike → 🌪 LIQUIDITY VACUUM
#   - suggest_trade() — four independent trade paths:
#       Path 1: directional bias
#       Path 2: trap fade (works even when bias == RANGE, gamma >= 0 guard)
#       Path 3: gamma flip breakout
#       Path 4: liquidity acceleration
#   - Gamma pinning regime uses gamma sign directly, not flip event
#   - Regime tracker bypass for structural events (no 3-candle delay)
#   - Liquidity acceleration detector: price + IV + OI all firing = launch phase
#   - Hold the Line engine: HOLD / TRAIL / EXIT verdict for running trades
#   - Move probability meter: synthesises all signals into 0-100 score + direction
#   - Meta+T hotkey: log trade entry from last suggestion or manual input
#   - Meta+X hotkey: manually exit active trade
#   - Minimal terminal layout: actionable data only, all verbose reasons → debug
#==============================================================================
# This file is intentionally thin. All domain logic lives in:
#
#   config.py          — all constants and thresholds
#   state.py           — MarketState, RegimeTracker, restore_from_csv
#   market_data.py     — Kite data fetching, instrument cache
#   gamma_engine.py    — BS gamma, IV solver, GEX, flip level
#   oi_signals.py      — walls, velocity, straddle, bias, gravity, magnet
#   detectors.py       — trap, vacuum, wall break, flip breakout,
#                        liq accel, hold the line, move probability
#   position_sizing.py — interpret_market, suggest_trade, sizing engine
#
# This file handles:
#   - print_dashboard() — minimal terminal layout
#   - ML signal display
#   - run_logger() — main loop
#   - save_rows() / archive_daily_log() — CSV persistence
#   - start_hotkey_listener() — Meta+D/T/X
# =============================================================================
#
# KNOWN BUGS / QUICK FIXES NEEDED
# --------------------------------
#   - htl variable referenced in debug block even when state.active_trade is None
#     → fix: initialize htl = None before the active trade check, gate debug
#       block on `if htl is not None`
#   - atm_row_d in debug block re-queries same df redundantly
#     → fix: reuse atm_row computed earlier in the function
#
# =============================================================================
#
# BACKLOG
# -------
#   1. TRADE LOG TO CSV
#      Persist every trade entry/exit to a trade_log.csv with:
#        entry_time, strike, option_type, entry_price, lots, exit_price,
#        exit_time, htl_verdict_at_exit, move_prob_at_entry, trade_type,
#        pnl_pts, pnl_inr
#      After 20-30 trades: backtest win rates by trade_type (directional /
#      trap_fade / flip_breakout / liq_accel) and adjust position sizing
#      multipliers accordingly.
#
#   2. EXPIRY DAY CALIBRATION
#      ACCEL_PRICE_SPEED_MIN = 3.0 is too sensitive on 0 DTE — bid/ask
#      spread noise can trigger it. Auto-scale to 5.0 when days_to_expiry == 0.
#      Similarly HTL_WALL_EXIT_PTS = 25 may be too tight on high-vol days —
#      add a volatility scalar based on straddle size relative to 30-day avg.
#
#   4. MOVE PROBABILITY BACKTESTING
#      Log move_prob score and direction every candle alongside actual next-candle
#      move. After 2-3 weeks: calibrate whether 60% prob actually correlates with
#      60% of candles moving in that direction. Adjust weights in MPM_WEIGHTS if
#      not well-calibrated.
#
#   5. ADAPTIVE POSITION SIZING FROM TRADE LOG
#      Once trade log (item 1) has 30+ trades, compute per-path win rate and
#      average RR. Feed back into compute_position_size() as a multiplier so
#      high-accuracy paths automatically get more size and underperforming paths
#      get throttled without manual tuning.
#
#   6. Add a session_cumulative_move feature — just spot_close - spot_open_of_day 
#      at each candle. Right now the model has no concept of "we've been going up 
#      all day." Every candle looks the same to it whether it's the first move of 
#      the day or the 15th candle in a relentless trend. That single feature would 
#      probably do more for accuracy than any amount of hyperparameter tuning.
#
# =============================================================================
