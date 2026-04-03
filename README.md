# Options War Room

Real-time options market surveillance and trade management system. Combines Black-Scholes gamma analysis, OI signal detection, ML predictions (XGBoost), heavyweight stock momentum tracking, and regime-aware position sizing to surface high-conviction trade suggestions and push Telegram alerts.

Runs live during market hours (9:15-15:30 IST) against Zerodha Kite API data, refreshing every ~1 minute.

---

## Usage

```bash
# Run for Nifty (default)
python options_war_room.py NIFTY

# Run for Sensex
python options_war_room.py SENSEX
```

Both can run simultaneously in separate terminals -- they use separate CSV logs, instrument caches, and ML model files.

**First run** -- opens a browser for Kite OAuth authentication. Token is saved to `access_token.txt` and reused on subsequent runs.

### Trade Logger (browser UI)

```bash
# Local only
python trade_logger.py

# With auth
TRADE_LOGGER_USER=mahendra TRADE_LOGGER_PASS=secret python trade_logger.py
```

Opens at `http://localhost:5050`. Log entries/exits via browser instead of Telegram.

**Remote access via ngrok** -- to use the trade logger from your phone or another machine:

```bash
# Install ngrok (once)
brew install ngrok

# Expose the trade logger
ngrok http 5050
```

ngrok gives you a public URL (e.g. `https://abc123.ngrok-free.app`) that tunnels to your local trade logger. Use this URL on your phone browser during market hours.

### Hotkeys (terminal only, not IDE)

| Hotkey | Action |
|--------|--------|
| `Meta+I` / `Ctrl+T` | Log trade entry from last suggestion |
| `Meta+X` / `Ctrl+X` | Manually exit active trade |
| `Meta+D` / `Ctrl+D` | Toggle debug mode |

---

## Instrument Profiles

Configured in `config.py` under `INSTRUMENT_PROFILES`:

| Parameter | NIFTY | SENSEX |
|-----------|-------|--------|
| Exchange | NFO | BFO |
| Strike step | 50 pts | 100 pts |
| Lot size | 65 | 20 |
| Gamma sigma | 120 | 300 |
| Flip danger zone | 20 pts | 50 pts |

---

## Trade Management

### Sniper (entry)

`sniper_display.py` evaluates whether to enter a trade and in which direction. Aggregates gamma, bias, momentum, OI signals, and ML into a composite score. Outputs STAND DOWN / DEVELOPING / TAKE TRADE / SEND IT.

### Hold The Line (exit)

`detectors.py:hold_the_line()` manages open positions with a multi-signal verdict system:

- **IV momentum** -- is the move still expanding or fading?
- **Gamma regime** -- are dealers amplifying or absorbing?
- **OI flow** -- are smart money writers re-entering against the trade?
- **Wall distance** -- how far to the next OI wall?
- **Heavyweight momentum** -- are top-10 index stocks driving or stalling?
- **Price tracker** -- real-time option LTP monitoring (see below)

Verdict: HOLD / TRAIL / EXIT with a 0-100 hold score.

### Price Tracker

`price_tracker.py` runs a background thread polling the traded option's LTP every 10 seconds. Three-phase stop system fused into HTL:

| Phase | Trigger | Stop |
|-------|---------|------|
| HARD STOP | Entry | Exit if option drops 10% from entry |
| BREAKEVEN | Option up 10% | Stop moved to entry price |
| TRAILING | Gain established | Exit if 40% of gain retraced (tightens to 25% if no new peak in 5 min) |

Hard stop and breakeven stop override HTL. Trailing drawdown feeds into HTL score as deductions.

Config in `config.py`: `PT_HARD_STOP_PCT`, `PT_BREAKEVEN_TRIGGER`, `PT_TRAIL_DRAWDOWN`, `PT_STALE_PEAK_SEC`, `PT_STALE_DRAWDOWN`.

---

## Data Files

Each instrument keeps its own files under `data/`:

| File | Description |
|------|-------------|
| `options_log_1min_{instrument}.csv` | Live intraday log (archived at 15:30) |
| `options_log_1min_{instrument}_DDMMYYYY.csv` | Archived daily logs |
| `signals_log_{instrument}_DDMMYYYY.csv` | Per-minute signal snapshots |
| `snapshot_{instrument}.json` | Latest tick state (consumed by trade logger) |
| `trade_log.csv` | All trade entries/exits with PnL |
| `ml_model_{instrument}.joblib` / `.ubj` | Trained XGBoost model |
| `ml_dataset_{instrument}.csv` | Resampled 15-min training dataset |
| `ml_threshold_{instrument}.txt` | Dynamic prediction threshold |
| `ml_feedback_{instrument}.csv` | Trade outcome feedback ledger |
| `instrument_cache_{instrument}.pkl` | Kite instrument list (24-hr cache) |

---

## Module Structure

| Module | Responsibility |
|--------|---------------|
| `config.py` | All constants, thresholds, instrument profiles |
| `state.py` | Singleton market state, RegimeTracker |
| `market_data.py` | Kite data fetching, instrument cache |
| `gamma_engine.py` | Black-Scholes gamma, IV solver, GEX, flip level |
| `oi_signals.py` | OI walls, velocity, straddle, bias |
| `detectors.py` | Trap, vacuum, wall break, flip breakout, acceleration, HTL, MPM |
| `position_sizing.py` | Regime-to-action mapping, position sizing, strike selection |
| `sniper_display.py` | Entry signal aggregator (STAND DOWN / TAKE TRADE / SEND IT) |
| `price_tracker.py` | Background option LTP monitor, three-phase stop system |
| `ml_engine.py` | XGBoost 15-min direction predictor, rolling retraining |
| `heavyweight_momentum.py` | Top-10 index stock ROC, stall detection |
| `options_war_room.py` | Main loop, dashboard, hotkeys, CSV persistence |
| `trade_logger.py` | Flask UI for trade entry/exit logging |
| `telegram_bot.py` | Telegram command polling (enter/exit/status) |
| `notifier.py` | Telegram alerts |
| `kite_interface.py` | Kite OAuth, token management |

---

## Backlog

1. **Expiry day calibration** -- auto-scale `ACCEL_PRICE_SPEED_MIN` and `HTL_WALL_EXIT_PTS` on 0 DTE
2. **Move probability backtesting** -- log MPM score vs actual next-candle move to calibrate weights
3. **Adaptive position sizing** -- feed trade log win rates back into sizing multipliers
4. **Price tracker breakeven cushion** -- if false exits observed, add 2% buffer below entry price
