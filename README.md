# Options War Room

Real-time options market surveillance and trade suggestion system. Combines Black-Scholes gamma analysis, OI signal detection, ML predictions (XGBoost), and regime-aware position sizing to surface high-conviction trade suggestions and push Telegram alerts.

Runs live during market hours (9:15–15:30 IST) against Zerodha Kite API data, refreshing every ~1 minute.

---

## Usage

```bash
# Run for Nifty (default)
python options_war_room.py NIFTY

# Run for Sensex
python options_war_room.py SENSEX
```

Both can run simultaneously in separate terminals — they use separate CSV logs, instrument caches, and ML model files.

**First run** — opens a browser for Kite OAuth authentication. Token is saved to `access_token.txt` and reused on subsequent runs.

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

## Data Files

Each instrument keeps its own files under `data/`:

| File | Description |
|------|-------------|
| `options_log_1min_{instrument}.csv` | Live intraday log (archived at 15:30) |
| `options_log_1min_{instrument}_DDMMYYYY.csv` | Archived daily logs |
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
| `position_sizing.py` | Regime→action mapping, position sizing, strike selection |
| `ml_engine.py` | XGBoost 15-min direction predictor, rolling retraining |
| `options_war_room.py` | Main loop, dashboard, hotkeys, CSV persistence |
| `notifier.py` | Telegram alerts |
| `kite_interface.py` | Kite OAuth, token management |

---

## Backlog

1. **Trade log to CSV** — persist entry/exit with PnL for win rate analysis by trade type
2. **Expiry day calibration** — auto-scale `ACCEL_PRICE_SPEED_MIN` and `HTL_WALL_EXIT_PTS` on 0 DTE
3. **Move probability backtesting** — log MPM score vs actual next-candle move to calibrate weights
4. **Adaptive position sizing** — feed trade log win rates back into sizing multipliers
5. **Session cumulative move feature** — add `spot_close - spot_open_of_day` as ML feature for trend context
