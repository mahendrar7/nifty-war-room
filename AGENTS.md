# AGENTS.md — Nifty War Room Architecture Guide

## Project Overview

**Nifty War Room** is a real-time options market surveillance and trade suggestion system for NIFTY 50 index options. It combines Black-Scholes gamma analysis, OI signal detection, ML predictions, and regime-aware position sizing to identify high-conviction trading opportunities.

**Not a backtest tool** — this runs live during market hours (9:15–15:30 IST) against live Kite data, refreshing every ~1 min.

---

## Big Picture Architecture

### Core Data Flow
```
[Kite API] → [market_data] → [state] ← [all modules update state]
     ↓
[gamma_engine, oi_signals] compute core metrics
     ↓
[detectors] classify patterns (traps, vacuums, accelerations)
     ↓
[position_sizing] interprets regime + suggests trades
     ↓
[nifty_war_room] prints dashboard + Telegram alerts
```

### Module Boundaries

| Module | Responsibility | Key Exports |
|--------|---------------|------------|
| **config.py** | All constants, thresholds, risk params | ~40 tunable parameters |
| **state.py** | Singleton state object, RegimeTracker (3-candle confirmation), alert flags | `state` instance, `MarketState` class |
| **market_data.py** | Kite data fetching, instrument cache (24-hour), spot/expiry/strikes/quotes | `load_instruments()`, `get_spot()`, `build_optionchain_df()` |
| **gamma_engine.py** | BS gamma, IV solver (Newton-Raphson), GEX pressure/walls, flip level scan | `implied_vol()`, `compute_gamma_flip_level()`, GEX functions |
| **oi_signals.py** | OI walls (gamma-weighted 3-strike clusters), velocity, PCR, straddle, bias | `compute_oi_walls()`, `compute_straddle()`, `interpret_bias()` |
| **detectors.py** | 7 signal detectors: trap, vacuum, wall break, flip breakout, liq accel, HTL, MPM | `classify_option_trap()`, `detect_liquidity_acceleration()` + formatters |
| **position_sizing.py** | Regime→action mapping, trade mode, position sizing (expiry-aware, capital guards) | `interpret_market()`, `suggest_trade()`, `compute_position_size()` |
| **ml_engine.py** | XGBoost 15-min direction predictor, trains on 1-min CSV, persistence | `MLEngine` class, `predict_latest()` |
| **notifier.py** | Telegram sender (synchronous) | `send_telegram_message()` |
| **kite_interface.py** | Kite auth (OAuth callback handler on port 8080), token refresh, client singleton | `get_kite_client()` |
| **nifty_war_room.py** | Main loop (1-min ticks), dashboard printer, trade hotkeys, orchestrator | Entry point |

---

## Critical Developer Workflows

### Starting the System
```bash
# First run — generates Kite auth token via browser callback
python nifty_war_room.py

# Subsequent runs — loads token from access_token.txt
python nifty_war_room.py
```

### Data & ML Pipeline
- **Live CSV logging**: `market_data.build_optionchain_df()` writes 1-min rows to `data/options_log_1min_DDMMMYYYY.csv`
- **Model training**: `ml_engine.build_dataset()` resamples 1-min CSVs → 15-min features, trains on `CANDLE_MINUTES=15`, `FORWARD_CANDLES=2`
- **Persistence**: Model saved as `.joblib` + `.ubj` (XGBoost binary)

### Instrument Cache
- Instruments loaded once per day (timestamp check in `load_instruments()`)
- Pickled to `instrument_cache.pkl` — avoids 1000+ Kite API calls on restart
- Expires at midnight; forces fresh fetch if date changes

### Hotkey System (Terminal Mode)
- **Meta+T**: Log trade entry (last suggestion or manual prompt)
- **Meta+X**: Manually exit active trade
- Hotkeys require `pynput` installed; run in terminal (not IDE)

### Debugging Trade Logic
1. Set `DEBUG = True` in `config.py` for verbose terminal output
2. Check `state.last_suggestion` for last computed trade
3. Review `detectors.py` formatters (e.g., `format_trap()`, `format_vacuum()`) to trace why a detector fired/didn't

---

## Project-Specific Conventions & Patterns

### 1. State Management via Singleton
```python
from state import state  # Imported everywhere
state.last_suggestion = {...}  # All modules mutate this
state.gamma_flip_alerted = False  # Alert flag to prevent spam
```
- **Why**: Avoids parameter threading through 10 function calls
- **Pattern**: Update flags after sending Telegram (see `nifty_war_room.py` main loop)

### 2. Regime Confirmation (3-Candle Hysteresis)
```python
RegimeTracker.update(new_bias, new_regime, new_action, new_confidence)
# Only switches confirmed_* after 3 consecutive matching candles
```
- **Why**: Stops whipsaw from 1-2 min noise
- **Usage**: `state.regime_tracker.confirmed_regime` is what you display; candidates held in `candidate_bias`

### 3. Gamma Sign = Causal Engine
- **Positive gamma** (ATM): Price self-corrects → traps have high confidence
- **Negative gamma** (OTM): Price accelerates → trap confidence *= 0.4 (breakout mode)
- **Zero crossing** (flip level): Structural reversal point
- **Pattern**: Never hardcode gamma multipliers; use `config.REGIME_RISK[regime]` instead

### 4. Expiry-Aware Risk Scaling
```python
# config.py
EXPIRY_RISK_SCALAR = {0: 0.50, 1: 0.70, 2: 0.85}
EXPIRY_LOT_CAP = {0: 2, 1: 3, 2: 4}
EXPIRY_STOP_PCT = {0: 0.50, 1: 0.40, 2: 0.35}
# 0 DTE = half risk, tightest stops, smallest positions
```
- **Why**: Expiry dynamics (gamma explosion, vega crush) require different rules
- **Fallback**: Missing DTE keys use defaults (OPTION_STOP_PCT, etc.)

### 5. IV Solver Fallback
```python
# gamma_engine.implied_vol() returns None if Newton-Raphson fails
# Detector code falls back to OI-proxy gamma:
if sigma is None:
    gamma_bs = None  # Use OI-weighted gamma instead
```
- **Why**: Some far-OTM strikes have thin/stale quotes
- **Pattern**: Always check for None; don't assume every strike has valid IV

### 6. Alert De-duplication
```python
state.gamma_flip_alerted = True  # Set ONCE when flip detected
# Next candle, won't re-alert (check before sending Telegram)
if not state.gamma_flip_alerted:
    send_telegram_message(...)
    state.gamma_flip_alerted = True
```
- **Why**: Prevents spam when flip lingers for multiple candles
- **Reset**: `state.reset_session()` clears all flags at market close

### 7. OI Walls via Gamma Weighting
```python
# oi_signals.compute_oi_walls()
df["call_wall_strength"] = df["call_gamma_bs"] * df["call_oi"]
df["call_cluster"] = df["call_wall_strength"].rolling(3, center=True).sum()
call_wall = df.loc[df["call_cluster"].idxmax(), "strike"]
```
- **Why**: OI alone ignores leverage; gamma-weighted OI = true market pressure
- **Fallback**: If BS gamma unavailable, uses raw OI

### 8. Dashboard Minimalism
- Only actionable metrics printed (not debug logs)
- All verbose reasoning sent to debug output or Telegram
- Status line shows: `[TIME] BIAS | REGIME | ACTION | CONFIDENCE%`
- Chart shows: Spot, walls, flip level, straddle IV

---

## Integration Points & External Dependencies

### Kite Connect (Live Data)
- **Endpoint**: `kite.quote()` for spot, `kite.quote(option_chain)` for strikes
- **Rate limit**: ~5–10 per second (market_data.py batches queries)
- **Auth**: OAuth callback on `http://127.0.0.1:8080` (see `kite_interface.py`)
- **Token refresh**: Automatic on 24-hour expiry; saved to `access_token.txt`

### CSV Logging (Feature Engineering)
- **File**: `data/options_log_1min_DDMMMYYYY.csv` (one per day)
- **Columns**: `timestamp, spot, atm_strike, call_iv, put_iv, call_oi, put_oi, straddle, call_wall, put_wall, ...`
- **Usage**: ML resampler reads these for training

### ML Model Lifecycle
- **Train trigger**: Manual call to `ml_engine.build_dataset()` + `ml_engine.train()`
- **Input**: 1-min CSVs (glob pattern matches `data/options_log_1min_*.csv`)
- **Output**: `.joblib` model + XGB binary `.ubj`
- **Prediction**: `engine.predict_latest()` runs on most recent 15-min features

### Telegram Alerts
- **Endpoint**: `https://api.telegram.org/bot{BOT_TOKEN}/sendMessage`
- **Format**: Markdown (bold, italic, code blocks)
- **Async**: Calls are synchronous; long delays block dashboard
- **Error handling**: Graceful; exceptions logged but don't crash main loop

---

## Common Tasks & Quick References

### Adding a New Detector
1. Write detector logic in `detectors.py` (returns dict with `{"name": "...", "confidence": 0–100, ...}`)
2. Add formatter function (e.g., `format_new_detector()`)
3. Call in `nifty_war_room.py` main loop
4. Add to Move Probability Meter (MPM) weights in `config.py` if structural

### Tuning Risk Parameters
- **Global**: `BASE_RISK_PCT` in `config.py`
- **Per-regime**: `REGIME_RISK` dict (covers 16 regimes)
- **Per-expiry**: `EXPIRY_RISK_SCALAR` (0/1/2 DTE)
- **Capital guard**: `MAX_CAPITAL_PCT` (never > 5% per trade)

### Debugging Gamma Flip
1. Check `gamma_engine.compute_gamma_flip_level()` scans ±500 pts
2. Verify IV solver converged (check Newton-Raphson logs if enabled)
3. Confirm `GAMMA_FLIP_DANGER_ZONE` threshold (default 20 pts) in config
4. Test with live data: `python -c "from gamma_engine import compute_gamma_flip_level; ..."`

### Extending the Dashboard
- Print logic in `nifty_war_room.print_dashboard()`
- Data flows from `state` + compute functions
- Use `Fore.YELLOW / RED / GREEN` for colors (colorama)
- Avoid printing more than 40 lines (terminal overflow)

---

## Key Files to Study First

| Learning Order | File | Purpose |
|-------|------|---------|
| 1 | **config.py** | Understand the 40+ tunable parameters & regimes |
| 2 | **state.py** | Grasp singleton state management + RegimeTracker |
| 3 | **nifty_war_room.py** | See how modules orchestrate in 1-min loop |
| 4 | **gamma_engine.py** | Study BS gamma + IV solver + flip level |
| 5 | **oi_signals.py** | Learn wall detection + bias inference |
| 6 | **detectors.py** | See pattern detectors (trap, vacuum, accel, etc.) |
| 7 | **position_sizing.py** | Understand regime→trade mapping + sizing engine |

---

## Running Tests / Manual Checks

### Check Kite Auth
```python
from kite_interface import get_kite_client
kite = get_kite_client()
print(kite.quote("NSE:NIFTY 50"))
```

### Verify ML Model
```python
from ml_engine import MLEngine
engine = MLEngine()
engine.load_model()
signal = engine.predict_latest()
print(signal)
```

### Replay Live CSV
```python
import pandas as pd
df = pd.read_csv("data/options_log_1min_16032026.csv")
# Pass df to detectors for offline analysis
```

### Inspect Gamma Surface
```python
from gamma_engine import compute_gamma_pressure
pressure = compute_gamma_pressure(df, spot=21000, atm=21000)
print(pressure)  # Dict of {strike: GEX}
```

---

## Performance Notes

- **Main loop**: ~800ms per iteration (Kite calls dominate)
- **Gamma computation**: O(strikes) — negligible
- **IV solver**: O(strikes × 100 iterations) — ~200ms for 10 strikes
- **Instrument cache**: Saves 1000+ API calls on restart
- **Alert spam protection**: State flags prevent thundering herd

---

## FAQ for AI Agents

**Q: How do I add a new regime?**
A: Add entry to `REGIME_RISK` dict in `config.py` + update `RegimeTracker` candidate logic in `position_sizing.py`

**Q: Why does the flip level sometimes jump?**
A: IV solver fails on illiquid strikes → switches to OI-proxy gamma → less smooth. Check `gamma_engine.implied_vol()` fallback.

**Q: How are trades stored?**
A: `state.active_trade` holds current trade; `state.last_suggestion` holds last computed trade. No database — in-memory only.

**Q: Can I run this outside market hours?**
A: Yes, but all quotes will be stale. Add a warning or fetch pre-market data (not currently implemented).

**Q: How do I train the ML model?**
A: Call `ml_engine.build_dataset(["data/options_log_1min_*.csv"])` then `ml_engine.train()`. Requires ≥40 samples.




