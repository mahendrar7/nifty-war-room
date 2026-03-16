# =============================================================================
# NIFTY ML ENGINE — XGBoost 15-minute direction predictor
# =============================================================================
# Predicts whether NIFTY will move X points in either direction
# over the next two 15-minute candles.
#
# Target classes:
#   1  = bullish breakout  (high of next 2 candles - current close >= X)
#  -1  = bearish breakout  (current close - low of next 2 candles >= X)
#   0  = no significant move
#
# Usage:
#   engine = MLEngine()
#   engine.build_dataset("data/options_log_1min.csv")   # can pass multiple files
#   engine.train()
#   signal = engine.predict_latest()
# =============================================================================

import os
import glob
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# CONFIG
# =============================================================================
MODEL_FILE      = "data/ml_model.joblib"
MODEL_XGB_FILE  = "data/ml_model.ubj"
DATASET_FILE    = "data/ml_dataset.csv"
THRESHOLD_FILE  = "data/ml_threshold.txt"

CANDLE_MINUTES  = 15          # resample resolution
FORWARD_CANDLES = 2           # how many candles ahead to predict
MIN_SAMPLES     = 40          # minimum samples needed to train
PROBA_THRESHOLD = 0.55        # minimum confidence to fire a signal
DEFAULT_X       = 40          # default point threshold — overridden by straddle/2


# =============================================================================
# RESAMPLER — 1-min CSV → 15-min feature rows
# =============================================================================
class Resampler:
    """
    Reads the 1-minute options log CSV(s) and resamples to 15-minute candles,
    applying correct aggregation per feature type (close for levels, sum for flows).
    """

    def __init__(self, candle_minutes=CANDLE_MINUTES):
        self.freq = f"{candle_minutes}min"

    # Columns ML engine requires vs optional war-room additions
    REQUIRED_COLS = ["timestamp","symbol","spot","atm_strike","strike",
                     "option_type","ltp","oi","volume","expiry","days_to_expiry"]
    OPTIONAL_COLS = {"gamma_pressure":0.0,"straddle":0.0,"market_bias":"RANGE",
                     "trap_probability":0,"breakout_cycles":0}

    def load_csv(self, path_or_paths):
        """
        Accept a single path, list of paths, or a glob pattern.
        Handles old (11-col) and new (16-col) CSV schemas gracefully:
        missing optional columns are backfilled with defaults instead
        of skipping the file.
        """
        if isinstance(path_or_paths, str):
            files = glob.glob(path_or_paths)
        else:
            files = list(path_or_paths)

        if not files:
            raise FileNotFoundError(f"No CSV files found: {path_or_paths}")

        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f, parse_dates=["timestamp"], on_bad_lines="skip")

                # Backfill missing optional columns
                for col, default in self.OPTIONAL_COLS.items():
                    if col not in df.columns:
                        df[col] = default

                # Verify required columns exist
                missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
                if missing:
                    print(f"  Skipped {os.path.basename(f)}: missing columns {missing}")
                    continue

                dfs.append(df)
                print(f"  Loaded {os.path.basename(f)} — {len(df):,} rows "                      f"({df['timestamp'].nunique()} timestamps)")

            except Exception as e:
                print(f"  Skipped {os.path.basename(f)}: {e}")

        if not dfs:
            raise ValueError("No usable CSV files found.")

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def resample(self, raw_df):
        """
        Resample the raw 1-minute log into 15-minute candles.
        Returns one row per candle with all engineered features.
        """
        df = raw_df.copy()
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")

        # We need one row per (timestamp, strike, option_type)
        # Separate calls and puts
        calls = df[df["option_type"] == "CE"].copy()
        puts  = df[df["option_type"] == "PE"].copy()

        candles = []

        # Group by 15-minute window
        for ts, group in df.groupby(pd.Grouper(freq=self.freq)):
            if group.empty:
                continue

            row = self._build_candle(ts, group, calls.loc[
                calls.index.floor(self.freq) == ts
            ] if ts in calls.index.floor(self.freq) else pd.DataFrame(),
            puts.loc[
                puts.index.floor(self.freq) == ts
            ] if ts in puts.index.floor(self.freq) else pd.DataFrame())

            if row is not None:
                candles.append(row)

        result = pd.DataFrame(candles)
        if not result.empty:
            result = result.set_index("timestamp").sort_index()
        return result

    def resample_v2(self, raw_df):
        """
        Cleaner resample approach — works directly on the aggregated 1-min data.
        Preferred method.

        Partial candle filtering: a 15-min candle must contain at least
        MIN_MINUTES_IN_CANDLE distinct minute-timestamps to be included.
        Drops first/last candles of each session which are nearly always
        incomplete and produce misleading range/momentum features.
        """
        MIN_MINUTES_IN_CANDLE = 10   # require at least 10 of 15 minutes

        df = raw_df.copy()
        df = df.sort_values("timestamp")
        df["candle"] = df["timestamp"].dt.floor(self.freq)

        rows = []
        skipped = 0

        for candle_ts, group in df.groupby("candle"):
            # Count distinct minute-timestamps in this candle
            n_minutes = group["timestamp"].nunique()
            if n_minutes < MIN_MINUTES_IN_CANDLE:
                skipped += 1
                continue

            row = self._extract_features(candle_ts, group)
            if row is not None:
                # Tag with trading date for day-boundary lag fix
                row["trading_date"] = str(candle_ts.date())
                rows.append(row)

        if skipped:
            print(f"  Dropped {skipped} partial candles (< {MIN_MINUTES_IN_CANDLE} mins of data)")

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.set_index("timestamp").sort_index()
        return result

    def _extract_features(self, ts, group):
        """Extract all features from a 15-minute window of 1-min rows."""
        try:
            # ── Spot: OHLC ────────────────────────────────────────────────────
            spot_series = group.groupby("timestamp")["spot"].first()
            if spot_series.empty:
                return None

            spot_open  = spot_series.iloc[0]
            spot_close = spot_series.iloc[-1]
            spot_high  = spot_series.max()
            spot_low   = spot_series.min()

            # ── Levels — take CLOSE value of window ───────────────────────────
            last = group[group["timestamp"] == group["timestamp"].max()]

            gamma_close      = last["gamma_pressure"].mean()
            straddle_close   = last["straddle"].mean()
            trap_prob_max    = group["trap_probability"].max()       # peak pressure
            breakout_max     = group["breakout_cycles"].max()        # peak count
            bias_close       = last["market_bias"].mode()[0] if not last.empty else "RANGE"
            atm_close        = last["atm_strike"].mean()

            # ── Flows — use first vs last or sum ──────────────────────────────
            first = group[group["timestamp"] == group["timestamp"].min()]

            gamma_open   = first["gamma_pressure"].mean()
            gamma_shift  = gamma_close - gamma_open                  # flow signal

            straddle_open     = first["straddle"].mean()
            straddle_momentum = (straddle_close - straddle_open) / (straddle_open + 1e-9) * 100

            # ── OI: aggregate per strike per option type ───────────────────────
            # Take close OI (last value per strike/type) and open OI (first)
            oi_close = (group.sort_values("timestamp")
                            .groupby(["strike", "option_type"])
                            .last()
                            .reset_index())

            oi_open = (group.sort_values("timestamp")
                           .groupby(["strike", "option_type"])
                           .first()
                           .reset_index())

            calls_close = oi_close[oi_close["option_type"] == "CE"]
            puts_close  = oi_close[oi_close["option_type"] == "PE"]
            calls_open  = oi_open[oi_open["option_type"] == "CE"]
            puts_open   = oi_open[oi_open["option_type"] == "PE"]

            total_call_oi_close = calls_close["oi"].sum()
            total_put_oi_close  = puts_close["oi"].sum()
            total_call_oi_open  = calls_open["oi"].sum()
            total_put_oi_open   = puts_open["oi"].sum()

            # OI flows over the candle (sum = correct for flow signals)
            call_oi_change = total_call_oi_close - total_call_oi_open
            put_oi_change  = total_put_oi_close  - total_put_oi_open

            # PCR at close
            pcr = total_put_oi_close / (total_call_oi_close + 1e-9)

            # Volume — sum over window (it's a flow)
            call_vol = calls_close["volume"].sum() if "volume" in calls_close else 0
            put_vol  = puts_close["volume"].sum()  if "volume" in puts_close  else 0

            # ── Derived distances ──────────────────────────────────────────────
            # OI walls from close snapshot
            if not calls_close.empty and not puts_close.empty:
                call_wall = calls_close.loc[calls_close["oi"].idxmax(), "strike"]
                put_wall  = puts_close.loc[puts_close["oi"].idxmax(), "strike"]
            else:
                call_wall = spot_close
                put_wall  = spot_close

            dist_call_wall = call_wall - spot_close
            dist_put_wall  = spot_close - put_wall
            dist_atm       = spot_close - atm_close

            # ── Time features ─────────────────────────────────────────────────
            minutes_since_open = (ts.hour * 60 + ts.minute) - (9 * 60 + 15)
            days_to_expiry     = group["days_to_expiry"].iloc[-1] if "days_to_expiry" in group else 0

            return {
                "timestamp":          ts,
                # Price
                "spot_open":          spot_open,
                "spot_close":         spot_close,
                "spot_high":          spot_high,
                "spot_low":           spot_low,
                "candle_range":       spot_high - spot_low,
                "candle_body":        spot_close - spot_open,
                # Gamma — levels at close, shift as flow
                "gamma_close":        gamma_close,
                "gamma_shift":        gamma_shift,
                "gamma_positive":     int(gamma_close > 0),
                # Straddle
                "straddle_close":     straddle_close,
                "straddle_momentum":  straddle_momentum,
                # OI levels at close
                "total_call_oi":      total_call_oi_close,
                "total_put_oi":       total_put_oi_close,
                "pcr":                pcr,
                # OI flows (difference over candle)
                "call_oi_change":     call_oi_change,
                "put_oi_change":      put_oi_change,
                # Volume flows
                "call_vol":           call_vol,
                "put_vol":            put_vol,
                "vol_ratio":          call_vol / (put_vol + 1e-9),
                # Distances
                "dist_call_wall":     dist_call_wall,
                "dist_put_wall":      dist_put_wall,
                "dist_atm":           dist_atm,
                # Structural signals (peak/close of window)
                "trap_probability":   trap_prob_max,
                "breakout_cycles":    breakout_max,
                # Bias encoded as integer
                "bias_encoded":       {"BULLISH": 1, "BEARISH": -1, "RANGE": 0}.get(bias_close, 0),
                # Time
                "minutes_since_open": minutes_since_open,
                "days_to_expiry":     days_to_expiry,
            }

        except Exception as e:
            print(f"  Feature extraction failed at {ts}: {e}")
            return None


# =============================================================================
# TARGET BUILDER
# =============================================================================
def build_targets(candles_df, x_points=None):
    """
    Build the target variable for each candle:
      1  = spot moves up >= x_points within next FORWARD_CANDLES candles
     -1  = spot moves down >= x_points
      0  = no significant move

    x_points defaults to half the average straddle (market-implied expected move).
    """
    df = candles_df.copy()

    if x_points is None:
        # Adaptive threshold: use the median candle range (high - low) of
        # the dataset. This is grounded in actual price movement, not premium.
        # Straddle/2 is theoretically correct but in practice far too high
        # intraday — a 480pt straddle does not mean NIFTY moves 240pts per candle.
        median_range = df["candle_range"].median()
        # Cap between 20 and 80 points — practical intraday range for NIFTY
        x_points = float(np.clip(median_range * 0.8, 20, 80))
        straddle_avg = df["straddle_close"].mean()
        print(f"  Auto threshold X = {x_points:.1f} points "              f"(80% of median candle range {median_range:.1f}pts, "              f"straddle avg={straddle_avg:.1f})")

    # Rolling max high and min low over next N candles
    df["future_high"] = (df["spot_high"]
                           .shift(-1)
                           .rolling(FORWARD_CANDLES, min_periods=1)
                           .max())

    df["future_low"]  = (df["spot_low"]
                           .shift(-1)
                           .rolling(FORWARD_CANDLES, min_periods=1)
                           .min())

    df["target"] = np.where(
        df["future_high"] - df["spot_close"] >= x_points,  1,
        np.where(
            df["spot_close"] - df["future_low"] >= x_points, -1,
            0
        )
    )

    # Drop last N rows — no future data available
    df = df.iloc[:-FORWARD_CANDLES]

    return df, x_points


# =============================================================================
# LAG FEATURES — adds previous candle context
# =============================================================================
def add_lag_features(df, lags=3):
    """
    Add lagged versions of key features, respecting day boundaries.

    Lags are computed per trading day so candle[0] of day N never
    inherits state from the last candle of day N-1 — different sessions,
    different market context, completely invalid to mix.
    """
    lag_cols = [
        "gamma_close", "gamma_shift", "straddle_momentum",
        "call_oi_change", "put_oi_change", "pcr",
        "candle_body", "candle_range", "trap_probability"
    ]

    df = df.copy()

    if "trading_date" in df.columns:
        # Process each day separately so lags don't bleed across sessions
        day_frames = []
        for date, day_df in df.groupby("trading_date"):
            for col in lag_cols:
                if col in day_df.columns:
                    for lag in range(1, lags + 1):
                        day_df[f"{col}_lag{lag}"] = day_df[col].shift(lag)
            day_frames.append(day_df)
        df = pd.concat(day_frames).sort_index()
    else:
        # Fallback: no date column, compute globally (may bleed across days)
        for col in lag_cols:
            if col in df.columns:
                for lag in range(1, lags + 1):
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Drop rows where any lag is NaN (first N candles of each day)
    df = df.dropna()
    return df


# =============================================================================
# ML ENGINE
# =============================================================================
class MLEngine:

    def __init__(self):
        self.model      = None
        self.features   = None
        self.x_points   = DEFAULT_X
        self.dataset    = None

    # ── Build dataset from CSV(s) ─────────────────────────────────────────────
    def build_dataset(self, path_or_paths="data/options_log_1min*.csv"):
        print("\n📊 Building dataset...")

        resampler = Resampler()

        print("  Loading raw data...")
        raw = resampler.load_csv(path_or_paths)
        print(f"  Total raw rows: {len(raw):,}")

        print("  Resampling to 15-minute candles...")
        candles = resampler.resample_v2(raw)
        print(f"  Candles produced: {len(candles)}")

        if len(candles) < MIN_SAMPLES // 2:
            print(f"  ⚠ Only {len(candles)} candles — model will be weak. Keep accumulating data.")

        print("  Building targets...")
        candles, self.x_points = build_targets(candles)
        print(f"  Target distribution:\n{candles['target'].value_counts().to_string()}")

        print("  Adding lag features...")
        candles = add_lag_features(candles)

        self.dataset = candles

        # Save for inspection
        os.makedirs("data", exist_ok=True)
        candles.to_csv(DATASET_FILE)
        print(f"  Dataset saved → {DATASET_FILE}")

        # Save threshold for use at prediction time
        with open(THRESHOLD_FILE, "w") as f:
            f.write(str(self.x_points))

        return candles

    # ── Feature columns ───────────────────────────────────────────────────────
    def _get_feature_cols(self, df):
        exclude = {"target", "future_high", "future_low",
                   "spot_open", "spot_high", "spot_low", "trading_date"}
        return [c for c in df.columns if c not in exclude]

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, df=None, feedback_weights=None):
        """
        feedback_weights: optional np.array of per-sample multipliers from
        FeedbackLedger. Applied on top of class-balance weights so that
        historically wrong predictions get extra attention during retraining.
        """
        if df is None:
            df = self.dataset
        if df is None:
            raise ValueError("No dataset. Call build_dataset() first.")

        print("\n🤖 Training XGBoost model...")

        feature_cols = self._get_feature_cols(df)
        self.features = feature_cols

        X = df[feature_cols].values
        y = df["target"].values

        # Map -1, 0, 1 → 0, 1, 2 for XGBoost multiclass
        label_map    = {-1: 0, 0: 1, 1: 2}
        y_mapped     = np.array([label_map[v] for v in y])

        # Handle class imbalance
        sample_weights = compute_sample_weight("balanced", y_mapped)

        # Multiply in feedback weights if provided
        if feedback_weights is not None:
            fw = np.array(feedback_weights)
            if len(fw) == len(sample_weights):
                sample_weights = sample_weights * fw
                print(f"  Feedback weights applied — "
                      f"max={fw.max():.1f}x  mean={fw.mean():.2f}x")
            else:
                print(f"  ⚠ Feedback weight length mismatch ({len(fw)} vs "
                      f"{len(sample_weights)}) — ignoring")

        n_samples = len(X)
        print(f"  Samples: {n_samples} | Features: {len(feature_cols)}")

        if n_samples < MIN_SAMPLES:
            print(f"  ⚠ Only {n_samples} samples — using all for training (no CV)")
            cv_results = None
        else:
            # TimeSeriesSplit — never shuffle time series data
            tscv = TimeSeriesSplit(n_splits=min(5, n_samples // 10))
            cv_results = self._cross_validate(X, y_mapped, sample_weights, tscv)

        # Guard: XGBoost multiclass needs at least 2 distinct classes.
        # With very few samples and a tight threshold all targets can be 0.
        unique_classes = np.unique(y_mapped)
        if len(unique_classes) < 2:
            print(f"  ⚠ Only one class present in targets {unique_classes}.")
            print("  Threshold may be too high — lowering X by 20% and retrying.")
            # Retry with a lower threshold
            lower_x = self.x_points * 0.8
            self.dataset, self.x_points = build_targets(
                self.dataset.drop(columns=["target","future_high","future_low"], errors="ignore"),
                x_points=lower_x
            )
            self.dataset = add_lag_features(self.dataset)
            print(f"  Retrying with X = {self.x_points:.1f} pts")
            dist = self.dataset['target'].value_counts().to_string()
            print(f"  New target distribution:\n{dist}")
            df = self.dataset
            X = df[feature_cols].values
            y = df["target"].values
            y_mapped = np.array([label_map[v] for v in y])
            unique_classes = np.unique(y_mapped)
            sample_weights = compute_sample_weight("balanced", y_mapped)
            if len(unique_classes) < 2:
                print("  ⚠ Still only one class after lowering threshold.")
                print("  Cannot train yet — accumulate more data across varied market sessions.")
                return None

        # Final model trained on all available data
        self.model = self._build_model(n_samples)
        self.model.fit(
            X, y_mapped,
            sample_weight=sample_weights,
            eval_set=[(X, y_mapped)],
            verbose=False
        )

        # Feature importance
        self._print_feature_importance(feature_cols)

        # Save model
        # NEW — save XGBoost natively, metadata separately
        self.model.save_model(MODEL_XGB_FILE)  # native format, no pickle, no warning
        joblib.dump({
            "features": self.features,
            "x_points": self.x_points,
            "label_map": label_map,
            "inv_map": {v: k for k, v in label_map.items()},
            "trained_at": datetime.now().isoformat(),
            "n_samples": n_samples,
        }, MODEL_FILE)

        print(f"  ✅ Model saved → {MODEL_XGB_FILE} + {MODEL_FILE}")
        return cv_results

    def _build_model(self, n_samples):
        # Conservative hyperparameters for small datasets
        # More data → increase n_estimators and max_depth
        return xgb.XGBClassifier(
            objective        = "multi:softprob",
            num_class        = 3,
            n_estimators     = min(200, max(50, n_samples * 2)),
            max_depth        = 3,              # shallow — avoids overfit on small data
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = max(1, n_samples // 20),
            reg_alpha        = 0.1,            # L1 regularisation
            reg_lambda       = 1.0,            # L2 regularisation
            use_label_encoder= False,
            eval_metric      = "mlogloss",
            random_state     = 42,
            n_jobs           = -1,
        )

    def _cross_validate(self, X, y, weights, tscv):
        print("  Running TimeSeriesSplit cross-validation...")
        fold_scores = []

        inv_map = {0: -1, 1: 0, 2: 1}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < 10:
                continue

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr        = weights[train_idx]

            model = self._build_model(len(train_idx))
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

            preds = model.predict(X_val)

            # Precision on non-zero classes (the ones we'd actually trade)
            y_val_orig  = np.array([inv_map[v] for v in y_val])
            preds_orig  = np.array([inv_map[v] for v in preds])

            non_zero_mask = preds_orig != 0
            if non_zero_mask.sum() > 0:
                precision = (y_val_orig[non_zero_mask] == preds_orig[non_zero_mask]).mean()
                fold_scores.append(precision)
                print(f"    Fold {fold+1}: precision on signals = {precision:.2%} "
                      f"({non_zero_mask.sum()} signals fired)")
            else:
                print(f"    Fold {fold+1}: no signals fired")

        if fold_scores:
            print(f"  Mean signal precision: {np.mean(fold_scores):.2%}")
        return fold_scores

    def _print_feature_importance(self, feature_cols, top_n=15):
        importance = self.model.feature_importances_
        pairs = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        print(f"\n  Top {top_n} features by importance:")
        for name, score in pairs[:top_n]:
            bar = "█" * int(score * 200)
            print(f"    {name:<35} {bar} {score:.4f}")

    # ── Load saved model ──────────────────────────────────────────────────────
    def load(self):
        if not os.path.exists(MODEL_XGB_FILE) or not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"No saved model. Train first.")
        # Load metadata (no XGBoost model inside — no pickle warning)
        saved = joblib.load(MODEL_FILE)
        self.features = saved["features"]
        self.x_points = saved["x_points"]
        self._inv_map = saved["inv_map"]
        # Load XGBoost model natively
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_XGB_FILE)
        print(f"  ✅ Model loaded (trained at {saved['trained_at']} on {saved['n_samples']} samples)")

    # ── Predict on latest candle ──────────────────────────────────────────────
    def predict_latest(self, candle_features: dict):
        """
        Pass in a dict of feature values for the current completed 15-min candle.
        Returns: signal (-1, 0, 1), confidence (float), and full probabilities.

        Example:
            features = engine.extract_live_features(df, prev_df)
            signal, confidence, probs = engine.predict_latest(features)
        """
        if self.model is None:
            self.load()

        row = pd.DataFrame([candle_features])

        # Align to training features — fill missing with 0
        for col in self.features:
            if col not in row.columns:
                row[col] = 0.0

        row = row[self.features]

        proba  = self.model.predict_proba(row)[0]   # [p_bearish, p_no_move, p_bullish]
        pred   = np.argmax(proba)
        inv    = {0: -1, 1: 0, 2: 1}
        signal = inv[pred]

        # Confidence = probability of the predicted class
        confidence = proba[pred]

        # Only fire if confidence exceeds threshold
        if confidence < PROBA_THRESHOLD:
            signal = 0

        label = {1: "🟢 BULLISH BREAKOUT", -1: "🔴 BEARISH BREAKOUT", 0: "⚪ NO MOVE"}[signal]

        return {
            "signal":       signal,
            "label":        label,
            "confidence":   round(confidence, 3),
            "p_bearish":    round(proba[0], 3),
            "p_no_move":    round(proba[1], 3),
            "p_bullish":    round(proba[2], 3),
            "x_points":     self.x_points,
            "threshold":    PROBA_THRESHOLD,
        }

    # ── Extract live features from current war room state ─────────────────────
    def extract_live_features(self, current_df, prev_df,
                               spot, atm, gamma, straddle,
                               trap_prob, breakout_cycles,
                               bias, minutes_since_open, days_to_expiry,
                               prev_gamma=None, prev_straddle=None):
        """
        Convenience method — pass your war room computed values directly
        and get back a feature dict ready for predict_latest().
        """
        from ml_engine import Resampler  # avoid circular if used standalone

        calls = current_df.copy()
        puts  = current_df.copy()

        total_call_oi = current_df["call_oi"].sum()
        total_put_oi  = current_df["put_oi"].sum()
        call_vol      = current_df["call_vol"].sum()
        put_vol       = current_df["put_vol"].sum()
        pcr           = total_put_oi / (total_call_oi + 1e-9)

        call_wall = current_df.loc[current_df["call_oi"].idxmax(), "strike"]
        put_wall  = current_df.loc[current_df["put_oi"].idxmax(),  "strike"]

        prev_call_oi = prev_df["call_oi"].sum() if prev_df is not None else total_call_oi
        prev_put_oi  = prev_df["put_oi"].sum()  if prev_df is not None else total_put_oi

        gamma_shift       = gamma - (prev_gamma or gamma)
        straddle_momentum = ((straddle - (prev_straddle or straddle))
                             / ((prev_straddle or straddle) + 1e-9) * 100)

        bias_map = {"BULLISH": 1, "BEARISH": -1, "RANGE": 0}

        return {
            "spot_close":         spot,
            "candle_range":       0,      # not available at candle close — set 0
            "candle_body":        0,
            "gamma_close":        gamma,
            "gamma_shift":        gamma_shift,
            "gamma_positive":     int(gamma > 0),
            "straddle_close":     straddle,
            "straddle_momentum":  straddle_momentum,
            "total_call_oi":      total_call_oi,
            "total_put_oi":       total_put_oi,
            "pcr":                pcr,
            "call_oi_change":     total_call_oi - prev_call_oi,
            "put_oi_change":      total_put_oi  - prev_put_oi,
            "call_vol":           call_vol,
            "put_vol":            put_vol,
            "vol_ratio":          call_vol / (put_vol + 1e-9),
            "dist_call_wall":     call_wall - spot,
            "dist_put_wall":      spot - put_wall,
            "dist_atm":           spot - atm,
            "trap_probability":   trap_prob,
            "breakout_cycles":    breakout_cycles,
            "bias_encoded":       bias_map.get(bias, 0),
            "minutes_since_open": minutes_since_open,
            "days_to_expiry":     days_to_expiry,
        }

    # ── Rolling retrain — call this once per day before market open ───────────
    def rolling_retrain(self, lookback_days=30):
        """
        Retrain on the last N days of archived logs.
        Archived files follow the pattern: data/options_log_1min_DDMMYYYY.csv
        Also includes today's live file if it exists.
        """
        print(f"\n🔄 Rolling retrain on last {lookback_days} days...")

        archived = sorted(glob.glob("data/options_log_1min_????????.csv"))
        recent   = archived[-lookback_days:]

        # Include today's live file if it exists
        if os.path.exists("data/options_log_1min.csv"):
            recent.append("data/options_log_1min.csv")

        if not recent:
            print("  No historical files found. Skipping retrain.")
            return

        print(f"  Using {len(recent)} files: {[os.path.basename(f) for f in recent]}")
        self.build_dataset(recent)

        # Apply feedback weights if ledger exists
        feedback_file = FEEDBACK_FILE
        if os.path.exists(feedback_file) and self.dataset is not None:
            try:
                ledger = FeedbackLedger(feedback_file)
                ts     = self.dataset.index.tolist()
                dates  = self.dataset.get("trading_date", pd.Series([""] * len(ts))).tolist()
                fb_weights = ledger.get_sample_weights(ts, dates)
                print(f"  📊 Feedback weights applied: "
                      f"nice={NICE_WEIGHT}x  naughty={NAUGHTY_WEIGHT}x  "
                      f"({(fb_weights > NEUTRAL_WEIGHT).sum()} weighted samples)")
                ledger.print_report()
                self.train(feedback_weights=fb_weights)
            except Exception as e:
                print(f"  ⚠ Feedback weighting failed ({e}) — training without weights")
                self.train()
        else:
            self.train()



# =============================================================================
# FEEDBACK LEDGER — tracks prediction outcomes, rewards nice, punishes naughty
# =============================================================================
FEEDBACK_FILE = "data/ml_feedback.csv"

# Sample weight multipliers applied during retraining
NICE_WEIGHT    = 3.0   # correct prediction — reinforce this pattern
NAUGHTY_WEIGHT = 4.0   # wrong prediction — learn hard from mistakes
NEUTRAL_WEIGHT = 1.0   # no-signal candles — normal weight


class FeedbackLedger:
    """
    Records every ML prediction and checks outcomes 2 candles (30 min) later.
    Feeds outcome-weighted sample weights back into retraining so the model
    learns more from its mistakes and reinforces its correct calls.

    Lifecycle per prediction:
        1. record_prediction()  — called at candle boundary when signal fires
        2. check_outcomes()     — called every minute; resolves any pending
                                  predictions whose 30-min window has closed
        3. get_sample_weights() — called during retrain; returns per-candle
                                  weight multipliers based on past outcomes
    """

    def __init__(self, path=FEEDBACK_FILE):
        self.path    = path
        self.pending = {}   # candle_ts → {signal, spot_at_signal, x_points, features}
        self._load()

    def _load(self):
        """Load existing feedback ledger from disk."""
        if os.path.exists(self.path):
            try:
                self._df = pd.read_csv(self.path, parse_dates=["candle_ts", "resolved_at"])
                print(f"  📖 Feedback ledger loaded: {len(self._df)} past outcomes")
            except Exception:
                self._df = self._empty_df()
        else:
            self._df = self._empty_df()

    def _empty_df(self):
        return pd.DataFrame(columns=[
            "candle_ts", "signal", "confidence", "spot_at_signal",
            "x_points", "outcome",   # outcome: correct / wrong / no_signal
            "actual_move", "verdict", "resolved_at",
            "trading_date"
        ])

    def _save(self):
        os.makedirs("data", exist_ok=True)
        self._df.to_csv(self.path, index=False)

    def record_prediction(self, candle_ts, result, spot):
        """
        Store a prediction immediately when it fires.
        Only records non-zero signals (no-signal candles aren't worth tracking).
        """
        if result["signal"] == 0:
            return

        self.pending[candle_ts] = {
            "signal":         result["signal"],
            "confidence":     result["confidence"],
            "spot_at_signal": spot,
            "x_points":       result["x_points"],
            "candle_ts":      candle_ts,
        }

    def check_outcomes(self, current_spot, current_time):
        """
        Called every minute. Resolves any pending predictions whose
        30-minute window (2 × 15-min candles) has elapsed.
        Returns list of newly resolved verdicts for dashboard display.
        """
        resolved = []
        to_remove = []

        for candle_ts, pred in self.pending.items():
            elapsed = (current_time - candle_ts).total_seconds() / 60

            # Wait for full 30-minute window to close
            if elapsed < 30:
                continue

            signal         = pred["signal"]
            spot_at_signal = pred["spot_at_signal"]
            x_points       = pred["x_points"]
            actual_move    = current_spot - spot_at_signal

            # Evaluate outcome
            if signal == 1:
                # Bullish call — did spot rise >= X?
                correct = actual_move >= x_points
            else:
                # Bearish call — did spot fall >= X?
                correct = actual_move <= -x_points

            outcome = "correct" if correct else "wrong"
            verdict = "🎉 NICE" if correct else "👿 NAUGHTY"

            row = {
                "candle_ts":      candle_ts,
                "signal":         signal,
                "confidence":     pred["confidence"],
                "spot_at_signal": spot_at_signal,
                "x_points":       x_points,
                "outcome":        outcome,
                "actual_move":    round(actual_move, 2),
                "verdict":        verdict,
                "resolved_at":    current_time,
                "trading_date":   candle_ts.date(),
            }

            self._df = pd.concat(
                [self._df, pd.DataFrame([row])], ignore_index=True
            )
            self._save()
            resolved.append(row)
            to_remove.append(candle_ts)

        for ts in to_remove:
            del self.pending[ts]

        return resolved

    def get_sample_weights(self, candle_timestamps, trading_dates):
        """
        For each candle in the training dataset, return a weight multiplier
        based on whether predictions made around that time were right or wrong.

        Logic:
        - Candles where the model was previously CORRECT  → weight × NICE_WEIGHT
          (reinforce — this situation produced a good signal)
        - Candles where the model was previously WRONG    → weight × NAUGHTY_WEIGHT
          (penalise — the model needs to learn harder from these)
        - Candles with no prediction history              → weight × NEUTRAL_WEIGHT

        The result: retraining naturally upweights the situations where the
        model has been confidently wrong — exactly the patterns it needs to fix.
        """
        if self._df.empty:
            return np.ones(len(candle_timestamps))

        weights = np.full(len(candle_timestamps), NEUTRAL_WEIGHT)

        for i, (ts, date) in enumerate(zip(candle_timestamps, trading_dates)):
            # Match feedback rows from the same trading date and nearby time
            same_day = self._df[self._df["trading_date"].astype(str) == str(date)]
            if same_day.empty:
                continue

            same_day = same_day.copy()
            same_day["candle_ts"] = pd.to_datetime(same_day["candle_ts"])

            # Find predictions within ±15 minutes of this candle
            nearby = same_day[
                (same_day["candle_ts"] >= ts - pd.Timedelta(minutes=15)) &
                (same_day["candle_ts"] <= ts + pd.Timedelta(minutes=15))
            ]

            if nearby.empty:
                continue

            correct_count = (nearby["outcome"] == "correct").sum()
            wrong_count   = (nearby["outcome"] == "wrong").sum()

            if wrong_count > correct_count:
                weights[i] = NAUGHTY_WEIGHT
            elif correct_count > wrong_count:
                weights[i] = NICE_WEIGHT
            # else tied → neutral

        return weights

    def print_report(self):
        """Print a summary of prediction performance."""
        if self._df.empty:
            print("  No feedback data yet.")
            return

        total   = len(self._df)
        correct = (self._df["outcome"] == "correct").sum()
        wrong   = (self._df["outcome"] == "wrong").sum()
        acc     = correct / total * 100 if total > 0 else 0

        print(f"\n📊 ML FEEDBACK REPORT")
        print(f"  Total predictions tracked : {total}")
        print(f"  Correct (🎉 NICE)         : {correct}  ({acc:.1f}%)")
        print(f"  Wrong   (👿 NAUGHTY)      : {wrong}   ({100-acc:.1f}%)")

        # By signal direction
        for sig, label in [(1, "BULLISH"), (-1, "BEARISH")]:
            subset = self._df[self._df["signal"] == sig]
            if not subset.empty:
                sig_acc = (subset["outcome"] == "correct").mean() * 100
                print(f"  {label} precision           : {sig_acc:.1f}%  ({len(subset)} calls)")

        # Recent form — last 10 predictions
        recent = self._df.tail(10)
        if len(recent) >= 3:
            recent_acc = (recent["outcome"] == "correct").mean() * 100
            streak = self._current_streak()
            print(f"  Last 10 accuracy          : {recent_acc:.1f}%")
            print(f"  Current streak            : {streak}")

        # Dynamic threshold suggestion
        if total >= 10:
            suggested = self._suggest_threshold()
            print(f"  Suggested confidence floor: {suggested:.0%}  (current: {PROBA_THRESHOLD:.0%})")

    def _current_streak(self):
        if self._df.empty:
            return "N/A"
        outcomes = self._df["outcome"].tolist()
        last     = outcomes[-1]
        count    = 0
        for o in reversed(outcomes):
            if o == last:
                count += 1
            else:
                break
        emoji = "🎉" if last == "correct" else "👿"
        return f"{emoji} {count} in a row"

    def _suggest_threshold(self):
        """
        Suggest a confidence threshold based on recent accuracy.
        If accuracy < 50%: raise threshold (be more selective).
        If accuracy > 65%: lower threshold slightly (capture more signals).
        """
        recent_acc = (self._df.tail(20)["outcome"] == "correct").mean()
        if recent_acc < 0.45:
            return min(PROBA_THRESHOLD + 0.10, 0.80)
        elif recent_acc < 0.55:
            return min(PROBA_THRESHOLD + 0.05, 0.75)
        elif recent_acc > 0.65:
            return max(PROBA_THRESHOLD - 0.05, 0.50)
        return PROBA_THRESHOLD

    @property
    def dynamic_threshold(self):
        """Live-adjusted confidence threshold based on recent form."""
        if len(self._df) < 10:
            return PROBA_THRESHOLD   # not enough data to adjust yet
        return self._suggest_threshold()

# =============================================================================
# WAR ROOM INTEGRATION — drop-in hook for nifty_war_room.py
# =============================================================================
class MLSignal:
    """
    Lightweight wrapper that manages the 15-minute candle accumulation
    and fires predictions at candle boundaries.

    Add to run_logger():
        ml = MLSignal()
        ...
        ml_result = ml.on_tick(df, spot, atm, gamma, straddle,
                               trap_prob, counter, bias,
                               minutes_since_open, days_to_expiry,
                               prev_gamma)
        if ml_result:
            print(ml_result)
    """

    def __init__(self, candle_minutes=CANDLE_MINUTES):
        self.engine         = MLEngine()
        self.candle_minutes = candle_minutes
        self.current_candle = None
        self.prev_snapshot  = None
        self.prev_gamma     = None
        self.prev_straddle  = None
        self.ledger         = FeedbackLedger()   # outcome tracker

        # Try loading existing model
        try:
            self.engine.load()
            self.ready = True
        except FileNotFoundError:
            print("⚠ No ML model found. Train with MLEngine().rolling_retrain() first.")
            self.ready = False

    def on_tick(self, df, spot, atm, gamma, straddle,
                trap_prob, breakout_cycles, bias,
                minutes_since_open, days_to_expiry):
        """
        Call this every minute from run_logger().
        Returns a prediction dict at 15-minute boundaries, None otherwise.
        """
        if not self.ready:
            return None

        now           = datetime.now()
        candle_bucket = now.replace(
            minute=(now.minute // self.candle_minutes) * self.candle_minutes,
            second=0, microsecond=0
        )

        # Detect candle boundary
        if self.current_candle != candle_bucket:
            self.current_candle = candle_bucket

            # Fire prediction at candle close
            features = self.engine.extract_live_features(
                current_df        = df,
                prev_df           = self.prev_snapshot,
                spot              = spot,
                atm               = atm,
                gamma             = gamma,
                straddle          = straddle,
                trap_prob         = trap_prob,
                breakout_cycles   = breakout_cycles,
                bias              = bias,
                minutes_since_open= minutes_since_open,
                days_to_expiry    = days_to_expiry,
                prev_gamma        = self.prev_gamma,
                prev_straddle     = self.prev_straddle,
            )

            # Use dynamic threshold — adjusts based on recent form
            self.engine.x_points = self.engine.x_points   # keep threshold stable
            result = self.engine.predict_latest(features)

            # Override with dynamic threshold from feedback
            dyn_threshold = self.ledger.dynamic_threshold
            if result["confidence"] < dyn_threshold and result["signal"] != 0:
                result["signal"]    = 0
                result["label"]     = "⚪ NO MOVE"
                result["threshold"] = dyn_threshold

            # Record prediction for outcome tracking
            self.ledger.record_prediction(candle_bucket, result, spot)

            # Update state
            self.prev_snapshot = df.copy()
            self.prev_gamma    = gamma
            self.prev_straddle = straddle

            return result

        # Every tick — check if any pending predictions have resolved
        resolved = self.ledger.check_outcomes(spot, datetime.now())
        if resolved:
            return {"resolved": resolved}   # signal to war room to print verdicts

        return None


# =============================================================================
# CLI — train and inspect from command line
# =============================================================================
if __name__ == "__main__":
    import sys

    engine = MLEngine()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "retrain":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        engine.rolling_retrain(lookback_days=days)

    elif cmd == "feedback":
        # Print feedback report without retraining
        ledger = FeedbackLedger()
        ledger.print_report()

    elif cmd == "suggest":
        # Suggest a new confidence threshold based on recent form
        ledger = FeedbackLedger()
        ledger.print_report()
        print(f"\n  Run with: PROBA_THRESHOLD = {ledger.dynamic_threshold:.2f}")

    else:
        # Default: build from all available logs and train
        print("Building dataset from all available logs...")
        engine.build_dataset("data/options_log_1min*.csv")

        if engine.dataset is not None and len(engine.dataset) >= 5:
            engine.train()
            print("\n✅ Done.")
            print("Commands: retrain [days] | feedback | suggest")
        else:
            print("Not enough data to train yet. Keep logging!")