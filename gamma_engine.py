"""
gamma_engine.py — Black-Scholes gamma engine.
IV solver, GEX computation, gamma wall, flip level detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm as scipy_norm
from colorama import Fore

from config import RISK_FREE_RATE, GAMMA_SIGMA, LOT_SIZE, GAMMA_FLIP_DANGER_ZONE


# =============================================================================
# BLACK-SCHOLES PRIMITIVES
# =============================================================================

def _bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _bs_gamma(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return scipy_norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)


def _bs_put_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * scipy_norm.cdf(-d2) - S * scipy_norm.cdf(-d1)


def _bs_theta(S, K, T, r, sigma, option_type="CE"):
    """
    Black-Scholes theta — daily ₹ decay per unit held long (negative = loss).
    Uses calendar-day T so result is already ₹/calendar-day.
    """
    if sigma <= 0 or T <= 0:
        return 0.0
    d1   = _bs_d1(S, K, T, r, sigma)
    d2   = d1 - sigma * np.sqrt(T)
    pdf1 = scipy_norm.pdf(d1)
    decay = -(S * pdf1 * sigma) / (2 * np.sqrt(T))
    if option_type == "CE":
        theta = decay - r * K * np.exp(-r * T) * scipy_norm.cdf(d2)
    else:
        theta = decay + r * K * np.exp(-r * T) * scipy_norm.cdf(-d2)
    return theta / 365.0


# =============================================================================
# IMPLIED VOLATILITY SOLVER
# =============================================================================

def implied_vol(S, K, T, r, market_price, option_type="CE",
                tol=None, max_iter=100):
    """
    Newton-Raphson implied volatility solver.
    Returns IV as float, or None if it fails to converge.
    """
    if T <= 0 or market_price <= 0:
        return None
    if tol is None:
        tol = max(market_price * 1e-4, 1e-5)

    intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)
    if market_price < intrinsic - 0.5:
        return None

    sigma    = 0.20
    price_fn = _bs_call_price if option_type == "CE" else _bs_put_price

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        vega  = S * scipy_norm.pdf(_bs_d1(S, K, T, r, sigma)) * np.sqrt(T)

        if vega < 1e-10:
            return None

        diff  = price - market_price
        sigma -= diff / vega

        if sigma <= 0:
            return None
        if abs(diff) < tol:
            return sigma

    return None


# =============================================================================
# STRIKE GAMMA COMPUTATION
# =============================================================================

def compute_strike_gammas(df, spot, expiry, r=RISK_FREE_RATE):
    """
    Compute true BS gamma for every strike in df.
    Returns df with call_iv, put_iv, call_gamma_bs, put_gamma_bs columns.
    Failed solves fall back to median gamma so pipeline never sees NaN.
    """
    today  = datetime.now().date()
    T_days = (expiry - today).days
    T      = max(T_days / 365.0, 1 / 365.0)

    call_ivs, put_ivs       = [], []
    call_gammas, put_gammas = [], []

    for _, row in df.iterrows():
        K    = row["strike"]
        c_iv = implied_vol(spot, K, T, r, row["call_ltp"], "CE")
        call_ivs.append(c_iv)
        call_gammas.append(_bs_gamma(spot, K, T, r, c_iv) if c_iv is not None else None)

        p_iv = implied_vol(spot, K, T, r, row["put_ltp"], "PE")
        put_ivs.append(p_iv)
        put_gammas.append(_bs_gamma(spot, K, T, r, p_iv) if p_iv is not None else None)

    df = df.copy()
    df["call_iv"]       = call_ivs
    df["put_iv"]        = put_ivs
    df["call_gamma_bs"] = call_gammas
    df["put_gamma_bs"]  = put_gammas

    median_gamma = pd.Series(
        [g for g in call_gammas + put_gammas if g is not None]
    ).median()

    if pd.isna(median_gamma) or median_gamma == 0:
        median_gamma = 1e-6

    df["call_gamma_bs"] = df["call_gamma_bs"].fillna(median_gamma)
    df["put_gamma_bs"]  = df["put_gamma_bs"].fillna(median_gamma)
    return df


# =============================================================================
# STRIKE THETA COMPUTATION
# =============================================================================

def compute_strike_thetas(df, spot, expiry, r=RISK_FREE_RATE):
    """
    Compute BS theta for every strike in df.
    Requires call_iv / put_iv columns — run compute_strike_gammas() first.
    Adds call_theta, put_theta, straddle_theta columns (₹/day per unit).
    """
    today  = datetime.now().date()
    T_days = (expiry - today).days
    T      = max(T_days / 365.0, 1 / 365.0)

    call_thetas, put_thetas = [], []

    for _, row in df.iterrows():
        K    = row["strike"]
        c_iv = row.get("call_iv")
        p_iv = row.get("put_iv")
        call_thetas.append(_bs_theta(spot, K, T, r, c_iv, "CE") if c_iv else None)
        put_thetas.append(_bs_theta(spot, K, T, r, p_iv, "PE")  if p_iv else None)

    df = df.copy()
    df["call_theta"] = call_thetas
    df["put_theta"]  = put_thetas

    median_theta = pd.Series(
        [t for t in call_thetas + put_thetas if t is not None]
    ).median()
    if pd.isna(median_theta) or median_theta == 0:
        median_theta = -0.01

    df["call_theta"]     = df["call_theta"].fillna(median_theta)
    df["put_theta"]      = df["put_theta"].fillna(median_theta)
    df["straddle_theta"] = df["call_theta"].abs() + df["put_theta"].abs()
    return df


def compute_atm_theta_metrics(df, atm, spot, straddle_price, expiry,
                               r=RISK_FREE_RATE):
    """
    Pure numbers — no buy/sell decisions (those live in theta_buyer.py).

    Returns:
        atm_theta_rs   — daily ₹ decay of ATM straddle (absolute, positive)
        theta_pct      — theta as % of straddle premium per day
        theta_per_5m   — ₹ decay per 5-minute tick (75 ticks/session)
        atm_iv         — ATM IV in % (avg of call/put)
        iv_rank_proxy  — 0-100 rough rank vs recent straddle history
        T_days         — calendar days to expiry
    """
    today  = datetime.now().date()
    T_days = (expiry - today).days
    T      = max(T_days / 365.0, 1 / 365.0)

    atm_row = df[df["strike"] == atm]
    if atm_row.empty:
        atm_row = df.iloc[(df["strike"] - atm).abs().argsort()[:1]]
    row = atm_row.iloc[0]

    if "call_theta" in df.columns:
        c_theta = row["call_theta"]
        p_theta = row["put_theta"]
    else:
        c_iv = row.get("call_iv") or implied_vol(spot, atm, T, r, row["call_ltp"], "CE")
        p_iv = row.get("put_iv")  or implied_vol(spot, atm, T, r, row["put_ltp"],  "PE")
        c_theta = _bs_theta(spot, atm, T, r, c_iv, "CE") if c_iv else 0.0
        p_theta = _bs_theta(spot, atm, T, r, p_iv, "PE") if p_iv else 0.0

    c_iv_val = row.get("call_iv", 0.0) or 0.0
    p_iv_val = row.get("put_iv",  0.0) or 0.0
    atm_iv   = ((c_iv_val + p_iv_val) / 2.0
                if (c_iv_val and p_iv_val) else max(c_iv_val, p_iv_val))

    atm_theta_rs = abs(c_theta) + abs(p_theta)
    theta_pct    = (atm_theta_rs / straddle_price * 100) if straddle_price > 0 else 0.0
    theta_per_5m = atm_theta_rs / 75.0   # 75 five-minute ticks per 6h15m session

    iv_rank_proxy = None
    try:
        from state import state
        if len(state.straddle_history) >= 10:
            prices    = [p for _, p in state.straddle_history]
            pct_above = sum(1 for p in prices if p > straddle_price) / len(prices)
            iv_rank_proxy = round((1 - pct_above) * 100)
    except Exception:
        pass

    return {
        "atm_theta_rs":  round(atm_theta_rs, 2),
        "theta_pct":     round(theta_pct, 2),
        "theta_per_5m":  round(theta_per_5m, 4),
        "atm_iv":        round(atm_iv * 100, 2),
        "iv_rank_proxy": iv_rank_proxy,
        "T_days":        T_days,
    }


# =============================================================================
# GAUSSIAN WEIGHT + GEX
# =============================================================================

def _gaussian_weight(strikes, spot, sigma=GAMMA_SIGMA):
    return np.exp(-0.5 * ((strikes - spot) / sigma) ** 2)


def compute_gamma_pressure(df, spot, expiry=None, lot_size=LOT_SIZE, sigma=GAMMA_SIGMA):
    """
    Dealer GEX = Σ[(call_gamma - put_gamma) × OI × LotSize × Spot² × GaussWeight]
    Positive → dealers long gamma → pin. Negative → dealers short → trend.
    """
    df = df.copy()
    df["gauss"] = _gaussian_weight(df["strike"].values, spot, sigma=sigma)

    if "call_gamma_bs" in df.columns and "put_gamma_bs" in df.columns:
        df["gex"] = (
            (df["call_gamma_bs"] * df["call_oi"] -
             df["put_gamma_bs"]  * df["put_oi"])
            * lot_size * (spot ** 2) * df["gauss"]
        )
    else:
        df["gex"] = (df["call_oi"] - df["put_oi"]) * df["gauss"]

    return df["gex"].sum()


def compute_gamma_wall(df, spot, expiry=None, lot_size=LOT_SIZE, sigma=GAMMA_SIGMA):
    """Strike with highest total GEX — where dealer hedging is most concentrated."""
    df = df.copy()
    df["gauss"] = _gaussian_weight(df["strike"].values, spot, sigma=sigma)

    if "call_gamma_bs" in df.columns and "put_gamma_bs" in df.columns:
        df["total_gex"] = (
            (df["call_gamma_bs"] * df["call_oi"] +
             df["put_gamma_bs"]  * df["put_oi"])
            * lot_size * (spot ** 2) * df["gauss"]
        )
    else:
        df["total_gex"] = (df["call_oi"] + df["put_oi"]) * df["gauss"]

    return df.loc[df["total_gex"].idxmax(), "strike"]


# =============================================================================
# GAMMA FLIP
# =============================================================================

def detect_gamma_flip(current_gamma, previous_gamma):
    if previous_gamma is None:
        return None
    if previous_gamma > 0 and current_gamma < 0:
        return "NEGATIVE GAMMA FLIP"
    elif previous_gamma < 0 and current_gamma > 0:
        return "POSITIVE GAMMA FLIP"
    return None


def find_gamma_flip_level(df, spot, scan_range=500, num_points=200, sigma=GAMMA_SIGMA):
    """
    Scan ±scan_range pts around spot to find where dealer GEX crosses zero.
    Uses fast OI-proxy — ~1ms per tick. Returns level or None.
    """
    test_prices  = np.linspace(spot - scan_range, spot + scan_range, num_points)
    gamma_values = []

    for p in test_prices:
        weights = _gaussian_weight(df["strike"].values, p, sigma=sigma)
        gex     = ((df["call_oi"] - df["put_oi"]) * weights).sum()
        gamma_values.append(gex)

    for i in range(len(gamma_values) - 1):
        if gamma_values[i] * gamma_values[i + 1] < 0:
            g0, g1 = gamma_values[i], gamma_values[i + 1]
            p0, p1 = test_prices[i], test_prices[i + 1]
            return round(p0 - g0 * (p1 - p0) / (g1 - g0), 1)

    return None


# =============================================================================
# GAMMA SHIFT + SQUEEZE
# =============================================================================

def compute_gamma_shift(current_gamma, previous_gamma):
    if previous_gamma is None or previous_gamma == 0:
        return 0
    return ((current_gamma - previous_gamma) / abs(previous_gamma)) * 100


def classify_gamma_shift(gamma_shift_pct):
    if gamma_shift_pct > 15:
        return Fore.GREEN + "DEALER BUY PRESSURE"
    elif gamma_shift_pct < -15:
        return Fore.RED + "DEALER SELL PRESSURE"
    return "STABLE DEALER POSITIONING"


def detect_gamma_squeeze(gamma, momentum_data, oi_signal):
    if not momentum_data:
        return None

    vol_expanding  = momentum_data["momentum_5m"] > 4
    accelerating   = momentum_data.get("acceleration")
    bullish_unwind = "Call Covering" in oi_signal and "Put Writing" in oi_signal
    bearish_unwind = "Call Writing" in oi_signal and "Put Unwinding" in oi_signal

    if gamma < 0 and vol_expanding:
        if accelerating:
            if bullish_unwind: return "🚀 EXPLOSIVE UPSIDE SQUEEZE"
            if bearish_unwind: return "🚀 EXPLOSIVE DOWNSIDE SQUEEZE"
        else:
            if bullish_unwind: return "UPSIDE SQUEEZE BUILDING"
            if bearish_unwind: return "DOWNSIDE SQUEEZE BUILDING"
    return None