"""
market_data.py — Kite connect data fetching.
Instruments cache, spot price, expiry, strikes, quotes, dataframe builder.
"""

import os
import pickle
import pandas as pd
from datetime import datetime

from config import CACHE_FILE, STRIKE_STEP, NUM_STRIKES
from kite_interface import get_kite_client

kite = get_kite_client()


def load_instruments(exchange="NFO", cache_file=CACHE_FILE):
    today_str = datetime.now().date().isoformat()

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("date") == today_str:
            return cached["instruments"]

    instruments = kite.instruments(exchange)

    with open(cache_file, "wb") as f:
        pickle.dump({"date": today_str, "instruments": instruments}, f)

    return instruments


def get_spot(spot_symbol="NSE:NIFTY 50"):
    quote = kite.quote(spot_symbol)
    return quote[spot_symbol]["last_price"]


def get_nearest_expiry(instruments, name="NIFTY"):
    today = datetime.now().date()
    expiries = sorted(set(
        i["expiry"]
        for i in instruments
        if i["name"] == name
        and i["instrument_type"] == "CE"
        and i["expiry"] >= today
    ))
    return expiries[0]


def get_strikes(spot, strike_step=STRIKE_STEP):
    atm     = round(spot / strike_step) * strike_step
    strikes = [atm + i * strike_step for i in range(-NUM_STRIKES, NUM_STRIKES + 1)]
    return atm, strikes


def build_symbol_list(instruments, expiry, strikes, name="NIFTY", exchange="NFO"):
    symbols = []
    for ins in instruments:
        if ins["name"] != name:
            continue
        if ins["expiry"] != expiry:
            continue
        if ins["strike"] not in strikes:
            continue
        if ins["instrument_type"] not in ["CE", "PE"]:
            continue
        symbols.append(f"{exchange}:{ins['tradingsymbol']}")
    return symbols


def fetch_quotes(symbols):
    quotes = kite.quote(symbols)
    rows = []
    for sym in quotes:
        q = quotes[sym]
        rows.append({
            "symbol": sym,
            "ltp":    q["last_price"],
            "oi":     q["oi"],
            "volume": q["volume"]
        })
    return rows


def build_option_dataframe(rows, spot):
    data = []
    for r in rows:
        sym      = r["symbol"]
        strike   = int(sym[-7:-2])
        opt_type = sym[-2:]
        data.append({
            "strike": strike,
            "type":   opt_type,
            "ltp":    r["ltp"],
            "oi":     r["oi"],
            "volume": r["volume"]
        })

    df    = pd.DataFrame(data)
    calls = df[df["type"] == "CE"].rename(
        columns={"ltp": "call_ltp", "oi": "call_oi", "volume": "call_vol"}
    )
    puts  = df[df["type"] == "PE"].rename(
        columns={"ltp": "put_ltp", "oi": "put_oi", "volume": "put_vol"}
    )
    return pd.merge(
        calls[["strike", "call_ltp", "call_oi", "call_vol"]],
        puts [["strike", "put_ltp",  "put_oi",  "put_vol"]],
        on="strike"
    )