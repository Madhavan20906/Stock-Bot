"""
Market-wide (macro) features: VIX, S&P500 trend, sector ETF momentum, etc.
Uses yfinance for live/historical data to augment the Kaggle dataset.
"""

import pandas as pd
import numpy as np
from loguru import logger

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# Macro symbols
MACRO_TICKERS = {
    "spy":  "SPY",   # S&P 500
    "vix":  "^VIX",  # Volatility index
    "qqq":  "QQQ",   # Nasdaq 100
    "tlt":  "TLT",   # 20-year Treasury (risk-off signal)
    "gld":  "GLD",   # Gold (flight to safety)
    "usd":  "DX-Y.NYB",  # US Dollar index
}


def fetch_macro_features(start: str, end: str) -> pd.DataFrame:
    """Download macro time series from Yahoo Finance and engineer features.

    Returns:
        DataFrame indexed by date with macro feature columns
    """
    if not YF_AVAILABLE:
        logger.warning("yfinance not installed — returning empty macro features.")
        return pd.DataFrame()

    frames = {}
    for name, symbol in MACRO_TICKERS.items():
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            frames[name] = df["Close"].rename(name)
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1).sort_index()
    macro = macro.ffill().dropna()

    # Engineer features
    result = pd.DataFrame(index=macro.index)
    result["spy_ret_1d"] = macro["spy"].pct_change()
    result["spy_ret_5d"] = macro["spy"].pct_change(5)
    result["spy_ret_20d"] = macro["spy"].pct_change(20)
    result["spy_above_sma50"] = (macro["spy"] > macro["spy"].rolling(50).mean()).astype(int)

    if "vix" in macro.columns:
        result["vix"] = macro["vix"]
        result["vix_z"] = (macro["vix"] - macro["vix"].rolling(252).mean()) / (macro["vix"].rolling(252).std() + 1e-9)
        result["high_vix"] = (macro["vix"] > 25).astype(int)

    if "tlt" in macro.columns:
        result["tlt_ret_5d"] = macro["tlt"].pct_change(5)
        result["risk_off"] = (macro["tlt"].pct_change(5) > 0.01).astype(int)

    if "gld" in macro.columns:
        result["gld_ret_5d"] = macro["gld"].pct_change(5)

    result = result.dropna()
    logger.info(f"Macro features: {result.shape} — {list(result.columns)}")
    return result


def merge_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro features onto a per-ticker OHLCV+indicators DataFrame."""
    if macro.empty:
        return df
    merged = df.join(macro, how="left")
    macro_cols = list(macro.columns)
    merged[macro_cols] = merged[macro_cols].ffill().fillna(0.0)
    return merged
