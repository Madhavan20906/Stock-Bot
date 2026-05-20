"""
Technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, etc.
Uses pandas-ta for fast vectorized computation.
"""

import pandas as pd
import pandas_ta as ta
from loguru import logger


def add_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Add a full suite of technical indicators to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        config: Optional config dict (from configs/default.yaml features.technical)

    Returns:
        DataFrame with new indicator columns appended
    """
    cfg = config or {}
    df = df.copy()

    # ── Trend ────────────────────────────────────────────────────────────────
    # Moving Averages
    df["sma_10"] = ta.sma(df["close"], length=10)
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    # MACD
    macd = ta.macd(
        df["close"],
        fast=cfg.get("macd_fast", 12),
        slow=cfg.get("macd_slow", 26),
        signal=cfg.get("macd_signal", 9),
    )
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    # ── Momentum ─────────────────────────────────────────────────────────────
    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=cfg.get("rsi_period", 14))

    # Stochastic Oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None:
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # Rate of Change
    df["roc_10"] = ta.roc(df["close"], length=10)

    # Williams %R
    df["willr"] = ta.willr(df["high"], df["low"], df["close"])

    # ── Volatility ───────────────────────────────────────────────────────────
    # Bollinger Bands
    bb = ta.bbands(df["close"], length=cfg.get("bb_period", 20), std=cfg.get("bb_std", 2.0))
    if bb is not None:
        upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
        mid_col   = [c for c in bb.columns if c.startswith("BBM")][0]
        lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
        df["bb_upper"] = bb[upper_col]
        df["bb_mid"]   = bb[mid_col]
        df["bb_lower"] = bb[lower_col]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ATR (Average True Range)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=cfg.get("atr_period", 14))

    # ── Volume ───────────────────────────────────────────────────────────────
    # On-Balance Volume
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Volume SMA
    df["volume_sma_20"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)

    # Money Flow Index
    df["mfi_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    # ── Price-derived ────────────────────────────────────────────────────────
    df["returns"] = df["close"].pct_change()
    import numpy as np
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["high_low_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["close_open_ratio"] = (df["close"] - df["open"]) / (df["open"] + 1e-9)

    # Volatility (rolling std of returns)
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()

    before = df.shape[1]
    df.dropna(inplace=True)
    logger.info(f"Technical indicators added: {df.shape[1] - 7} features, {len(df)} rows after dropna")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return non-OHLCV feature columns."""
    base = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in base]
