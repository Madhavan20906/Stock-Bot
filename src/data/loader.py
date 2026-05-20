"""
Loads the Kaggle Stock Market Dataset.
Dataset: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

Expected structure after download:
  data/raw/
    stocks/     ← individual stock CSVs (AAPL.csv, MSFT.csv, ...)
    etfs/       ← ETF CSVs
    symbols_valid_meta.csv
"""

import logging
from pathlib import Path

import pandas as pd
from loguru import logger


RAW_PATH = Path("data/raw")
STOCKS_PATH = RAW_PATH / "stocks"
ETFS_PATH = RAW_PATH / "etfs"
META_PATH = RAW_PATH / "symbols_valid_meta.csv"

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def load_ticker(ticker: str, adjust: bool = True) -> pd.DataFrame:
    """Load OHLCV data for a single ticker.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'
        adjust: If True, use Adj Close instead of Close

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    for folder in [STOCKS_PATH, ETFS_PATH]:
        path = folder / f"{ticker.upper()}.csv"
        if path.exists():
            raw = pd.read_csv(path, nrows=0)
            cols = list(raw.columns)

            if "Date" in cols:
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            else:
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df.index.name = "Date"

            df.sort_index(inplace=True)
            df.columns = [c.strip() for c in df.columns]

            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns in {ticker}: {missing}. Found: {list(df.columns)}")

            if adjust and "Adj Close" in df.columns:
                df["close"] = df["Adj Close"]
            else:
                df["close"] = df["Close"]

            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.dropna()
            logger.info(f"Loaded {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            return df
    raise FileNotFoundError(
        f"Ticker '{ticker}' not found in {STOCKS_PATH} or {ETFS_PATH}. "
        "Run: python scripts/download_data.py"
    )


def load_multiple(tickers: list[str], start: str = None, end: str = None) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple tickers.

    Args:
        tickers: List of symbols
        start: Start date string 'YYYY-MM-DD' (optional)
        end: End date string 'YYYY-MM-DD' (optional)

    Returns:
        Dict mapping ticker -> DataFrame
    """
    data = {}
    for ticker in tickers:
        try:
            df = load_ticker(ticker)
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]
            if len(df) > 0:
                data[ticker] = df
            else:
                logger.warning(f"{ticker}: no data in range {start} – {end}")
        except FileNotFoundError as e:
            logger.warning(str(e))
    return data


def load_metadata() -> pd.DataFrame:
    """Load the symbols metadata CSV."""
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")
    return pd.read_csv(META_PATH)


def load_prices_panel(tickers: list[str], start: str = None, end: str = None) -> pd.DataFrame:
    """Return a wide-format DataFrame of adjusted close prices.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker
    """
    data = load_multiple(tickers, start=start, end=end)
    prices = pd.DataFrame({t: df["close"] for t, df in data.items()})
    prices.sort_index(inplace=True)
    logger.info(f"Prices panel: {prices.shape} — {len(prices.columns)} tickers")
    return prices
