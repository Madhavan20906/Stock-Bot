"""
Feature store: caches computed features to Parquet to avoid recomputing
technical indicators and sentiment on every run.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from loguru import logger

STORE_DIR = Path("data/processed/feature_store")


def _cache_key(ticker: str, config: dict) -> str:
    payload = json.dumps({"ticker": ticker, **config}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def get(ticker: str, config: dict) -> pd.DataFrame | None:
    """Return cached features if available, else None."""
    key = _cache_key(ticker, config)
    path = STORE_DIR / f"{ticker}_{key}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Feature cache HIT: {ticker} ({key})")
        return df
    logger.info(f"Feature cache MISS: {ticker} ({key})")
    return None


def put(ticker: str, config: dict, df: pd.DataFrame):
    """Write features to the cache."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(ticker, config)
    path = STORE_DIR / f"{ticker}_{key}.parquet"
    df.to_parquet(path)
    logger.info(f"Feature cache WRITE: {ticker} ({key}) — {df.shape}")


def build_features(ticker: str, config: dict, force: bool = False) -> pd.DataFrame:
    """Build or load cached features for a ticker.

    Pipeline:
        load_ticker → add_all_indicators → merge_sentiment → return
    """
    if not force:
        cached = get(ticker, config)
        if cached is not None:
            return cached

    from src.data.loader import load_ticker
    from src.features.technical import add_all_indicators

    df = load_ticker(ticker)
    df = add_all_indicators(df, config.get("features", {}).get("technical", {}))

    # Optionally merge sentiment
    try:
        from src.features.sentiment import build_sentiment_features, merge_sentiment_with_ohlcv
        sentiment_df = build_sentiment_features(
            tickers=[ticker],
            start=str(df.index[0].date()),
            end=str(df.index[-1].date()),
        )
        df = merge_sentiment_with_ohlcv(df, sentiment_df, ticker)
        logger.info(f"Sentiment merged for {ticker}")
    except Exception as e:
        logger.warning(f"Could not merge sentiment for {ticker}: {e}")
        df["sentiment_score"] = 0.0

    put(ticker, config, df)
    return df


def clear_cache():
    """Remove all cached feature files."""
    if STORE_DIR.exists():
        for f in STORE_DIR.glob("*.parquet"):
            f.unlink()
        logger.info("Feature cache cleared.")
