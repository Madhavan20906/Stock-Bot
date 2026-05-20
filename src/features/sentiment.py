"""
Aggregates FinBERT sentiment scores into per-ticker daily features
ready for use as input to the LSTM and signal generator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


def load_sentiment_cache(path: str = "data/processed/sentiment.parquet") -> pd.DataFrame | None:
    """Load cached sentiment scores if available."""
    p = Path(path)
    if p.exists():
        df = pd.read_parquet(p)
        logger.info(f"Loaded sentiment cache: {df.shape}")
        return df
    return None


def save_sentiment_cache(df: pd.DataFrame, path: str = "data/processed/sentiment.parquet"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Saved sentiment cache: {df.shape}")


def build_sentiment_features(
    tickers: list[str],
    start: str,
    end: str,
    decay: float = 0.8,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Builds a DataFrame of daily sentiment scores per ticker.

    Returns:
        DataFrame with MultiIndex (date, ticker) and column 'sentiment_score'
    """
    cache = load_sentiment_cache() if use_cache else None
    if cache is not None:
        logger.info("Using cached sentiment scores.")
        return cache

    # Lazy import to avoid loading FinBERT unless needed
    from src.models.nlp.sentiment_model import FinBERTSentiment
    from src.models.nlp.news_fetcher import fetch_news

    model = FinBERTSentiment()
    records = []

    for ticker in tickers:
        logger.info(f"Fetching news for {ticker}...")
        news_df = fetch_news(ticker)
        if news_df.empty:
            logger.warning(f"No news found for {ticker}, filling zeros.")
            date_range = pd.date_range(start, end, freq="B")
            for date in date_range:
                records.append({"date": date, "ticker": ticker, "sentiment_score": 0.0})
            continue

        scored = model.score_dataframe(news_df, text_col="headline")
        daily = model.aggregate_daily(scored, date_col="date", decay=decay)

        full_range = pd.date_range(start, end, freq="B")
        daily = daily.reindex(full_range).fillna(method="ffill").fillna(0.0)

        for date, score in daily.items():
            records.append({"date": date, "ticker": ticker, "sentiment_score": float(score)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()

    if use_cache:
        save_sentiment_cache(df)

    return df


def merge_sentiment_with_ohlcv(
    ohlcv_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Merge daily sentiment scores into an OHLCV+indicators DataFrame."""
    if (slice(None), ticker) in sentiment_df.index:
        ticker_sentiment = sentiment_df.xs(ticker, level="ticker")["sentiment_score"]
    else:
        logger.warning(f"No sentiment data for {ticker}, filling zeros.")
        ticker_sentiment = pd.Series(0.0, index=ohlcv_df.index, name="sentiment_score")

    merged = ohlcv_df.copy()
    merged["sentiment_score"] = ticker_sentiment.reindex(merged.index).fillna(0.0)
    return merged
