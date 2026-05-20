"""
Fetches financial news headlines from NewsAPI and Yahoo Finance RSS.
"""

import os
from datetime import datetime, timedelta

import feedparser
import pandas as pd
from loguru import logger

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False


# Yahoo Finance RSS feeds per ticker
YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


def fetch_yahoo_rss(ticker: str, max_items: int = 50) -> pd.DataFrame:
    """Fetch headlines from Yahoo Finance RSS for a ticker."""
    url = YF_RSS.format(ticker=ticker)
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        rows.append({
            "date": entry.get("published", ""),
            "ticker": ticker,
            "source": "yahoo_finance",
            "headline": entry.get("title", ""),
            "url": entry.get("link", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.dropna(subset=["date"])
    logger.info(f"Yahoo RSS: {len(df)} articles for {ticker}")
    return df


def fetch_newsapi(ticker: str, company_name: str, days_back: int = 7) -> pd.DataFrame:
    """Fetch from NewsAPI (requires NEWSAPI_KEY env var)."""
    if not NEWSAPI_AVAILABLE:
        logger.warning("newsapi-python not installed. Skipping NewsAPI.")
        return pd.DataFrame()

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_KEY not set. Skipping NewsAPI fetch.")
        return pd.DataFrame()

    client = NewsApiClient(api_key=api_key)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        response = client.get_everything(
            q=f"{company_name} OR {ticker}",
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=100,
        )
        articles = response.get("articles", [])
        rows = [{
            "date": a["publishedAt"],
            "ticker": ticker,
            "source": a["source"]["name"],
            "headline": a["title"],
            "url": a["url"],
        } for a in articles if a.get("title")]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        logger.info(f"NewsAPI: {len(df)} articles for {ticker}")
        return df
    except Exception as e:
        logger.error(f"NewsAPI error for {ticker}: {e}")
        return pd.DataFrame()


def fetch_news(
    ticker: str,
    company_name: str = "",
    days_back: int = 7,
) -> pd.DataFrame:
    """Fetch from all available sources and deduplicate."""
    rss = fetch_yahoo_rss(ticker)
    api = fetch_newsapi(ticker, company_name or ticker, days_back=days_back)
    combined = pd.concat([rss, api], ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"])
    combined = combined.sort_values("date", ascending=False)
    return combined.reset_index(drop=True)
