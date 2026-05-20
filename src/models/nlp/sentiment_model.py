"""
FinBERT-based sentiment analysis for financial news.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial phrase bank.
Labels: positive, negative, neutral
"""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger


FINBERT_MODEL = "ProsusAI/finbert"
LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}


class FinBERTSentiment:
    """Wrapper around ProsusAI/finbert for financial sentiment scoring."""

    def __init__(self, model_name: str = FINBERT_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading FinBERT from '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}
        logger.info("FinBERT loaded.")

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Score a list of news headlines/articles.

        Returns:
            List of dicts: {text, label, score, sentiment_score}
            where sentiment_score ∈ [-1, 1]
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for text, prob in zip(batch, probs):
                label_idx = int(np.argmax(prob))
                label = self.labels[label_idx]
                # Weighted sentiment score: +1 × P(pos) - 1 × P(neg)
                pos_idx = [k for k, v in self.labels.items() if v == "positive"][0]
                neg_idx = [k for k, v in self.labels.items() if v == "negative"][0]
                sentiment_score = float(prob[pos_idx] - prob[neg_idx])
                results.append({
                    "text": text[:100],
                    "label": label,
                    "confidence": float(prob[label_idx]),
                    "sentiment_score": sentiment_score,
                })
        return results

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
        """Add sentiment columns to a DataFrame of news items."""
        texts = df[text_col].tolist()
        preds = self.predict(texts)
        df = df.copy()
        df["sentiment_label"] = [p["label"] for p in preds]
        df["sentiment_score"] = [p["sentiment_score"] for p in preds]
        df["sentiment_confidence"] = [p["confidence"] for p in preds]
        return df

    def aggregate_daily(
        self, df: pd.DataFrame, date_col: str = "date", decay: float = 0.8
    ) -> pd.Series:
        """Aggregate sentiment scores per day with exponential recency weighting.

        Returns:
            pd.Series indexed by date with daily sentiment score ∈ [-1, 1]
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df = df.sort_values([date_col, "sentiment_confidence"], ascending=[True, False])

        daily = {}
        for date, group in df.groupby(date_col):
            scores = group["sentiment_score"].values
            weights = np.array([decay ** i for i in range(len(scores))])
            daily[date] = float(np.average(scores, weights=weights))

        return pd.Series(daily, name="sentiment_score")
