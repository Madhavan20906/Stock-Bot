"""
Signal Generator: combines LSTM, RL, and sentiment signals into a unified
buy/sell/hold decision with confidence score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    ticker: str
    signal: Signal
    confidence: float          # 0 to 1
    lstm_score: float          # Directional prediction from LSTM (-1 to 1)
    rl_action: float           # RL agent action (-1 to 1)
    sentiment_score: float     # NLP sentiment (-1 to 1)
    composite_score: float     # Weighted combination
    reasoning: str


class SignalGenerator:
    """Combines multiple model signals with configurable weights."""

    def __init__(
        self,
        lstm_weight: float = 0.4,
        rl_weight: float = 0.4,
        sentiment_weight: float = 0.2,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ):
        total = lstm_weight + rl_weight + sentiment_weight
        self.w_lstm = lstm_weight / total
        self.w_rl = rl_weight / total
        self.w_sent = sentiment_weight / total
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate(
        self,
        ticker: str,
        lstm_prediction: float,       # predicted next-day return (normalized)
        current_price: float,
        rl_action: float,             # RL agent output in [-1, 1]
        sentiment_score: float,       # FinBERT score in [-1, 1]
    ) -> TradeSignal:
        """Generate a composite trade signal.

        Args:
            lstm_prediction: Predicted price (scaled) for next day
            current_price:   Current price (scaled)
        """
        # Normalize LSTM to [-1, 1]: positive if predicted > current
        lstm_score = float(np.tanh((lstm_prediction - current_price) * 10))

        # All scores in [-1, 1], map to [0, 1] for weighting
        def to_prob(score: float) -> float:
            return (score + 1) / 2

        composite = (
            self.w_lstm * to_prob(lstm_score)
            + self.w_rl * to_prob(rl_action)
            + self.w_sent * to_prob(sentiment_score)
        )

        if composite >= self.buy_threshold:
            signal = Signal.BUY
            confidence = (composite - self.buy_threshold) / (1 - self.buy_threshold)
        elif composite <= self.sell_threshold:
            signal = Signal.SELL
            confidence = (self.sell_threshold - composite) / self.sell_threshold
        else:
            signal = Signal.HOLD
            confidence = 1.0 - abs(composite - 0.5) * 2

        reasoning = (
            f"LSTM={lstm_score:+.3f}(w={self.w_lstm:.2f}) | "
            f"RL={rl_action:+.3f}(w={self.w_rl:.2f}) | "
            f"Sentiment={sentiment_score:+.3f}(w={self.w_sent:.2f}) | "
            f"Composite={composite:.3f}"
        )

        return TradeSignal(
            ticker=ticker,
            signal=signal,
            confidence=float(confidence),
            lstm_score=lstm_score,
            rl_action=rl_action,
            sentiment_score=sentiment_score,
            composite_score=composite,
            reasoning=reasoning,
        )

    def generate_batch(self, records: list[dict]) -> list[TradeSignal]:
        """Generate signals for multiple tickers at once."""
        signals = []
        for r in records:
            try:
                s = self.generate(**r)
                signals.append(s)
                logger.info(f"[{s.ticker}] {s.signal.value} (conf={s.confidence:.2f}) — {s.reasoning}")
            except Exception as e:
                logger.error(f"Signal error for {r.get('ticker')}: {e}")
        return signals
