"""Unit tests for signal generator."""

import pytest
from src.signals.generator import SignalGenerator, Signal


@pytest.fixture
def generator():
    return SignalGenerator(lstm_weight=0.4, rl_weight=0.4, sentiment_weight=0.2)


def test_buy_signal(generator):
    signal = generator.generate("AAPL", lstm_prediction=1.05, current_price=1.0, rl_action=0.8, sentiment_score=0.7)
    assert signal.signal == Signal.BUY
    assert signal.confidence > 0


def test_sell_signal(generator):
    signal = generator.generate("AAPL", lstm_prediction=0.92, current_price=1.0, rl_action=-0.9, sentiment_score=-0.8)
    assert signal.signal == Signal.SELL
    assert signal.confidence > 0


def test_hold_signal(generator):
    signal = generator.generate("AAPL", lstm_prediction=1.001, current_price=1.0, rl_action=0.0, sentiment_score=0.0)
    assert signal.signal == Signal.HOLD


def test_weights_normalized():
    gen = SignalGenerator(lstm_weight=2, rl_weight=2, sentiment_weight=1)
    assert abs(gen.w_lstm + gen.w_rl + gen.w_sent - 1.0) < 1e-9


def test_ticker_preserved(generator):
    signal = generator.generate("TSLA", lstm_prediction=1.1, current_price=1.0, rl_action=0.5, sentiment_score=0.5)
    assert signal.ticker == "TSLA"
