"""
Integration tests: wire together data → features → model → signal → backtest.
Uses synthetic data so no Kaggle download is required.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessor import StockPreprocessor
from src.models.lstm.model import LSTMPredictor
from src.models.rl_agent.environment import StockTradingEnv
from src.signals.generator import SignalGenerator, Signal
from src.portfolio.risk import full_risk_report
from src.backtesting.engine import BacktestEngine


def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    return pd.DataFrame({
        "open":   prices * np.random.uniform(0.99, 1.0, n),
        "high":   prices * np.random.uniform(1.0, 1.02, n),
        "low":    prices * np.random.uniform(0.98, 1.0, n),
        "close":  prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


class TestLSTMForwardPass:
    def test_forward_shape(self):
        model = LSTMPredictor(input_size=5, hidden_size=32, num_layers=2, output_steps=3)
        x = torch.randn(4, 20, 5)
        preds, attn = model(x)
        assert preds.shape == (4, 3)

    def test_no_nan_in_output(self):
        model = LSTMPredictor(input_size=5, hidden_size=32)
        x = torch.randn(2, 15, 5)
        preds, _ = model(x)
        assert not torch.isnan(preds).any()


class TestPreprocessorToLSTM:
    def test_full_pipeline(self):
        df = make_ohlcv(200)
        prep = StockPreprocessor(sequence_length=30)
        prep.fit(df)
        X, y = prep.make_sequences(df)

        model = LSTMPredictor(
            input_size=X.shape[-1], hidden_size=16, num_layers=1, output_steps=1
        )
        with torch.no_grad():
            preds, _ = model(torch.from_numpy(X[:4]))
        assert preds.shape == (4, 1)


class TestRLEnvironment:
    def test_gymnasium_api_compliance(self):
        df = make_ohlcv(150)
        features = df.values.astype(np.float32)
        prices = df["close"].values
        env = StockTradingEnv(features, prices, window_size=20)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

        total_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        assert isinstance(total_reward, float)
        assert "portfolio_value" in info


class TestSignalGenerator:
    def test_signal_generation_coverage(self):
        gen = SignalGenerator()
        test_cases = [
            (1.1, 1.0, 0.9, 0.8),    # Strong BUY
            (0.9, 1.0, -0.9, -0.8),  # Strong SELL
            (1.001, 1.0, 0.0, 0.0),  # Likely HOLD
        ]
        signals_seen = set()
        for lstm_pred, price, rl, sent in test_cases:
            s = gen.generate("TEST", lstm_pred, price, rl, sent)
            signals_seen.add(s.signal)
        assert Signal.BUY in signals_seen
        assert Signal.SELL in signals_seen


class TestBacktestEngine:
    def test_backtest_runs_and_returns_report(self):
        prices = make_ohlcv(100)
        prices_wide = pd.DataFrame({"AAPL": prices["close"]})

        signals = pd.DataFrame([
            {"date": prices.index[30], "ticker": "AAPL", "signal": "BUY",  "confidence": 0.8},
            {"date": prices.index[50], "ticker": "AAPL", "signal": "SELL", "confidence": 0.7},
            {"date": prices.index[70], "ticker": "AAPL", "signal": "BUY",  "confidence": 0.6},
        ])

        engine = BacktestEngine(prices_wide, signals, initial_balance=50_000)
        report = engine.run()

        assert "total_return" in report
        assert "sharpe_ratio" in report
        assert report["n_trades"] == 3

    def test_portfolio_value_tracks_correctly(self):
        prices = make_ohlcv(60)
        prices_wide = pd.DataFrame({"AAPL": prices["close"]})
        signals = pd.DataFrame([
            {"date": prices.index[10], "ticker": "AAPL", "signal": "BUY", "confidence": 1.0},
        ])
        engine = BacktestEngine(prices_wide, signals, initial_balance=10_000)
        engine.run()
        _, portfolio_df = engine.to_dataframes()
        assert len(portfolio_df) > 0
        assert (portfolio_df["portfolio_value"] >= 0).all()


class TestRiskMetrics:
    def test_full_report_on_growing_portfolio(self):
        np.random.seed(7)
        pv = pd.Series(100_000 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 252))))
        report = full_risk_report(pv)
        assert report["total_return"] > 0
        assert -1 <= report["max_drawdown"] <= 0
        assert 0 <= report["win_rate"] <= 1
