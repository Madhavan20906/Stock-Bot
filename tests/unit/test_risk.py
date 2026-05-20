"""Unit tests for risk metrics."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.risk import max_drawdown, sharpe_ratio, value_at_risk, full_risk_report


def test_max_drawdown_flat():
    pv = pd.Series([100.0] * 10)
    assert max_drawdown(pv) == pytest.approx(0.0)


def test_max_drawdown_decline():
    pv = pd.Series([100.0, 90.0, 80.0, 85.0, 70.0, 75.0])
    # Peak is 100, trough is 70 → -30%
    assert max_drawdown(pv) == pytest.approx(-0.30)


def test_sharpe_positive_returns():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))
    sr = sharpe_ratio(returns)
    assert sr > 0


def test_var_ordering():
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var_95 = value_at_risk(returns, 0.95)
    var_99 = value_at_risk(returns, 0.99)
    # 99% VaR should be more extreme (lower) than 95% VaR
    assert var_99 < var_95


def test_full_report_keys():
    np.random.seed(2)
    pv = pd.Series(100_000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 252))))
    report = full_risk_report(pv)
    required_keys = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate", "var_95"}
    assert required_keys.issubset(report.keys())
