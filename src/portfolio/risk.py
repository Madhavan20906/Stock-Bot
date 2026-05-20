"""
Risk metrics: VaR, CVaR, Max Drawdown, Sortino ratio.
"""

import numpy as np
import pandas as pd


def max_drawdown(portfolio_values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (1-day, annot negative number)."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall): mean of losses beyond VaR."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = returns - risk_free_rate / 252
    return float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sortino ratio — penalizes only downside volatility."""
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0].std()
    return float(excess.mean() / (downside + 1e-9) * np.sqrt(252))


def calmar_ratio(returns: pd.Series, portfolio_values: pd.Series) -> float:
    """Annual return / Max drawdown."""
    annual_return = returns.mean() * 252
    mdd = abs(max_drawdown(portfolio_values))
    return float(annual_return / (mdd + 1e-9))


def full_risk_report(portfolio_values: pd.Series) -> dict:
    """Return a complete risk metrics dictionary."""
    returns = portfolio_values.pct_change().dropna()
    return {
        "total_return": float((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1),
        "annualized_return": float(returns.mean() * 252),
        "annualized_volatility": float(returns.std() * np.sqrt(252)),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "calmar_ratio": calmar_ratio(returns, portfolio_values),
        "win_rate": float((returns > 0).mean()),
    }
