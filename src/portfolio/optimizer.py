"""
Portfolio optimization using Modern Portfolio Theory (Markowitz).
Wraps PyPortfolioOpt for mean-variance optimization and Sharpe maximization.
"""

import numpy as np
import pandas as pd
from loguru import logger

from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


class PortfolioOptimizer:
    """Markowitz mean-variance portfolio optimizer.

    Usage:
        opt = PortfolioOptimizer(prices_df)
        weights = opt.max_sharpe()
        allocation = opt.allocate(weights, total_portfolio_value=100_000)
    """

    def __init__(self, prices: pd.DataFrame, risk_free_rate: float = 0.05):
        """
        Args:
            prices: Wide DataFrame of adjusted close prices (index=date, cols=tickers)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.prices = prices.dropna(axis=1, how="any")
        self.risk_free_rate = risk_free_rate
        self._compute_inputs()

    def _compute_inputs(self):
        """Compute expected returns and covariance matrix."""
        self.mu = expected_returns.mean_historical_return(self.prices)
        self.S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()
        logger.info(f"Portfolio inputs computed for {len(self.mu)} assets")

    def max_sharpe(self, weight_bounds: tuple = (0.0, 0.25)) -> dict[str, float]:
        """Maximize Sharpe ratio.

        Returns:
            Dict of {ticker: weight}
        """
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        logger.info(
            f"Max Sharpe — Expected Return: {perf[0]:.2%} | "
            f"Volatility: {perf[1]:.2%} | Sharpe: {perf[2]:.3f}"
        )
        return dict(weights)

    def min_volatility(self, weight_bounds: tuple = (0.0, 0.25)) -> dict[str, float]:
        """Minimize portfolio volatility."""
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.min_volatility()
        weights = ef.clean_weights()
        return dict(weights)

    def efficient_risk(
        self, target_volatility: float = 0.15, weight_bounds: tuple = (0.0, 0.25)
    ) -> dict[str, float]:
        """Maximize return for a given target annual volatility."""
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.efficient_risk(target_volatility)
        weights = ef.clean_weights()
        return dict(weights)

    def allocate(
        self,
        weights: dict[str, float],
        total_portfolio_value: float,
    ) -> tuple[dict[str, int], float]:
        """Convert weights to integer share counts.

        Returns:
            (allocation dict {ticker: shares}, leftover cash)
        """
        latest_prices = get_latest_prices(self.prices)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        logger.info(f"Allocation: {allocation} | Leftover cash: ${leftover:.2f}")
        return allocation, leftover

    def performance_summary(self, weights: dict[str, float]) -> dict:
        """Return annualized performance metrics for a given weight dict."""
        ef = EfficientFrontier(self.mu, self.S)
        ef.set_weights(weights)
        ret, vol, sharpe = ef.portfolio_performance(
            verbose=False, risk_free_rate=self.risk_free_rate
        )
        return {
            "expected_annual_return": round(ret, 4),
            "annual_volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
        }
