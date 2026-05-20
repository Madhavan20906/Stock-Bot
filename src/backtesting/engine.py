"""
Event-driven backtesting engine.
Simulates portfolio performance by replaying historical signals day by day.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from loguru import logger

from src.portfolio.risk import full_risk_report


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float


@dataclass
class Trade:
    date: str
    ticker: str
    action: str           # BUY | SELL
    shares: float
    price: float
    commission: float
    pnl: float = 0.0


class BacktestEngine:
    """
    Replays a DataFrame of trade signals against historical price data.

    signals_df columns: date, ticker, signal (BUY|SELL|HOLD), confidence
    prices_df: wide DataFrame — index=date, columns=tickers
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.20,
    ):
        self.prices = prices
        self.signals = signals
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size

        self.cash = initial_balance
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.portfolio_history: list[dict] = []

    def _portfolio_value(self, date: str) -> float:
        stock_value = sum(
            pos.shares * self.prices.loc[date, pos.ticker]
            for pos in self.positions.values()
            if pos.ticker in self.prices.columns
        )
        return self.cash + stock_value

    def _execute_buy(self, date: str, ticker: str, confidence: float):
        if ticker not in self.prices.columns:
            return
        price = self.prices.loc[date, ticker]
        portfolio_val = self._portfolio_value(date)
        alloc = portfolio_val * self.max_position_size * confidence
        alloc = min(alloc, self.cash * 0.95)
        if alloc < price:
            return
        shares = (alloc * (1 - self.transaction_cost)) / price
        commission = alloc * self.transaction_cost
        self.cash -= alloc
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_cost = pos.shares * pos.avg_cost + alloc
            pos.shares += shares
            pos.avg_cost = total_cost / pos.shares
        else:
            self.positions[ticker] = Position(ticker, shares, price)
        self.trades.append(Trade(date, ticker, "BUY", shares, price, commission))

    def _execute_sell(self, date: str, ticker: str, confidence: float):
        if ticker not in self.positions or ticker not in self.prices.columns:
            return
        price = self.prices.loc[date, ticker]
        pos = self.positions[ticker]
        shares_to_sell = pos.shares * min(confidence, 1.0)
        proceeds = shares_to_sell * price
        commission = proceeds * self.transaction_cost
        pnl = shares_to_sell * (price - pos.avg_cost) - commission
        self.cash += proceeds - commission
        pos.shares -= shares_to_sell
        if pos.shares < 0.01:
            del self.positions[ticker]
        self.trades.append(Trade(date, ticker, "SELL", shares_to_sell, price, commission, pnl))

    def run(self) -> dict:
        """Run the backtest. Returns performance summary dict."""
        logger.info(f"Starting backtest: ${self.initial_balance:,.0f} | {len(self.prices)} days")

        dates = sorted(set(self.prices.index) & set(pd.to_datetime(self.signals["date"])))

        for date in dates:
            day_signals = self.signals[pd.to_datetime(self.signals["date"]) == date]

            for _, row in day_signals.iterrows():
                if row["signal"] == "BUY":
                    self._execute_buy(date, row["ticker"], row.get("confidence", 1.0))
                elif row["signal"] == "SELL":
                    self._execute_sell(date, row["ticker"], row.get("confidence", 1.0))

            portfolio_val = self._portfolio_value(date)
            self.portfolio_history.append({"date": date, "portfolio_value": portfolio_val})

        pv = pd.DataFrame(self.portfolio_history).set_index("date")["portfolio_value"]
        report = full_risk_report(pv)
        report["n_trades"] = len(self.trades)
        report["final_value"] = float(pv.iloc[-1])
        report["initial_value"] = self.initial_balance

        logger.info("═" * 50)
        logger.info("BACKTEST RESULTS")
        for k, v in report.items():
            logger.info(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        logger.info("═" * 50)

        return report

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (trades_df, portfolio_df)."""
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index("date")
        return trades_df, portfolio_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Load prices and signals, then instantiate BacktestEngine and call .run()")


if __name__ == "__main__":
    main()
