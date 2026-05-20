"""
Evaluate all trained models and print a unified performance report.
Compares: LSTM only, RL only, sentiment only, and full system vs Buy & Hold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.data.loader import load_ticker, load_prices_panel
from src.portfolio.risk import full_risk_report


def evaluate_buy_and_hold(ticker: str, config: dict) -> dict:
    df = load_ticker(ticker)
    test_prices = df.loc[config["data"]["test_start"]:]["close"]
    pv = test_prices / test_prices.iloc[0] * config.get("initial_balance", 100_000)
    report = full_risk_report(pv)
    report["strategy"] = "Buy & Hold"
    return report


def print_comparison_table(reports: list[dict]):
    metrics = ["strategy", "total_return", "annualized_return", "sharpe_ratio",
               "sortino_ratio", "max_drawdown", "win_rate", "n_trades"]
    header = f"{'Strategy':<20} {'Total Ret':>10} {'Ann Ret':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Win%':>7} {'Trades':>8}"
    print("\n" + "═" * 90)
    print("EVALUATION REPORT")
    print("═" * 90)
    print(header)
    print("─" * 90)
    for r in reports:
        print(
            f"{r.get('strategy', 'N/A'):<20} "
            f"{r.get('total_return', 0):>9.1%} "
            f"{r.get('annualized_return', 0):>9.1%} "
            f"{r.get('sharpe_ratio', 0):>8.3f} "
            f"{r.get('sortino_ratio', 0):>8.3f} "
            f"{r.get('max_drawdown', 0):>7.1%} "
            f"{r.get('win_rate', 0):>6.1%} "
            f"{r.get('n_trades', 0):>8}"
        )
    print("═" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--output", default="logs/evaluation_report.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    reports = []

    logger.info("Evaluating Buy & Hold baseline...")
    bh = evaluate_buy_and_hold(args.ticker, config)
    reports.append(bh)

    # TODO: Add LSTM-only, RL-only, and Full System evaluations
    # by loading checkpoints and running backtests with each signal source.
    # Example structure:
    #   lstm_report = run_backtest_with_signals(lstm_signals, "LSTM Only", config)
    #   rl_report   = run_backtest_with_signals(rl_signals, "RL Agent Only", config)
    #   full_report = run_backtest_with_signals(combined_signals, "Full System", config)
    #   reports.extend([lstm_report, rl_report, full_report])

    print_comparison_table(reports)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
