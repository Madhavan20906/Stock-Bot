"""
Extended backtest performance metrics beyond basic risk.
"""

import numpy as np
import pandas as pd


def annualized_return(portfolio_values: pd.Series, trading_days: int = 252) -> float:
    """Compound annual growth rate (CAGR)."""
    n_days = len(portfolio_values)
    total = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    return float(total ** (trading_days / n_days) - 1)


def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free: float = 0.05) -> pd.Series:
    """Rolling Sharpe ratio (default 63-day / quarterly window)."""
    excess = returns - risk_free / 252
    return (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)


def trade_statistics(trades_df: pd.DataFrame) -> dict:
    """Summary stats from a trades DataFrame (output of BacktestEngine)."""
    if trades_df.empty:
        return {}

    buys = trades_df[trades_df["action"] == "BUY"]
    sells = trades_df[trades_df["action"] == "SELL"]
    winning = sells[sells["pnl"] > 0]
    losing = sells[sells["pnl"] <= 0]

    avg_win = winning["pnl"].mean() if not winning.empty else 0
    avg_loss = losing["pnl"].mean() if not losing.empty else 0
    profit_factor = abs(winning["pnl"].sum() / (losing["pnl"].sum() + 1e-9))

    return {
        "total_trades": len(trades_df),
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / (len(sells) + 1e-9),
        "avg_win_pnl": float(avg_win),
        "avg_loss_pnl": float(avg_loss),
        "profit_factor": float(profit_factor),
        "total_commission": float(trades_df["commission"].sum()),
        "gross_pnl": float(sells["pnl"].sum()),
        "net_pnl": float(sells["pnl"].sum() - trades_df["commission"].sum()),
    }


def monthly_returns_table(portfolio_values: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns (rows=year, cols=month)."""
    returns = portfolio_values.resample("M").last().pct_change().dropna()
    df = pd.DataFrame({
        "year": returns.index.year,
        "month": returns.index.month,
        "return": returns.values,
    })
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    pivot["Annual"] = (1 + pivot).prod(axis=1) - 1
    return pivot.round(4)


def underwater_periods(portfolio_values: pd.Series) -> pd.DataFrame:
    """Identify drawdown periods with start, end, depth, recovery."""
    peak = portfolio_values.cummax()
    dd = (portfolio_values - peak) / peak
    in_drawdown = dd < -0.001

    periods = []
    start = None
    for date, flag in in_drawdown.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            depth = dd[start:date].min()
            periods.append({"start": start, "end": date, "depth": depth})
            start = None

    return pd.DataFrame(periods).sort_values("depth")
