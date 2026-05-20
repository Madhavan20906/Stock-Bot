"""
Reusable Plotly chart builders for notebooks and the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK_TEMPLATE = "plotly_dark"


def candlestick_with_indicators(
    df: pd.DataFrame,
    ticker: str,
    show_bb: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """Full OHLCV candlestick chart with Bollinger Bands and volume."""
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3] if show_volume else [1.0],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=ticker, increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if show_bb and all(c in df.columns for c in ["bb_upper", "bb_lower", "bb_mid"]):
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                                  line=dict(color="rgba(255,165,0,0.5)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                                  line=dict(color="rgba(255,165,0,0.5)", width=1),
                                  fill="tonexty", fillcolor="rgba(255,165,0,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], name="BB Mid",
                                  line=dict(color="rgba(255,165,0,0.3)", width=1, dash="dot")), row=1, col=1)

    if show_volume and "volume" in df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"],
                              marker_color=colors, name="Volume"), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price Chart",
        template=DARK_TEMPLATE,
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def prediction_vs_actual(
    actual: pd.Series,
    predicted: pd.Series,
    ticker: str,
    future_preds: list[float] = None,
) -> go.Figure:
    """Plot actual vs predicted prices, with optional future forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                              name="Actual", line=dict(color="#00b4d8", width=2)))
    fig.add_trace(go.Scatter(x=predicted.index, y=predicted.values,
                              name="Predicted", line=dict(color="#ff9f1c", width=2, dash="dot")))

    if future_preds:
        last_date = actual.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_preds) + 1, freq="B")[1:]
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, name="Forecast",
            line=dict(color="#f72585", width=2, dash="dash"),
            mode="lines+markers",
        ))
        fig.add_vline(x=last_date, line_dash="dot", line_color="gray",
                      annotation_text="Today", annotation_position="top")

    fig.update_layout(title=f"{ticker} — Actual vs Predicted", template=DARK_TEMPLATE, height=450)
    return fig


def portfolio_performance(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series = None,
    trades_df: pd.DataFrame = None,
) -> go.Figure:
    """Plot portfolio equity curve vs benchmark with trade markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values.values,
                              name="StockBot", line=dict(color="#06d6a0", width=2.5)))

    if benchmark_values is not None:
        fig.add_trace(go.Scatter(x=benchmark_values.index, y=benchmark_values.values,
                                  name="Buy & Hold", line=dict(color="#8d99ae", width=1.5, dash="dash")))

    if trades_df is not None and not trades_df.empty:
        buys = trades_df[trades_df["action"] == "BUY"]
        sells = trades_df[trades_df["action"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buys["date"]),
                y=portfolio_values.reindex(pd.to_datetime(buys["date"]), method="nearest"),
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=10, color="#06d6a0"),
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sells["date"]),
                y=portfolio_values.reindex(pd.to_datetime(sells["date"]), method="nearest"),
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=10, color="#ef476f"),
            ))

    fig.update_layout(title="Portfolio Equity Curve", template=DARK_TEMPLATE, height=500)
    return fig


def drawdown_chart(portfolio_values: pd.Series) -> go.Figure:
    """Plot rolling drawdown."""
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak * 100
    fig = go.Figure(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(239,71,111,0.3)",
        line=dict(color="#ef476f", width=1.5),
        name="Drawdown %",
    ))
    fig.update_layout(title="Drawdown (%)", template=DARK_TEMPLATE, height=300,
                      yaxis_ticksuffix="%")
    return fig


def correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """Ticker correlation heatmap."""
    corr = prices.pct_change().dropna().corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Return Correlation Matrix", template=DARK_TEMPLATE, height=500)
    return fig
