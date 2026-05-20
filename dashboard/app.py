"""
Plotly Dash real-time monitoring dashboard for StockBot.

Run: python dashboard/app.py
Open: http://localhost:8050
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="StockBot Dashboard",
)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "SPY"]


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("📈 StockBot", className="text-success fw-bold"), width=6),
        dbc.Col(html.P("AI Trading System Dashboard", className="text-muted mt-2"), width=6),
    ], className="my-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Ticker"),
            dcc.Dropdown(TICKERS, value="AAPL", id="ticker-dropdown", className="mb-3"),
        ], width=3),
        dbc.Col([
            dbc.Label("Period"),
            dcc.Dropdown(
                [{"label": "1 Month", "value": 30},
                 {"label": "3 Months", "value": 90},
                 {"label": "1 Year", "value": 252}],
                value=90, id="period-dropdown", className="mb-3"
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Signal Source"),
            dcc.Checklist(
                ["LSTM", "RL Agent", "Sentiment"],
                ["LSTM", "RL Agent", "Sentiment"],
                id="signal-checklist",
                className="mt-2",
            ),
        ], width=3),
        dbc.Col([
            dbc.Button("Refresh", id="refresh-btn", color="success", className="mt-4"),
        ], width=3),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Portfolio Value", className="text-muted"), html.H3(id="portfolio-value", className="text-success")])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Total Return", className="text-muted"), html.H3(id="total-return")])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Sharpe Ratio", className="text-muted"), html.H3(id="sharpe-ratio")])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H6("Current Signal", className="text-muted"), html.H3(id="current-signal")])
        ]), width=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="price-chart"), width=8),
        dbc.Col(dcc.Graph(id="signal-gauge"), width=4),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="portfolio-chart"), width=6),
        dbc.Col(dcc.Graph(id="sentiment-chart"), width=6),
    ]),

    dcc.Interval(id="interval", interval=60_000, n_intervals=0),  # Refresh every 60s
], fluid=True)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("price-chart", "figure"),
    Output("portfolio-value", "children"),
    Output("total-return", "children"),
    Output("sharpe-ratio", "children"),
    Output("current-signal", "children"),
    Output("signal-gauge", "figure"),
    Output("portfolio-chart", "figure"),
    Output("sentiment-chart", "figure"),
    Input("ticker-dropdown", "value"),
    Input("period-dropdown", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("interval", "n_intervals"),
)
def update_dashboard(ticker, period, n_clicks, n_intervals):
    # ── Synthetic demo data (replace with real model output) ─────────────
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=period, freq="B")
    base_price = {"AAPL": 180, "MSFT": 370, "GOOGL": 140, "AMZN": 175, "TSLA": 250, "JPM": 195, "GS": 380, "SPY": 475}.get(ticker, 200)
    returns = np.random.normal(0.0003, 0.015, period)
    prices = base_price * np.exp(np.cumsum(returns))

    portfolio_returns = np.random.normal(0.0005, 0.012, period)
    portfolio_values = 100_000 * np.exp(np.cumsum(portfolio_returns))
    sentiment_scores = np.random.normal(0.1, 0.3, period).clip(-1, 1)

    composite_signal = 0.62  # Demo signal score

    # ── Price chart ──────────────────────────────────────────────────────
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig_price.add_trace(go.Candlestick(
        x=dates,
        open=prices * np.random.uniform(0.99, 1.0, period),
        high=prices * np.random.uniform(1.0, 1.02, period),
        low=prices * np.random.uniform(0.98, 1.0, period),
        close=prices,
        name=ticker,
    ), row=1, col=1)
    volume = np.random.randint(20_000_000, 100_000_000, period)
    fig_price.add_trace(go.Bar(x=dates, y=volume, name="Volume", marker_color="rgba(0,200,100,0.4)"), row=2, col=1)
    fig_price.update_layout(
        title=f"{ticker} — Price & Volume",
        template="plotly_dark",
        height=450,
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )

    # ── Signal gauge ─────────────────────────────────────────────────────
    signal_label = "BUY" if composite_signal > 0.6 else ("SELL" if composite_signal < 0.4 else "HOLD")
    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[signal_label]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=composite_signal * 100,
        title={"text": f"Signal: {signal_label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": signal_color},
            "steps": [
                {"range": [0, 40], "color": "rgba(255,50,50,0.3)"},
                {"range": [40, 60], "color": "rgba(200,200,0,0.3)"},
                {"range": [60, 100], "color": "rgba(50,200,50,0.3)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": composite_signal * 100},
        },
    ))
    fig_gauge.update_layout(template="plotly_dark", height=300)

    # ── Portfolio chart ───────────────────────────────────────────────────
    bench_returns = np.random.normal(0.0002, 0.012, period)
    benchmark = 100_000 * np.exp(np.cumsum(bench_returns))
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(x=dates, y=portfolio_values, name="StockBot", line=dict(color="#00ff88", width=2)))
    fig_portfolio.add_trace(go.Scatter(x=dates, y=benchmark, name="Buy & Hold", line=dict(color="#888", width=1, dash="dash")))
    fig_portfolio.update_layout(title="Portfolio vs Benchmark", template="plotly_dark", height=300)

    # ── Sentiment chart ───────────────────────────────────────────────────
    colors = ["green" if s > 0 else "red" for s in sentiment_scores]
    fig_sentiment = go.Figure(go.Bar(
        x=dates, y=sentiment_scores, marker_color=colors, name="Daily Sentiment"
    ))
    fig_sentiment.update_layout(title=f"{ticker} News Sentiment", template="plotly_dark", height=300)

    # ── KPIs ─────────────────────────────────────────────────────────────
    pv_str = f"${portfolio_values[-1]:,.0f}"
    ret = (portfolio_values[-1] / 100_000 - 1) * 100
    ret_str = f"{ret:+.1f}%"
    sr = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    sr_str = f"{sr:.2f}"
    sig_str = signal_label

    return fig_price, pv_str, ret_str, sr_str, sig_str, fig_gauge, fig_portfolio, fig_sentiment


if __name__ == "__main__":
    app.run(
        host=os.getenv("DASH_HOST", "0.0.0.0"),
        port=int(os.getenv("DASH_PORT", 8050)),
        debug=True,
    )
