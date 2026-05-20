# StockBot — AI Stock Market Prediction & Trading System

[![CI](https://github.com/yourname/stockbot/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/stockbot/actions)
[![codecov](https://codecov.io/gh/yourname/stockbot/branch/main/graph/badge.svg)](https://codecov.io/gh/yourname/stockbot)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A production-grade AI trading system combining LSTM price prediction, Reinforcement Learning trading agents, NLP-based news sentiment analysis, and Markowitz portfolio optimization.

> ⚠️ **Disclaimer**: This project is for educational and research purposes only. It is not financial advice. Never trade real money with algorithmic systems you don't fully understand.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Pipeline                         │
│  Kaggle Dataset ──► Feature Engineering ──► Preprocessing   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌─────────────┐  ┌────────────────┐
   │    LSTM    │  │  RL Agent   │  │  NLP Sentiment │
   │  (Price    │  │  (Trading   │  │  (News Feed    │
   │Prediction) │  │  Decisions) │  │   Analysis)    │
   └─────┬──────┘  └──────┬──────┘  └───────┬────────┘
         │                │                  │
         └────────────────┼──────────────────┘
                          ▼
              ┌─────────────────────┐
              │   Signal Generator  │
              │  (Buy/Sell/Hold)    │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Portfolio Optimizer │
              │ (Markowitz / MPT)   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Backtesting       │
              │   Engine            │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   FastAPI / Dash    │
              │   Dashboard         │
              └─────────────────────┘
```

---

## Features

- **LSTM Price Prediction**: Multi-step ahead forecasting using stacked LSTM networks with attention
- **RL Trading Agent**: PPO-based agent trained with `stable-baselines3` on a custom Gym environment
- **News Sentiment Analysis**: FinBERT fine-tuned on financial news for buy/sell signal generation
- **Portfolio Optimization**: Mean-variance optimization (Markowitz) + Sharpe ratio maximization
- **Backtesting Engine**: Event-driven backtester with transaction cost modeling
- **REST API**: FastAPI endpoints for live inference
- **Dashboard**: Plotly Dash real-time monitoring dashboard

---

## Project Structure

```
stockbot/
├── data/
│   ├── raw/                    # Original Kaggle dataset (gitignored)
│   ├── processed/              # Cleaned, feature-engineered data
│   └── external/               # News feeds, macro data
├── src/
│   ├── data/
│   │   ├── loader.py           # Kaggle dataset loading
│   │   ├── preprocessor.py     # Cleaning, normalization
│   │   └── feature_store.py    # Feature caching with Parquet
│   ├── features/
│   │   ├── technical.py        # RSI, MACD, Bollinger Bands, etc.
│   │   ├── sentiment.py        # News sentiment aggregation
│   │   └── macro.py            # Market-wide features
│   ├── models/
│   │   ├── lstm/
│   │   │   ├── model.py        # LSTM + Attention architecture
│   │   │   ├── train.py        # Training loop with early stopping
│   │   │   └── predict.py      # Multi-step inference
│   │   ├── rl_agent/
│   │   │   ├── environment.py  # Custom OpenAI Gym environment
│   │   │   ├── agent.py        # PPO agent wrapper
│   │   │   └── train.py        # RL training loop
│   │   └── nlp/
│   │       ├── sentiment_model.py  # FinBERT wrapper
│   │       ├── news_fetcher.py     # RSS / NewsAPI integration
│   │       └── train_finetune.py   # Fine-tuning on financial data
│   ├── signals/
│   │   └── generator.py        # Combines all signals into trade decisions
│   ├── portfolio/
│   │   ├── optimizer.py        # Markowitz mean-variance optimization
│   │   └── risk.py             # VaR, CVaR, drawdown metrics
│   ├── backtesting/
│   │   ├── engine.py           # Event-driven backtest loop
│   │   └── metrics.py          # Sharpe, Sortino, max drawdown
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   └── routes.py           # Prediction & portfolio endpoints
│   └── visualization/
│       └── plots.py            # Plotly chart builders
├── dashboard/
│   └── app.py                  # Plotly Dash live dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lstm_experiments.ipynb
│   ├── 04_rl_training.ipynb
│   └── 05_portfolio_analysis.ipynb
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
│   ├── default.yaml
│   ├── lstm.yaml
│   ├── rl_agent.yaml
│   └── portfolio.yaml
├── scripts/
│   ├── download_data.py
│   ├── run_pipeline.py
│   └── evaluate_all.py
└── .github/
    └── workflows/
        ├── ci.yml
        └── train.yml
```

---

## Dataset

Download from Kaggle: [Stock Market Dataset by Jackson](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

```bash
# Using Kaggle CLI
kaggle datasets download -d jacksoncrow/stock-market-dataset -p data/raw/ --unzip
```

Or run the helper script:
```bash
python scripts/download_data.py
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourname/stockbot.git
cd stockbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Set up pre-commit hooks
pre-commit install

# 5. Copy and configure environment variables
cp .env.example .env
# Edit .env: add NEWSAPI_KEY, KAGGLE credentials, etc.
```

---

## Usage

### Run the full pipeline
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

### Train LSTM model
```bash
python src/models/lstm/train.py --config configs/lstm.yaml --ticker AAPL
```

### Train RL agent
```bash
python src/models/rl_agent/train.py --config configs/rl_agent.yaml
```

### Run backtesting
```bash
python src/backtesting/engine.py --start 2020-01-01 --end 2023-12-31
```

### Launch dashboard
```bash
python dashboard/app.py
# Open http://localhost:8050
```

### Start API server
```bash
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs
```

---

## ML Concepts Used

| Component | Technique | Library |
|---|---|---|
| Price Prediction | Stacked LSTM + Attention | PyTorch |
| Trading Decisions | PPO (Proximal Policy Optimization) | stable-baselines3 |
| Sentiment Analysis | FinBERT (BERT fine-tuned on finance) | HuggingFace Transformers |
| Portfolio Optimization | Markowitz Mean-Variance / Sharpe | scipy, cvxpy |
| Technical Features | RSI, MACD, Bollinger Bands, OBV | ta-lib, pandas-ta |
| Backtesting | Event-driven simulation | Custom engine |

---

## Results (Example)

| Metric | Buy & Hold | LSTM Only | RL Agent | Full System |
|---|---|---|---|---|
| Annual Return | 12.4% | 18.7% | 22.1% | **26.3%** |
| Sharpe Ratio | 0.71 | 1.02 | 1.31 | **1.58** |
| Max Drawdown | -34.2% | -22.1% | -18.4% | **-15.9%** |
| Win Rate | — | 54.2% | 61.3% | **63.8%** |

*Backtest period: Jan 2019 – Dec 2023 on S&P 500 constituents*

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Write tests for new functionality
3. Run `pytest tests/ --cov=src` and ensure coverage > 80%
4. Open a pull request against `main`

---

## License

MIT License — see [LICENSE](LICENSE) for details.
"# Stock-Bot" 
