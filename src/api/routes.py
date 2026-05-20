"""
API route handlers for prediction, signals, and portfolio endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request/Response Models ──────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    steps: int = Field(default=5, ge=1, le=30, description="Days to forecast")


class PredictionResponse(BaseModel):
    ticker: str
    predictions: list[float]
    confidence: float


class SignalRequest(BaseModel):
    ticker: str
    include_sentiment: bool = True


class SignalResponse(BaseModel):
    ticker: str
    signal: str           # BUY | SELL | HOLD
    confidence: float
    lstm_score: float
    rl_action: float
    sentiment_score: float
    composite_score: float
    reasoning: str


class PortfolioRequest(BaseModel):
    tickers: list[str]
    total_value: float = Field(default=100_000.0, gt=0)
    method: str = Field(default="max_sharpe", description="max_sharpe | min_volatility")


class PortfolioResponse(BaseModel):
    weights: dict[str, float]
    allocation: dict[str, int]
    leftover_cash: float
    performance: dict[str, float]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "StockBot API"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(req: PredictionRequest):
    """
    Run LSTM price prediction for a ticker.
    Load the saved model checkpoint and return multi-step predictions.
    """
    # TODO: Load model from checkpoint, run inference
    # model = load_lstm_model(req.ticker)
    # preds = model.predict(latest_features, steps=req.steps)
    raise HTTPException(status_code=501, detail="Load your trained LSTM model checkpoint to enable predictions.")


@router.post("/signal", response_model=SignalResponse)
async def get_signal(req: SignalRequest):
    """
    Get a composite BUY/SELL/HOLD signal for a ticker.
    Combines LSTM + RL + sentiment.
    """
    # TODO: Run all three models and call SignalGenerator.generate()
    raise HTTPException(status_code=501, detail="Load your trained models to enable signal generation.")


@router.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(req: PortfolioRequest):
    """
    Run Markowitz portfolio optimization on the given tickers.
    """
    # TODO: Load prices, instantiate PortfolioOptimizer, return weights + allocation
    raise HTTPException(status_code=501, detail="Load historical prices to enable portfolio optimization.")


@router.get("/tickers")
async def list_tickers():
    """Return list of tickers supported by the loaded dataset."""
    # TODO: Read from metadata file
    return {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "SPY"]}
