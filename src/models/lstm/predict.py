"""
LSTM inference: load a saved checkpoint and produce multi-step price predictions.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.models.lstm.model import LSTMPredictor
from src.data.preprocessor import StockPreprocessor
from src.data.feature_store import build_features


def load_model(ticker: str, checkpoint_dir: str = "checkpoints/lstm") -> LSTMPredictor:
    """Load the best saved LSTM checkpoint for a ticker."""
    path = Path(checkpoint_dir) / f"best_{ticker}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {path}. "
            f"Train the model first: python src/models/lstm/train.py --ticker {ticker}"
        )
    # We reconstruct model from saved state — shape inferred at load time
    # In production you'd also save the config; here we load and infer
    state = torch.load(path, map_location="cpu")

    # Infer input_size from first LSTM weight
    input_size = state["lstm.weight_ih_l0"].shape[1]
    hidden_size = state["lstm.weight_ih_l0"].shape[0] // 4

    # Count layers
    num_layers = sum(1 for k in state if k.startswith("lstm.weight_ih_l") and "reverse" not in k)
    output_steps = state["fc.3.weight"].shape[0]
    use_attention = "attention.v.weight" in state

    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_steps=output_steps,
        use_attention=use_attention,
    )
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded LSTM for {ticker}: input={input_size}, hidden={hidden_size}, layers={num_layers}, steps={output_steps}")
    return model


@torch.no_grad()
def predict_next_n(
    ticker: str,
    config: dict,
    checkpoint_dir: str = "checkpoints/lstm",
) -> dict:
    """
    Run inference for the most recent window of data.

    Returns:
        {
            "ticker": str,
            "predictions": list[float],   # predicted prices (inverse-scaled)
            "attention_weights": list[float] | None,
            "last_known_price": float,
            "direction": "UP" | "DOWN" | "FLAT",
            "confidence": float,
        }
    """
    model = load_model(ticker, checkpoint_dir)
    df = build_features(ticker, config)

    seq_len = config["data"]["sequence_length"]
    prep = StockPreprocessor(sequence_length=seq_len, target_col="close")
    prep.fit(df)

    # Use the last `seq_len` rows as the input window
    window = df.tail(seq_len + 1).iloc[:-1]  # exclude today (future)
    scaled = prep.transform(window)
    X = torch.tensor(scaled.values[np.newaxis, :, :], dtype=torch.float32)  # (1, T, F)

    preds, attn = model(X)
    preds_np = preds.squeeze(0).numpy()

    # Inverse-scale
    raw_preds = prep.inverse_transform_target(preds_np)
    last_price = float(df["close"].iloc[-1])

    direction_score = raw_preds[0] - last_price
    if abs(direction_score) < last_price * 0.002:
        direction = "FLAT"
    elif direction_score > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    confidence = float(np.clip(abs(direction_score) / (last_price * 0.05), 0, 1))

    return {
        "ticker": ticker,
        "predictions": raw_preds.tolist(),
        "attention_weights": attn.squeeze(0).tolist() if attn is not None else None,
        "last_known_price": last_price,
        "direction": direction,
        "confidence": confidence,
    }
