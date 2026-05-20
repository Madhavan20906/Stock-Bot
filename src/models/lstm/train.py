"""
LSTM training loop with early stopping, LR scheduling, and MLflow logging.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_ticker
from src.data.preprocessor import StockPreprocessor
from src.features.technical import add_all_indicators
from src.models.lstm.model import LSTMPredictor


def build_dataloaders(
    ticker: str, config: dict
) -> tuple[DataLoader, DataLoader, DataLoader, StockPreprocessor]:
    df = load_ticker(ticker)
    df = add_all_indicators(df, config.get("features", {}).get("technical", {}))

    prep = StockPreprocessor(
        sequence_length=config["data"]["sequence_length"],
        target_col="close",
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.train_val_test_split(
        df,
        train_end=config["data"]["train_end"],
        val_end=config["data"]["val_end"],
    )

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=shuffle)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val, y_val, shuffle=False),
        to_loader(X_test, y_test, shuffle=False),
        prep,
    )


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds, _ = model(X_batch)
        loss = criterion(preds[:, 0], y_batch)   # Next-day prediction
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds, _ = model(X_batch)
        loss = criterion(preds[:, 0], y_batch)
        total_loss += loss.item()
    return total_loss / len(loader)


def train(ticker: str, config: dict):
    lstm_cfg = config["lstm"]
    train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device} for ticker {ticker}")

    train_loader, val_loader, test_loader, prep = build_dataloaders(ticker, config)
    n_features = next(iter(train_loader))[0].shape[-1]

    model = LSTMPredictor(
        input_size=n_features,
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        output_steps=lstm_cfg["output_steps"],
        dropout=lstm_cfg["dropout"],
        use_attention=lstm_cfg["use_attention"],
    ).to(device)

    logger.info(f"Model parameters: {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )
    criterion = nn.HuberLoss()

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
    mlflow.set_experiment(f"{config['logging']['experiment_name']}_lstm_{ticker}")

    with mlflow.start_run(run_name=f"lstm_{ticker}"):
        mlflow.log_params({**lstm_cfg, **train_cfg, "ticker": ticker, "n_features": n_features})

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, train_cfg["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, train_cfg["gradient_clip"])
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / f"best_{ticker}.pt")
                logger.info(f"Epoch {epoch:03d} | train={train_loss:.5f} val={val_loss:.5f} ✓ saved")
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch:03d} | train={train_loss:.5f} val={val_loss:.5f} patience={patience_counter}")

            if patience_counter >= train_cfg["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final test evaluation
        model.load_state_dict(torch.load(checkpoint_dir / f"best_{ticker}.pt"))
        test_loss = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.pytorch.log_model(model, f"lstm_{ticker}")
        logger.info(f"Test loss: {test_loss:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lstm.yaml")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(args.ticker, config)


if __name__ == "__main__":
    main()
