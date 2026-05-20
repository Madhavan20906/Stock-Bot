"""
Data preprocessing: normalization, sequence generation, train/val/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from loguru import logger


class StockPreprocessor:
    """Scales features and creates sliding-window sequences for LSTM."""

    def __init__(self, sequence_length: int = 60, target_col: str = "close"):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "StockPreprocessor":
        """Fit scaler on training data."""
        self.feature_columns = [c for c in df.columns if c != self.target_col]
        self.scaler.fit(df[self.feature_columns + [self.target_col]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features."""
        cols = self.feature_columns + [self.target_col]
        scaled = self.scaler.transform(df[cols])
        return pd.DataFrame(scaled, columns=cols, index=df.index)

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Inverse-scale predicted close prices."""
        n_features = len(self.feature_columns) + 1
        target_idx = n_features - 1
        dummy = np.zeros((len(values), n_features))
        dummy[:, target_idx] = values
        return self.scaler.inverse_transform(dummy)[:, target_idx]

    def make_sequences(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate (X, y) sliding-window sequences.

        Returns:
            X: shape (n_samples, sequence_length, n_features)
            y: shape (n_samples,) — next day's scaled close price
        """
        scaled = self.transform(df)
        data = scaled.values
        target_idx = list(scaled.columns).index(self.target_col)

        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, target_idx])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
    ) -> tuple:
        """Time-aware split — no data leakage.

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        train_df = df[df.index <= train_end]
        val_df = df[(df.index > train_end) & (df.index <= val_end)]
        test_df = df[df.index > val_end]

        self.fit(train_df)  # Fit only on train

        logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

        return (
            self.make_sequences(train_df),
            self.make_sequences(val_df),
            self.make_sequences(test_df),
        )
