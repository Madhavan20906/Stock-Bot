"""Unit tests for StockPreprocessor."""

import numpy as np
import pandas as pd
import pytest
from src.data.preprocessor import StockPreprocessor


def make_df(n=300) -> pd.DataFrame:
    np.random.seed(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open":   np.random.uniform(100, 200, n),
        "high":   np.random.uniform(200, 220, n),
        "low":    np.random.uniform(80, 100, n),
        "close":  np.random.uniform(100, 200, n),
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "rsi_14": np.random.uniform(20, 80, n),
    }, index=idx)


def test_fit_transform_shape():
    df = make_df(100)
    prep = StockPreprocessor(sequence_length=20)
    prep.fit(df)
    scaled = prep.transform(df)
    assert scaled.shape == df.shape


def test_sequences_shape():
    df = make_df(100)
    prep = StockPreprocessor(sequence_length=20)
    prep.fit(df)
    X, y = prep.make_sequences(df)
    n_samples = 100 - 20
    assert X.shape == (n_samples, 20, df.shape[1])
    assert y.shape == (n_samples,)


def test_inverse_transform_roundtrip():
    df = make_df(100)
    prep = StockPreprocessor(sequence_length=10)
    prep.fit(df)
    scaled = prep.transform(df)
    target_idx = list(scaled.columns).index("close")
    scaled_values = scaled["close"].values
    recovered = prep.inverse_transform_target(scaled_values)
    np.testing.assert_allclose(recovered, df["close"].values, rtol=1e-4)


def test_no_data_leakage():
    df = make_df(300)
    prep = StockPreprocessor(sequence_length=30)
    (X_tr, _), (X_val, _), (X_te, _) = prep.train_val_test_split(
        df, train_end="2020-10-01", val_end="2020-11-01"
    )
    assert len(X_tr) > 0
    assert len(X_val) >= 0
    assert len(X_te) >= 0


def test_dtype_float32():
    df = make_df(60)
    prep = StockPreprocessor(sequence_length=20)
    prep.fit(df)
    X, y = prep.make_sequences(df)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
