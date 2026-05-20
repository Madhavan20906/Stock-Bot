"""Unit tests for LSTM model architecture."""

import torch
import pytest
from src.models.lstm.model import LSTMPredictor, BahdanauAttention


def test_lstm_output_shape():
    model = LSTMPredictor(input_size=16, hidden_size=64, num_layers=2, output_steps=5)
    x = torch.randn(8, 30, 16)  # (batch=8, seq=30, features=16)
    preds, attn = model(x)
    assert preds.shape == (8, 5)
    assert attn is not None
    assert attn.shape == (8, 30)


def test_lstm_no_attention():
    model = LSTMPredictor(input_size=16, hidden_size=64, output_steps=3, use_attention=False)
    x = torch.randn(4, 20, 16)
    preds, attn = model(x)
    assert preds.shape == (4, 3)
    assert attn is None


def test_attention_weights_sum_to_one():
    model = LSTMPredictor(input_size=8, hidden_size=32, use_attention=True)
    x = torch.randn(2, 10, 8)
    _, attn = model(x)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_parameter_count():
    model = LSTMPredictor(input_size=32, hidden_size=128, num_layers=3)
    assert model.num_parameters > 0


def test_gradient_flow():
    model = LSTMPredictor(input_size=4, hidden_size=16, num_layers=1, output_steps=1)
    x = torch.randn(2, 10, 4)
    y = torch.randn(2, 1)
    preds, _ = model(x)
    loss = torch.nn.functional.mse_loss(preds, y)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
