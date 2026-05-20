"""
Stacked LSTM with Bahdanau Attention for multi-step stock price prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention mechanism over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, hidden_size) — last hidden state
            keys:  (batch, seq_len, hidden_size) — all hidden states

        Returns:
            context: (batch, hidden_size)
            weights: (batch, seq_len)
        """
        q = self.W_query(query).unsqueeze(1)          # (B, 1, H)
        k = self.W_key(keys)                           # (B, T, H)
        score = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
        weights = F.softmax(score, dim=-1)             # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, H)
        return context, weights


class LSTMPredictor(nn.Module):
    """
    Stacked LSTM + optional attention for stock price prediction.

    Input:  (batch, sequence_length, input_size)
    Output: (batch, output_steps)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_steps: int = 5,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if use_attention:
            self.attention = BahdanauAttention(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_steps),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions: (batch, output_steps)
            attn_weights: (batch, seq_len) or None
        """
        lstm_out, (h_n, _) = self.lstm(x)
        # lstm_out: (B, T, H), h_n: (num_layers, B, H)

        if self.use_attention:
            last_hidden = h_n[-1]  # (B, H) — top layer
            context, attn_weights = self.attention(last_hidden, lstm_out)
            out = self.layer_norm(context)
        else:
            out = lstm_out[:, -1, :]  # last timestep
            attn_weights = None

        out = self.dropout(out)
        predictions = self.fc(out)
        return predictions, attn_weights

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
