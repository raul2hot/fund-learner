"""Temporal and Feature Encoders for SPH-Net"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Transformer-based encoder for price/return sequences.
    Input: [batch, T, P_price]
    Output: [batch, T, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, T, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x  # [batch, T, d_model]


class FeatureEncoder(nn.Module):
    """
    MLP-based encoder for engineered features.
    Input: [batch, T, P_feat]
    Output: [batch, T, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # x: [batch, T, input_dim]
        return self.encoder(x)  # [batch, T, d_model]
