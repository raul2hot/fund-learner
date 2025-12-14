"""SPH-Net: Hybrid Transformer for Financial Time Series"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead


class SPHNet(nn.Module):
    """
    SPH-Net Hybrid Transformer

    Architecture:
    1. Temporal Encoder (Transformer) for price/returns
    2. Feature Encoder (MLP) for engineered features
    3. Co-Attention Fusion
    4. Prediction Heads (regression, classification, uncertainty)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=config.engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Optional: Additional transformer block after fusion
        self.decoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        # Prediction heads
        self.regression_head = RegressionHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.classification_head = ClassificationHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.uncertainty_head = UncertaintyHead(
            config.d_model, config.forecast_horizon, config.dropout
        )

    def forward(self, prices, features):
        """
        Args:
            prices: [batch, T, P_price] - OHLCV or returns
            features: [batch, T, P_feat] - Engineered features

        Returns:
            dict with 'return_pred', 'direction_pred', 'uncertainty'
        """
        # Encode
        temporal_tokens = self.temporal_encoder(prices)      # [batch, T, d_model]
        feature_tokens = self.feature_encoder(features)      # [batch, T, d_model]

        # Fuse with co-attention
        fused = self.co_attention(temporal_tokens, feature_tokens)  # [batch, T, d_model]

        # Decode
        decoded = self.decoder(fused)  # [batch, T, d_model]

        # Use last token for prediction
        last_token = decoded[:, -1, :]  # [batch, d_model]

        # Predictions
        return_pred = self.regression_head(last_token)
        direction_pred = self.classification_head(last_token)
        uncertainty = self.uncertainty_head(last_token)

        return {
            'return_pred': return_pred,
            'direction_pred': direction_pred,
            'uncertainty': uncertainty
        }
