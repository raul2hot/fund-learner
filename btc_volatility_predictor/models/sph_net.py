"""SPH-Net: Hybrid Transformer for BTC Volatility Prediction"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead


class SPHNet(nn.Module):
    """
    SPH-Net Hybrid Transformer for Volatility Prediction

    Architecture:
    1. Temporal Encoder (Transformer) for price sequences
    2. Feature Encoder (MLP) for engineered features
    3. Co-Attention Fusion
    4. Prediction Heads (volatility regression, direction, uncertainty)
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

        # Additional transformer block after fusion
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )

        # Temporal attention pooling (instead of just last token)
        self.temporal_pool = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Softmax(dim=1)
        )

        # Prediction heads
        self.regression_head = RegressionHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.classification_head = ClassificationHead(
            config.d_model, config.forecast_horizon, config.dropout
        )

        if config.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                config.d_model, config.forecast_horizon, config.dropout
            )
        else:
            self.uncertainty_head = None

    def forward(self, prices, features):
        """
        Args:
            prices: [batch, T, n_price_features]
            features: [batch, T, n_engineered_features]

        Returns:
            dict with 'volatility_pred', 'direction_pred', 'uncertainty'
        """
        # Encode both streams
        temporal_tokens = self.temporal_encoder(prices)      # [batch, T, d_model]
        feature_tokens = self.feature_encoder(features)      # [batch, T, d_model]

        # Fuse with co-attention
        fused = self.co_attention(temporal_tokens, feature_tokens)  # [batch, T, d_model]

        # Decode
        decoded = self.decoder(fused)  # [batch, T, d_model]

        # Attention-weighted pooling (focus on important timesteps)
        attn_weights = self.temporal_pool(decoded)  # [batch, T, 1]
        pooled = (decoded * attn_weights).sum(dim=1)  # [batch, d_model]

        # Also keep last token for comparison
        last_token = decoded[:, -1, :]  # [batch, d_model]

        # Combine pooled and last token
        combined = pooled + last_token  # Simple residual combination

        # Predictions
        volatility_pred = self.regression_head(combined)
        direction_pred = self.classification_head(combined)

        output = {
            'volatility_pred': volatility_pred,
            'direction_pred': direction_pred,
        }

        if self.uncertainty_head is not None:
            output['uncertainty'] = self.uncertainty_head(combined)

        return output
