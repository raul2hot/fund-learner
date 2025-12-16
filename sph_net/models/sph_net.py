"""
SPH-Net: Hybrid Transformer for Trading Classification

Updated for 5-class classification with auxiliary regression.
"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import ClassificationHead, AuxiliaryRegressionHead


class SPHNet(nn.Module):
    """
    SPH-Net for 5-Class Trading Classification.

    Architecture:
    1. Temporal Encoder (Transformer) - processes OHLCV
    2. Feature Encoder (MLP) - processes engineered features
    3. Co-Attention Fusion - combines both streams
    4. Classification Head - 5-class output
    5. Auxiliary Regression Head - return prediction (optional)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Support both old and new config attribute names
        n_price_features = getattr(config, 'n_price_features', getattr(config, 'price_features', 5))
        n_engineered_features = getattr(config, 'n_engineered_features', getattr(config, 'engineered_features', 10))

        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=n_price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=n_engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Post-fusion transformer layer
        d_feedforward = getattr(config, 'd_feedforward', config.d_model * 4)
        self.decoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=d_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        # Pooling: use last token
        self.pool = nn.Identity()

        # Heads
        n_classes = getattr(config, 'n_classes', 5)
        self.classifier = ClassificationHead(
            config.d_model,
            n_classes=n_classes,
            dropout=config.dropout
        )

        self.aux_regressor = AuxiliaryRegressionHead(
            config.d_model,
            dropout=config.dropout
        )

    def forward(
        self,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> dict:
        """
        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]

        Returns:
            dict with:
            - logits: [batch, n_classes]
            - return_pred: [batch]
        """
        # Encode
        temporal_tokens = self.temporal_encoder(prices)
        feature_tokens = self.feature_encoder(features)

        # Fuse
        fused = self.co_attention(temporal_tokens, feature_tokens)

        # Decode
        decoded = self.decoder(fused)

        # Pool: use last token
        pooled = decoded[:, -1, :]

        # Predictions
        logits = self.classifier(pooled)
        return_pred = self.aux_regressor(pooled)

        return {
            'logits': logits,
            'return_pred': return_pred
        }

    def predict(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        outputs = self.forward(prices, features)
        return torch.argmax(outputs['logits'], dim=-1)

    def predict_proba(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        outputs = self.forward(prices, features)
        return torch.softmax(outputs['logits'], dim=-1)
