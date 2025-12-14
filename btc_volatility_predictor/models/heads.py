"""Prediction heads for SPH-Net"""

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """Predicts continuous return values"""

    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )

    def forward(self, x):
        # x: [batch, d_model] (last token or pooled)
        return self.head(x)


class ClassificationHead(nn.Module):
    """Predicts direction (up/down)"""

    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)  # Binary classification per horizon
        )

    def forward(self, x):
        # x: [batch, d_model]
        return self.head(x)  # Logits, apply sigmoid for probabilities


class UncertaintyHead(nn.Module):
    """Predicts aleatoric uncertainty (variance)"""

    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(self, x):
        return self.head(x)
