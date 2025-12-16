"""
Prediction Heads for SPH-Net

Updated for 5-class classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    5-Class classification head.

    Outputs logits for each class:
    0: HIGH_BULL
    1: BULL
    2: RANGE_BOUND
    3: BEAR
    4: LOW_BEAR
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]
        Returns:
            logits: [batch, n_classes]
        """
        return self.head(x)


class AuxiliaryRegressionHead(nn.Module):
    """
    Auxiliary head for return prediction.

    Helps with feature learning - predicts expected return.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]
        Returns:
            return_pred: [batch, 1]
        """
        return self.head(x).squeeze(-1)


# Keep legacy heads for backwards compatibility
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
        return self.head(x)


class UncertaintyHead(nn.Module):
    """Predicts aleatoric uncertainty (variance)"""

    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
            nn.Softplus()
        )

    def forward(self, x):
        return self.head(x)
