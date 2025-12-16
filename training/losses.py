"""
Loss Functions for Trading Classification

Handles class imbalance with focal loss and class weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0, reduces loss for well-classified examples,
    focusing training on hard negatives.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: [batch, n_classes] logits
            targets: [batch] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TradingLoss(nn.Module):
    """
    Combined loss for trading classification.

    Components:
    1. Focal loss for classification (handles imbalance)
    2. MSE loss for auxiliary return prediction
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.1
    ):
        super().__init__()

        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights
        )
        self.mse_loss = nn.MSELoss()
        self.aux_weight = aux_weight

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        next_return: torch.Tensor
    ) -> dict:
        """
        Args:
            outputs: dict with 'logits' and 'return_pred'
            targets: [batch] class labels
            next_return: [batch] actual returns

        Returns:
            dict with individual losses and total
        """
        # Classification loss
        cls_loss = self.focal_loss(outputs['logits'], targets)

        # Auxiliary regression loss
        aux_loss = self.mse_loss(outputs['return_pred'], next_return)

        # Total loss
        total_loss = cls_loss + self.aux_weight * aux_loss

        return {
            'total': total_loss,
            'classification': cls_loss,
            'auxiliary': aux_loss
        }
