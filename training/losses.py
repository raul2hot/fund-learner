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


class TradingAwareLoss(nn.Module):
    """
    Loss that specifically penalizes trading-relevant mistakes.

    Key insight: Not all errors are equal for trading:
    - Missing a good trade (false negative for tradeable) = lost opportunity
    - Taking a bad trade (false positive for tradeable) = potential loss
    - Wrong direction on tradeable = actual loss

    This loss applies asymmetric penalties to focus the model on what matters.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        fn_penalty: float = 3.0,  # Missing tradeable (0 or 4) predicted as 2
        fp_penalty: float = 1.5,  # Predicted tradeable but actually range-bound
        direction_penalty: float = 2.0  # Predicted wrong tradeable direction
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        self.direction_penalty = direction_penalty

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: [batch, n_classes] logits
            targets: [batch] class indices (0=HIGH_BULL, 4=LOW_BEAR, 2=RANGE_BOUND)
        """
        # Base focal loss (per sample)
        base_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-base_loss)
        focal_loss = ((1 - pt) ** 2.0) * base_loss

        preds = inputs.argmax(dim=1)

        # Identify tradeable classes (0 = HIGH_BULL, 4 = LOW_BEAR)
        is_tradeable = (targets == 0) | (targets == 4)
        pred_tradeable = (preds == 0) | (preds == 4)
        is_range = (targets == 2)
        pred_range = (preds == 2)

        # False negatives: tradeable candle predicted as range
        fn_mask = is_tradeable & pred_range

        # False positives: range candle predicted as tradeable
        fp_mask = is_range & pred_tradeable

        # Direction errors: predicted tradeable but wrong direction
        # E.g., predicted 0 (LONG) but actually 4 (SHORT)
        direction_error = (
            ((preds == 0) & (targets == 4)) |
            ((preds == 4) & (targets == 0))
        )

        # Apply penalties
        loss = focal_loss.clone()
        loss[fn_mask] = loss[fn_mask] * self.fn_penalty
        loss[fp_mask] = loss[fp_mask] * self.fp_penalty
        loss[direction_error] = loss[direction_error] * self.direction_penalty

        return loss.mean()


class TradingLoss(nn.Module):
    """
    Combined loss for trading classification.

    Components:
    1. Focal loss for classification (handles imbalance)
    2. MSE loss for auxiliary return prediction
    3. Optional trading-aware penalties
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.1,
        use_trading_aware: bool = False,
        fn_penalty: float = 3.0,
        fp_penalty: float = 1.5,
        direction_penalty: float = 2.0
    ):
        super().__init__()

        self.use_trading_aware = use_trading_aware

        if use_trading_aware:
            self.cls_loss = TradingAwareLoss(
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                fn_penalty=fn_penalty,
                fp_penalty=fp_penalty,
                direction_penalty=direction_penalty
            )
        else:
            self.cls_loss = FocalLoss(
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
        if self.use_trading_aware:
            cls_loss = self.cls_loss(outputs['logits'], targets)
        else:
            cls_loss = self.cls_loss(outputs['logits'], targets)

        # Auxiliary regression loss
        aux_loss = self.mse_loss(outputs['return_pred'], next_return)

        # Total loss
        total_loss = cls_loss + self.aux_weight * aux_loss

        return {
            'total': total_loss,
            'classification': cls_loss,
            'auxiliary': aux_loss
        }
