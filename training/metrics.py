"""
Training Metrics

Standard classification metrics + trading-specific metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List


class MetricTracker:
    """Tracks and computes metrics during training."""

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        self.all_returns = []
        self.all_mae_long = []
        self.all_mae_short = []
        self.losses = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor = None,
        loss: float = None,
        next_return: torch.Tensor = None,
        next_mae_long: torch.Tensor = None,
        next_mae_short: torch.Tensor = None
    ):
        """Add batch results."""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
        if loss is not None:
            self.losses.append(loss)
        if next_return is not None:
            self.all_returns.extend(next_return.cpu().numpy())
        if next_mae_long is not None:
            self.all_mae_long.extend(next_mae_long.cpu().numpy())
        if next_mae_short is not None:
            self.all_mae_short.extend(next_mae_short.cpu().numpy())

    def compute(self) -> Dict:
        """Compute all metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=list(range(self.n_classes)), zero_division=0
        )

        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']
        for i, name in enumerate(class_names):
            metrics[f'{name}_precision'] = precision[i]
            metrics[f'{name}_recall'] = recall[i]
            metrics[f'{name}_f1'] = f1[i]
            metrics[f'{name}_support'] = int(support[i])

        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)

        # Tradeable class accuracy (classes 0 and 4)
        tradeable_mask = np.isin(targets, [0, 4])
        if tradeable_mask.sum() > 0:
            metrics['tradeable_accuracy'] = accuracy_score(
                targets[tradeable_mask], preds[tradeable_mask]
            )
        else:
            metrics['tradeable_accuracy'] = 0.0

        # Average loss
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            targets, preds, labels=list(range(self.n_classes))
        )

        return metrics

    def compute_trading_metrics(self) -> Dict:
        """
        Compute trading-specific metrics.

        These evaluate whether the model's predictions
        would lead to profitable trades.
        """
        if not self.all_returns:
            return {}

        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        returns = np.array(self.all_returns)
        mae_long = np.array(self.all_mae_long) if self.all_mae_long else None
        mae_short = np.array(self.all_mae_short) if self.all_mae_short else None

        metrics = {}

        # === LONG TRADES (predicted HIGH_BULL) ===
        long_mask = preds == 0
        if long_mask.sum() > 0:
            long_returns = returns[long_mask]
            long_targets = targets[long_mask]

            metrics['long_trades'] = int(long_mask.sum())
            metrics['long_avg_return'] = float(long_returns.mean() * 100)
            metrics['long_total_return'] = float(long_returns.sum() * 100)
            metrics['long_win_rate'] = float((long_returns > 0).mean() * 100)
            metrics['long_correct_class'] = float((long_targets == 0).mean() * 100)

            if mae_long is not None:
                long_mae = mae_long[long_mask]
                # Survival rate: would not have been stopped at 0.5%
                metrics['long_survival_rate'] = float((long_mae < 0.005).mean() * 100)

        # === SHORT TRADES (predicted LOW_BEAR) ===
        short_mask = preds == 4
        if short_mask.sum() > 0:
            short_returns = -returns[short_mask]  # Invert for shorts
            short_targets = targets[short_mask]

            metrics['short_trades'] = int(short_mask.sum())
            metrics['short_avg_return'] = float(short_returns.mean() * 100)
            metrics['short_total_return'] = float(short_returns.sum() * 100)
            metrics['short_win_rate'] = float((short_returns > 0).mean() * 100)
            metrics['short_correct_class'] = float((short_targets == 4).mean() * 100)

            if mae_short is not None:
                short_mae = mae_short[short_mask]
                metrics['short_survival_rate'] = float((short_mae < 0.005).mean() * 100)

        # === OVERALL TRADING ===
        total_trades = metrics.get('long_trades', 0) + metrics.get('short_trades', 0)
        total_samples = len(preds)

        metrics['trade_frequency'] = float(total_trades / total_samples * 100)

        if total_trades > 0:
            combined_return = (
                metrics.get('long_total_return', 0) +
                metrics.get('short_total_return', 0)
            )
            metrics['combined_total_return'] = combined_return
            metrics['avg_return_per_trade'] = combined_return / total_trades

        return metrics

    def get_classification_report(self) -> str:
        """Get sklearn classification report as string."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']

        return classification_report(
            targets, preds,
            labels=list(range(self.n_classes)),
            target_names=class_names,
            zero_division=0
        )
