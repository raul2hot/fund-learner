"""
Comprehensive Model Evaluation

Produces detailed analysis of model performance.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive model evaluator.

    Produces:
    - Classification metrics (per class and overall)
    - Confusion matrix analysis
    - Trading performance simulation
    - Detailed report
    """

    def __init__(
        self,
        model,
        test_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run full evaluation.

        Returns:
            (metrics_dict, predictions_df)
        """
        tracker = MetricTracker()
        all_predictions = []

        for batch in self.test_loader:
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            next_mae_long = batch['next_mae_long'].to(self.device)
            next_mae_short = batch['next_mae_short'].to(self.device)

            outputs = self.model(prices, features)
            preds = torch.argmax(outputs['logits'], dim=-1)
            probs = torch.softmax(outputs['logits'], dim=-1)

            tracker.update(
                preds=preds,
                targets=labels,
                probs=probs,
                next_return=next_return,
                next_mae_long=next_mae_long,
                next_mae_short=next_mae_short
            )

            # Store predictions for detailed analysis
            for i in range(len(preds)):
                all_predictions.append({
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'prob_class_0': probs[i, 0].item(),
                    'prob_class_1': probs[i, 1].item(),
                    'prob_class_2': probs[i, 2].item(),
                    'prob_class_3': probs[i, 3].item(),
                    'prob_class_4': probs[i, 4].item(),
                    'next_return': next_return[i].item(),
                    'next_mae_long': next_mae_long[i].item(),
                    'next_mae_short': next_mae_short[i].item(),
                })

        metrics = tracker.compute()
        trading_metrics = tracker.compute_trading_metrics()
        metrics.update(trading_metrics)

        predictions_df = pd.DataFrame(all_predictions)

        return metrics, predictions_df

    def generate_report(
        self,
        metrics: Dict,
        predictions_df: pd.DataFrame,
        output_path: Path
    ) -> str:
        """Generate comprehensive evaluation report."""

        lines = []
        lines.append("=" * 70)
        lines.append("SPH-NET EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # === OVERALL METRICS ===
        lines.append("OVERALL CLASSIFICATION METRICS")
        lines.append("-" * 40)
        lines.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        lines.append(f"Macro Precision:    {metrics['macro_precision']:.4f}")
        lines.append(f"Macro Recall:       {metrics['macro_recall']:.4f}")
        lines.append(f"Macro F1:           {metrics['macro_f1']:.4f}")
        lines.append(f"Tradeable Accuracy: {metrics['tradeable_accuracy']:.4f}")
        lines.append("")

        # === PER-CLASS METRICS ===
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 40)
        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']

        header = f"{'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}"
        lines.append(header)
        lines.append("-" * 50)

        for name in class_names:
            prec = metrics[f'{name}_precision']
            rec = metrics[f'{name}_recall']
            f1 = metrics[f'{name}_f1']
            sup = metrics[f'{name}_support']
            lines.append(f"{name:<15} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {sup:>8}")
        lines.append("")

        # === CONFUSION MATRIX ===
        lines.append("CONFUSION MATRIX")
        lines.append("-" * 40)
        cm = metrics['confusion_matrix']
        lines.append("Rows: True | Cols: Predicted")
        lines.append("")

        header = "          " + " ".join([f"{n[:7]:>8}" for n in class_names])
        lines.append(header)

        for i, name in enumerate(class_names):
            row = f"{name[:9]:<10}" + " ".join([f"{cm[i,j]:>8}" for j in range(5)])
            lines.append(row)
        lines.append("")

        # === TRADING METRICS ===
        lines.append("TRADING PERFORMANCE METRICS")
        lines.append("-" * 40)

        if 'trade_frequency' in metrics:
            lines.append(f"Trade Frequency: {metrics['trade_frequency']:.2f}%")
            lines.append("")

            lines.append("LONG TRADES (predicted HIGH_BULL):")
            if 'long_trades' in metrics:
                lines.append(f"  Total Trades:    {metrics['long_trades']}")
                lines.append(f"  Avg Return:      {metrics['long_avg_return']:.4f}%")
                lines.append(f"  Total Return:    {metrics['long_total_return']:.2f}%")
                lines.append(f"  Win Rate:        {metrics['long_win_rate']:.2f}%")
                lines.append(f"  Correct Class:   {metrics['long_correct_class']:.2f}%")
                if 'long_survival_rate' in metrics:
                    lines.append(f"  Survival Rate:   {metrics['long_survival_rate']:.2f}%")
            else:
                lines.append("  No long trades made")
            lines.append("")

            lines.append("SHORT TRADES (predicted LOW_BEAR):")
            if 'short_trades' in metrics:
                lines.append(f"  Total Trades:    {metrics['short_trades']}")
                lines.append(f"  Avg Return:      {metrics['short_avg_return']:.4f}%")
                lines.append(f"  Total Return:    {metrics['short_total_return']:.2f}%")
                lines.append(f"  Win Rate:        {metrics['short_win_rate']:.2f}%")
                lines.append(f"  Correct Class:   {metrics['short_correct_class']:.2f}%")
                if 'short_survival_rate' in metrics:
                    lines.append(f"  Survival Rate:   {metrics['short_survival_rate']:.2f}%")
            else:
                lines.append("  No short trades made")
            lines.append("")

            if 'combined_total_return' in metrics:
                lines.append("COMBINED:")
                lines.append(f"  Total Return:       {metrics['combined_total_return']:.2f}%")
                lines.append(f"  Return per Trade:   {metrics['avg_return_per_trade']:.4f}%")

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save report
        with open(output_path / "evaluation_report.txt", 'w') as f:
            f.write(report_text)

        # Save metrics as JSON
        metrics_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        # Save predictions
        predictions_df.to_csv(output_path / "predictions.csv", index=False)

        return report_text
