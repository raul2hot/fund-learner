#!/usr/bin/env python
"""
Performance Analysis Script

Comprehensive analysis of model performance including:
1. Threshold calibration sweep
2. Performance by market regime (volatility)
3. Position sizing based on confidence
4. Enhanced metrics and reporting
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sph_net.config import SPHNetConfig
from sph_net.models.sph_net import SPHNet
from sph_net.models.two_stage import TwoStageModel
from data.dataset import TradingDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Results from a threshold evaluation."""
    threshold: float
    n_trades: int
    trade_frequency: float
    win_rate: float
    avg_return: float
    total_return: float
    survival_rate: float
    correct_class_rate: float
    sharpe_ratio: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading models.

    Features:
    - Threshold calibration sweep
    - Regime-based performance analysis
    - Position sizing simulation
    - Enhanced metrics
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

        # Collect all predictions once
        self.predictions_df = self._collect_predictions()

    @torch.no_grad()
    def _collect_predictions(self) -> pd.DataFrame:
        """Collect all model predictions with probabilities."""
        all_data = []

        for batch in self.test_loader:
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            next_mae_long = batch['next_mae_long'].to(self.device)
            next_mae_short = batch['next_mae_short'].to(self.device)

            outputs = self.model(prices, features)

            # Get tradeable probability (from two-stage model)
            if 'tradeable_logits' in outputs:
                tradeable_probs = F.softmax(outputs['tradeable_logits'], dim=-1)
                direction_probs = F.softmax(outputs['direction_logits'], dim=-1)
                trade_prob = tradeable_probs[:, 1]  # P(trade)
                long_prob = direction_probs[:, 0]   # P(long | trade)
                short_prob = direction_probs[:, 1]  # P(short | trade)
            else:
                # Standard 5-class model - approximate from logits
                probs = F.softmax(outputs['logits'], dim=-1)
                trade_prob = probs[:, 0] + probs[:, 4]  # P(HIGH_BULL) + P(LOW_BEAR)
                long_prob = probs[:, 0] / (trade_prob + 1e-8)
                short_prob = probs[:, 4] / (trade_prob + 1e-8)

            # 5-class predictions
            class_probs = F.softmax(outputs['logits'], dim=-1)
            preds = torch.argmax(outputs['logits'], dim=-1)

            # Extract volatility from features for regime analysis
            # Assuming features contain volatility info (adjust index as needed)
            volatility = features[:, -1, :].mean(dim=-1)  # Use last timestep

            for i in range(len(preds)):
                all_data.append({
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'trade_prob': trade_prob[i].item(),
                    'long_prob': long_prob[i].item(),
                    'short_prob': short_prob[i].item(),
                    'prob_class_0': class_probs[i, 0].item(),
                    'prob_class_1': class_probs[i, 1].item(),
                    'prob_class_2': class_probs[i, 2].item(),
                    'prob_class_3': class_probs[i, 3].item(),
                    'prob_class_4': class_probs[i, 4].item(),
                    'next_return': next_return[i].item(),
                    'next_mae_long': next_mae_long[i].item(),
                    'next_mae_short': next_mae_short[i].item(),
                    'volatility': volatility[i].item(),
                })

        return pd.DataFrame(all_data)

    def threshold_sweep(
        self,
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Sweep through trade probability thresholds.

        Returns DataFrame with performance at each threshold.
        """
        if thresholds is None:
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        results = []
        df = self.predictions_df.copy()

        for thresh in thresholds:
            result = self._evaluate_threshold(df, thresh)
            results.append(result)

        return pd.DataFrame([vars(r) for r in results])

    def _evaluate_threshold(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> ThresholdResult:
        """Evaluate performance at a specific threshold."""

        # Filter to trades above threshold
        trades = df[df['trade_prob'] >= threshold].copy()
        n_trades = len(trades)

        if n_trades == 0:
            return ThresholdResult(
                threshold=threshold,
                n_trades=0,
                trade_frequency=0.0,
                win_rate=0.0,
                avg_return=0.0,
                total_return=0.0,
                survival_rate=0.0,
                correct_class_rate=0.0,
                sharpe_ratio=0.0
            )

        # Determine direction based on which has higher probability
        trades['direction'] = np.where(trades['long_prob'] > trades['short_prob'], 'long', 'short')

        # Calculate returns based on direction
        trades['trade_return'] = np.where(
            trades['direction'] == 'long',
            trades['next_return'],
            -trades['next_return']
        )

        # Calculate MAE based on direction
        trades['trade_mae'] = np.where(
            trades['direction'] == 'long',
            trades['next_mae_long'],
            trades['next_mae_short']
        )

        # Correct class: true label matches predicted direction
        trades['correct_class'] = np.where(
            trades['direction'] == 'long',
            trades['true_label'] == 0,  # HIGH_BULL
            trades['true_label'] == 4   # LOW_BEAR
        )

        # Metrics
        win_rate = (trades['trade_return'] > 0).mean() * 100
        avg_return = trades['trade_return'].mean() * 100
        total_return = trades['trade_return'].sum() * 100
        survival_rate = (trades['trade_mae'] < 0.005).mean() * 100
        correct_class_rate = trades['correct_class'].mean() * 100

        # Sharpe ratio (annualized based on actual trade frequency)
        # Note: sqrt(35000) assumes trading every 15-min candle, which overstates Sharpe
        # Use actual trades per year for proper annualization
        if trades['trade_return'].std() > 0:
            # Estimate based on ~96 candles/day of test data
            trading_days = len(df) / 96
            trades_per_year = n_trades * (365 / max(trading_days, 1))
            sharpe = (trades['trade_return'].mean() / trades['trade_return'].std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        return ThresholdResult(
            threshold=threshold,
            n_trades=n_trades,
            trade_frequency=n_trades / len(df) * 100,
            win_rate=win_rate,
            avg_return=avg_return,
            total_return=total_return,
            survival_rate=survival_rate,
            correct_class_rate=correct_class_rate,
            sharpe_ratio=sharpe
        )

    def analyze_by_regime(self) -> pd.DataFrame:
        """
        Analyze performance by volatility regime.

        Splits data into low/medium/high volatility regimes.
        """
        df = self.predictions_df.copy()

        # Create volatility regimes based on percentiles
        vol_33 = df['volatility'].quantile(0.33)
        vol_66 = df['volatility'].quantile(0.66)

        df['vol_regime'] = pd.cut(
            df['volatility'],
            bins=[-np.inf, vol_33, vol_66, np.inf],
            labels=['low_vol', 'med_vol', 'high_vol']
        )

        results = []

        for regime in ['low_vol', 'med_vol', 'high_vol']:
            regime_df = df[df['vol_regime'] == regime].copy()

            # Get trades (using default 0.5 threshold)
            trades = regime_df[regime_df['trade_prob'] >= 0.5]

            if len(trades) == 0:
                continue

            # Direction
            trades = trades.copy()
            trades['direction'] = np.where(trades['long_prob'] > trades['short_prob'], 'long', 'short')
            trades['trade_return'] = np.where(
                trades['direction'] == 'long',
                trades['next_return'],
                -trades['next_return']
            )
            trades['trade_mae'] = np.where(
                trades['direction'] == 'long',
                trades['next_mae_long'],
                trades['next_mae_short']
            )

            results.append({
                'regime': regime,
                'n_samples': len(regime_df),
                'n_trades': len(trades),
                'trade_frequency': len(trades) / len(regime_df) * 100,
                'win_rate': (trades['trade_return'] > 0).mean() * 100,
                'avg_return': trades['trade_return'].mean() * 100,
                'total_return': trades['trade_return'].sum() * 100,
                'survival_rate': (trades['trade_mae'] < 0.005).mean() * 100,
            })

        return pd.DataFrame(results)

    def simulate_position_sizing(
        self,
        base_threshold: float = 0.5,
        max_position: float = 1.0
    ) -> Dict:
        """
        Simulate trading with position sizing based on confidence.

        Position size = (trade_prob - threshold) * scaling factor
        """
        df = self.predictions_df.copy()

        # Only consider samples above threshold
        trades = df[df['trade_prob'] >= base_threshold].copy()

        if len(trades) == 0:
            return {'total_return': 0.0, 'n_trades': 0}

        # Calculate confidence-based position size
        # Scale from 0 to max_position
        min_prob = base_threshold
        max_prob = 1.0
        trades['position_size'] = ((trades['trade_prob'] - min_prob) / (max_prob - min_prob)) * max_position

        # Direction confidence also matters
        trades['direction_confidence'] = np.abs(trades['long_prob'] - 0.5) * 2

        # Combined position sizing
        trades['final_position'] = trades['position_size'] * (0.7 + 0.3 * trades['direction_confidence'])
        trades['final_position'] = trades['final_position'].clip(0, max_position)

        # Direction and returns
        trades['direction'] = np.where(trades['long_prob'] > trades['short_prob'], 'long', 'short')
        trades['base_return'] = np.where(
            trades['direction'] == 'long',
            trades['next_return'],
            -trades['next_return']
        )

        # Position-sized returns
        trades['sized_return'] = trades['base_return'] * trades['final_position']

        # Compare with equal sizing
        equal_return = trades['base_return'].sum() * 100
        sized_return = trades['sized_return'].sum() * 100

        return {
            'equal_sizing': {
                'total_return': equal_return,
                'avg_return': trades['base_return'].mean() * 100,
            },
            'confidence_sizing': {
                'total_return': sized_return,
                'avg_return': trades['sized_return'].mean() * 100,
                'avg_position': trades['final_position'].mean(),
            },
            'n_trades': len(trades),
            'improvement': sized_return - equal_return,
        }

    def find_optimal_threshold(self) -> Tuple[float, ThresholdResult]:
        """Find the threshold that maximizes risk-adjusted returns."""
        sweep_df = self.threshold_sweep()

        # Filter to thresholds with at least 50 trades
        viable = sweep_df[sweep_df['n_trades'] >= 50]

        if len(viable) == 0:
            return 0.5, self._evaluate_threshold(self.predictions_df, 0.5)

        # Maximize Sharpe ratio
        best_idx = viable['sharpe_ratio'].idxmax()
        best_threshold = viable.loc[best_idx, 'threshold']

        return best_threshold, ThresholdResult(**viable.loc[best_idx].to_dict())

    def generate_analysis_report(self, output_path: Path) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # === THRESHOLD SWEEP ===
        lines.append("1. THRESHOLD CALIBRATION SWEEP")
        lines.append("-" * 60)
        sweep_df = self.threshold_sweep()

        lines.append(f"{'Thresh':>7} {'Trades':>8} {'Freq%':>7} {'Win%':>7} "
                    f"{'AvgRet':>8} {'TotRet':>8} {'Surv%':>7} {'Sharpe':>8}")
        lines.append("-" * 80)

        for _, row in sweep_df.iterrows():
            lines.append(
                f"{row['threshold']:>7.2f} {row['n_trades']:>8} "
                f"{row['trade_frequency']:>7.2f} {row['win_rate']:>7.2f} "
                f"{row['avg_return']:>8.4f} {row['total_return']:>8.2f} "
                f"{row['survival_rate']:>7.2f} {row['sharpe_ratio']:>8.2f}"
            )
        lines.append("")

        # Optimal threshold
        best_thresh, best_result = self.find_optimal_threshold()
        lines.append(f"OPTIMAL THRESHOLD: {best_thresh:.2f}")
        lines.append(f"  - Trades: {best_result.n_trades}")
        lines.append(f"  - Total Return: {best_result.total_return:.2f}%")
        lines.append(f"  - Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
        lines.append("")

        # === REGIME ANALYSIS ===
        lines.append("2. PERFORMANCE BY VOLATILITY REGIME")
        lines.append("-" * 60)
        regime_df = self.analyze_by_regime()

        lines.append(f"{'Regime':>10} {'Samples':>10} {'Trades':>8} {'Freq%':>8} "
                    f"{'Win%':>8} {'AvgRet':>8} {'TotRet':>10}")
        lines.append("-" * 70)

        for _, row in regime_df.iterrows():
            lines.append(
                f"{row['regime']:>10} {row['n_samples']:>10} {row['n_trades']:>8} "
                f"{row['trade_frequency']:>8.2f} {row['win_rate']:>8.2f} "
                f"{row['avg_return']:>8.4f} {row['total_return']:>10.2f}"
            )
        lines.append("")

        # Best regime
        if len(regime_df) > 0:
            best_regime = regime_df.loc[regime_df['avg_return'].idxmax(), 'regime']
            lines.append(f"BEST PERFORMING REGIME: {best_regime}")
            lines.append("")

        # === POSITION SIZING ===
        lines.append("3. POSITION SIZING ANALYSIS")
        lines.append("-" * 60)

        sizing_results = self.simulate_position_sizing()

        lines.append("Equal Sizing (all trades same size):")
        lines.append(f"  - Total Return: {sizing_results['equal_sizing']['total_return']:.2f}%")
        lines.append(f"  - Avg Return/Trade: {sizing_results['equal_sizing']['avg_return']:.4f}%")
        lines.append("")

        lines.append("Confidence-Based Sizing (scale by probability):")
        lines.append(f"  - Total Return: {sizing_results['confidence_sizing']['total_return']:.2f}%")
        lines.append(f"  - Avg Return/Trade: {sizing_results['confidence_sizing']['avg_return']:.4f}%")
        lines.append(f"  - Avg Position Size: {sizing_results['confidence_sizing']['avg_position']:.2f}")
        lines.append("")
        lines.append(f"IMPROVEMENT: {sizing_results['improvement']:+.2f}%")
        lines.append("")

        # === RECOMMENDATIONS ===
        lines.append("4. RECOMMENDATIONS")
        lines.append("-" * 60)

        recommendations = self._generate_recommendations(
            sweep_df, regime_df, sizing_results, best_thresh
        )
        for rec in recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

        lines.append("=" * 80)

        report_text = "\n".join(lines)

        # Save report
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "performance_analysis.txt", 'w') as f:
            f.write(report_text)

        # Save data
        sweep_df.to_csv(output_path / "threshold_sweep.csv", index=False)
        regime_df.to_csv(output_path / "regime_analysis.csv", index=False)

        with open(output_path / "sizing_results.json", 'w') as f:
            json.dump(sizing_results, f, indent=2)

        # Save predictions with probabilities
        self.predictions_df.to_csv(output_path / "predictions_detailed.csv", index=False)

        return report_text

    def _generate_recommendations(
        self,
        sweep_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        sizing_results: Dict,
        best_thresh: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Threshold recommendation
        if best_thresh > 0.5:
            recommendations.append(
                f"Increase trade threshold from 0.50 to {best_thresh:.2f} to filter low-confidence trades"
            )

        # Regime recommendation
        if len(regime_df) > 0:
            best_regime_row = regime_df.loc[regime_df['avg_return'].idxmax()]
            worst_regime_row = regime_df.loc[regime_df['avg_return'].idxmin()]

            if best_regime_row['avg_return'] > worst_regime_row['avg_return'] * 2:
                recommendations.append(
                    f"Consider trading only in {best_regime_row['regime']} conditions "
                    f"(avg return: {best_regime_row['avg_return']:.4f}% vs {worst_regime_row['avg_return']:.4f}%)"
                )

        # Position sizing
        if sizing_results['improvement'] > 1.0:
            recommendations.append(
                "Implement confidence-based position sizing - "
                f"potential improvement of {sizing_results['improvement']:.2f}%"
            )

        # Check survival rate at optimal threshold
        best_result = sweep_df[sweep_df['threshold'] == best_thresh].iloc[0] if len(sweep_df[sweep_df['threshold'] == best_thresh]) > 0 else None
        if best_result is not None and best_result['survival_rate'] < 60:
            recommendations.append(
                f"Survival rate ({best_result['survival_rate']:.1f}%) is below target - "
                "consider tighter stop losses or regime filtering"
            )

        # Trade frequency
        current_freq = sweep_df[sweep_df['threshold'] == 0.5]['trade_frequency'].values[0] if len(sweep_df) > 0 else 0
        if current_freq > 15:
            recommendations.append(
                f"Trade frequency ({current_freq:.1f}%) may be too high - "
                "model may be over-trading"
            )

        if not recommendations:
            recommendations.append("Model performance appears reasonable - continue monitoring")

        return recommendations


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    MODEL_DIR = Path("experiments/run_001")
    OUTPUT_DIR = MODEL_DIR / "analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not (MODEL_DIR / "best_model.pt").exists():
        logger.error(f"Model not found in {MODEL_DIR}")
        logger.error("Please run 'python scripts/train.py' first")
        return

    # === Load Test Data ===
    logger.info("Loading test data...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)

    price_cols = feature_info['price_columns']
    eng_cols = [c for c in feature_info['engineered_columns'] if c in test_df.columns]

    logger.info(f"Test samples: {len(test_df)}")

    # === Load Model ===
    logger.info("Loading model...")
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # Detect model type from config
    model_type = getattr(config, 'model_type', 'standard')
    logger.info(f"Model type: {model_type}")

    if model_type == "two_stage":
        model = TwoStageModel(config)
        logger.info("Loaded Two-Stage Model")
    else:
        model = SPHNet(config)
        logger.info("Loaded Standard 5-class Model")

    model.load_state_dict(checkpoint['model_state_dict'])

    # === Create Test DataLoader ===
    test_dataset = TradingDataset(
        test_df, price_cols, eng_cols,
        window_size=config.window_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    logger.info(f"Test batches: {len(test_loader)}")

    # === Run Analysis ===
    logger.info("\nRunning performance analysis...")
    analyzer = PerformanceAnalyzer(model, test_loader, device=config.device)

    report = analyzer.generate_analysis_report(OUTPUT_DIR)

    print("\n" + report)

    logger.info(f"\nAnalysis results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
