#!/usr/bin/env python
"""
Performance Review Script

Comprehensive verification and analysis including:
1. Sharpe ratio verification with proper calculation
2. Confidence bucket analysis
3. Tail risk analysis (large individual losses)
4. Walk-forward validation framework

Based on user feedback from performance analysis.
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
from datetime import datetime

from sph_net.config import SPHNetConfig
from sph_net.models.two_stage import TwoStageModel, CalibratedTwoStageModel, apply_stop_loss_to_returns
from data.dataset import TradingDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. SHARPE RATIO VERIFICATION
# =============================================================================

def verify_sharpe_calculation(trades_df: pd.DataFrame, trading_days: int = 343) -> Dict:
    """
    Verify Sharpe ratio with multiple calculation methods.

    The original calculation uses sqrt(35000) which assumes trading every 15-min candle.
    With sparse trading (e.g., 320 trades over 343 days), we need to adjust.

    Returns multiple Sharpe calculations for comparison.
    """
    if len(trades_df) == 0:
        return {'error': 'No trades to analyze'}

    returns = trades_df['trade_return']
    n_trades = len(trades_df)

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return {'error': 'Zero standard deviation'}

    results = {
        'n_trades': n_trades,
        'trading_days': trading_days,
        'mean_return': mean_return,
        'std_return': std_return,
        'min_return': returns.min(),
        'max_return': returns.max(),
        'median_return': returns.median(),
    }

    # Method 1: Original calculation (per 15-min candle annualization)
    # 35000 candles/year * sqrt scaling
    sharpe_original = (mean_return / std_return) * np.sqrt(35000)
    results['sharpe_original_15min'] = sharpe_original

    # Method 2: Trade-based annualization
    # If we make N trades over T days, annualized trades = N * (365/T)
    trades_per_year = n_trades * (365 / trading_days)
    sharpe_trade_based = (mean_return / std_return) * np.sqrt(trades_per_year)
    results['sharpe_trade_based'] = sharpe_trade_based
    results['trades_per_year'] = trades_per_year

    # Method 3: Daily Sharpe (aggregate daily returns, then annualize)
    if 'date' in trades_df.columns or 'timestamp' in trades_df.columns:
        date_col = 'date' if 'date' in trades_df.columns else 'timestamp'
        daily_returns = trades_df.groupby(trades_df[date_col].dt.date)['trade_return'].sum()
        if len(daily_returns) > 1:
            daily_mean = daily_returns.mean()
            daily_std = daily_returns.std()
            if daily_std > 0:
                sharpe_daily = (daily_mean / daily_std) * np.sqrt(252)
                results['sharpe_daily'] = sharpe_daily

    # Method 4: Information ratio style (vs benchmark)
    # Assuming 0 benchmark, this equals Sharpe
    results['information_ratio'] = mean_return / std_return

    # Confidence interval for Sharpe (approximate)
    # SE(Sharpe) ≈ sqrt((1 + 0.5*Sharpe^2) / n)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe_trade_based**2) / n_trades)
    results['sharpe_95ci_low'] = sharpe_trade_based - 1.96 * se_sharpe
    results['sharpe_95ci_high'] = sharpe_trade_based + 1.96 * se_sharpe

    return results


# =============================================================================
# 2. CONFIDENCE BUCKET ANALYSIS
# =============================================================================

def analyze_confidence_buckets(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by confidence level buckets.

    Does higher confidence = better returns?
    """
    if len(trades_df) == 0:
        return pd.DataFrame()

    # Define confidence buckets
    buckets = [
        (0.50, 0.55, 'Low (0.50-0.55)'),
        (0.55, 0.60, 'Medium (0.55-0.60)'),
        (0.60, 0.65, 'High (0.60-0.65)'),
        (0.65, 0.70, 'Very High (0.65-0.70)'),
        (0.70, 0.80, 'Extreme (0.70-0.80)'),
        (0.80, 1.00, 'Ultra (0.80-1.00)'),
    ]

    # Use trade_prob or direction_confidence for bucketing
    conf_col = 'trade_prob' if 'trade_prob' in trades_df.columns else 'direction_confidence'

    results = []
    for low, high, name in buckets:
        mask = (trades_df[conf_col] >= low) & (trades_df[conf_col] < high)
        subset = trades_df[mask]

        if len(subset) > 0:
            results.append({
                'bucket': name,
                'range': f'{low:.2f}-{high:.2f}',
                'n_trades': len(subset),
                'pct_of_total': len(subset) / len(trades_df) * 100,
                'win_rate': (subset['trade_return'] > 0).mean() * 100,
                'avg_return': subset['trade_return'].mean() * 100,
                'std_return': subset['trade_return'].std() * 100,
                'total_return': subset['trade_return'].sum() * 100,
                'return_per_trade': subset['trade_return'].mean() * 100,
                'best_trade': subset['trade_return'].max() * 100,
                'worst_trade': subset['trade_return'].min() * 100,
            })

    return pd.DataFrame(results)


# =============================================================================
# 3. TAIL RISK ANALYSIS
# =============================================================================

def analyze_tail_risk(trades_df: pd.DataFrame) -> Dict:
    """
    Analyze tail risk and large individual losses.

    Key questions:
    - How many trades needed to recover from worst loss?
    - What's the max drawdown within trades?
    - Are there patterns in large losses?
    """
    if len(trades_df) == 0:
        return {'error': 'No trades to analyze'}

    returns = trades_df['trade_return']

    # Basic statistics
    results = {
        'n_trades': len(trades_df),
        'mean_return': returns.mean() * 100,
        'std_return': returns.std() * 100,
    }

    # Percentile analysis
    for pct in [1, 5, 10, 25, 75, 90, 95, 99]:
        results[f'pct_{pct}'] = np.percentile(returns, pct) * 100

    # Worst losses
    worst_5 = returns.nsmallest(5)
    results['worst_5_trades'] = [r * 100 for r in worst_5.values]

    # Best gains
    best_5 = returns.nlargest(5)
    results['best_5_trades'] = [r * 100 for r in best_5.values]

    # Trades to recover from worst loss
    if returns.mean() > 0:
        worst_loss = returns.min()
        avg_gain = returns.mean()
        trades_to_recover = abs(worst_loss / avg_gain)
        results['trades_to_recover_worst'] = trades_to_recover
    else:
        results['trades_to_recover_worst'] = float('inf')

    # Maximum drawdown within cumulative trades
    cumulative = returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    results['max_drawdown'] = drawdown.min() * 100
    results['max_drawdown_trades'] = drawdown.idxmin()

    # Value at Risk (VaR) and Expected Shortfall (CVaR)
    results['var_95'] = np.percentile(returns, 5) * 100  # 95% VaR
    results['var_99'] = np.percentile(returns, 1) * 100  # 99% VaR

    # Expected Shortfall (average of losses below VaR)
    var_95_threshold = np.percentile(returns, 5)
    tail_losses = returns[returns <= var_95_threshold]
    if len(tail_losses) > 0:
        results['cvar_95'] = tail_losses.mean() * 100

    # Skewness and Kurtosis
    results['skewness'] = returns.skew()
    results['kurtosis'] = returns.kurtosis()  # Excess kurtosis

    # Loss analysis
    losses = returns[returns < 0]
    gains = returns[returns > 0]

    results['n_losses'] = len(losses)
    results['n_gains'] = len(gains)
    results['avg_loss'] = losses.mean() * 100 if len(losses) > 0 else 0
    results['avg_gain'] = gains.mean() * 100 if len(gains) > 0 else 0

    # Profit factor = gross profits / gross losses
    if len(losses) > 0:
        gross_profit = gains.sum()
        gross_loss = abs(losses.sum())
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win-loss ratio (expectancy)
    if len(losses) > 0 and len(gains) > 0:
        win_rate = len(gains) / len(returns)
        avg_win = gains.mean()
        avg_loss = abs(losses.mean())
        results['expectancy_ratio'] = (win_rate * avg_win) / ((1 - win_rate) * avg_loss)

    return results


def identify_loss_patterns(trades_df: pd.DataFrame, threshold: float = -0.01) -> pd.DataFrame:
    """
    Identify patterns in large losing trades.

    Args:
        trades_df: DataFrame with trade data
        threshold: Loss threshold (e.g., -0.01 = -1%)

    Returns:
        DataFrame with large losses and their characteristics
    """
    large_losses = trades_df[trades_df['trade_return'] < threshold].copy()

    if len(large_losses) == 0:
        return pd.DataFrame()

    # Add analysis columns
    large_losses['loss_pct'] = large_losses['trade_return'] * 100

    # If direction info available
    if 'is_long' in trades_df.columns:
        large_losses['direction'] = large_losses['is_long'].map({True: 'LONG', False: 'SHORT'})

    # Confidence at time of loss
    if 'trade_prob' in trades_df.columns:
        large_losses['confidence'] = large_losses['trade_prob']

    return large_losses.sort_values('trade_return')


# =============================================================================
# 4. WALK-FORWARD VALIDATION FRAMEWORK
# =============================================================================

@dataclass
class WalkForwardPeriod:
    """Definition of a walk-forward test period."""
    name: str
    train_end: str  # Train on all data before this
    test_start: str
    test_end: str
    description: str


# Historical market regime periods for BTC
MARKET_PERIODS = [
    WalkForwardPeriod(
        name='covid_crash',
        train_end='2020-03-01',
        test_start='2020-03-01',
        test_end='2020-06-01',
        description='COVID-19 Market Crash'
    ),
    WalkForwardPeriod(
        name='may_2021_crash',
        train_end='2021-05-01',
        test_start='2021-05-01',
        test_end='2021-08-01',
        description='May 2021 BTC Crash'
    ),
    WalkForwardPeriod(
        name='luna_3ac',
        train_end='2022-05-01',
        test_start='2022-05-01',
        test_end='2022-08-01',
        description='Luna/3AC Collapse'
    ),
    WalkForwardPeriod(
        name='ftx_crash',
        train_end='2022-11-01',
        test_start='2022-11-01',
        test_end='2023-02-01',
        description='FTX Collapse'
    ),
    WalkForwardPeriod(
        name='etf_rally',
        train_end='2024-01-01',
        test_start='2024-01-01',
        test_end='2024-04-01',
        description='ETF Approval Rally'
    ),
]


def run_walk_forward_validation(
    data_df: pd.DataFrame,
    model_class,
    config,
    periods: List[WalkForwardPeriod] = None,
    min_train_samples: int = 10000
) -> pd.DataFrame:
    """
    Run walk-forward validation across multiple market periods.

    Args:
        data_df: Full dataset with timestamp/datetime index
        model_class: Model class to instantiate
        config: Model configuration
        periods: List of WalkForwardPeriod to test
        min_train_samples: Minimum samples required for training

    Returns:
        DataFrame with results for each period
    """
    if periods is None:
        periods = MARKET_PERIODS

    # Ensure datetime index
    if not isinstance(data_df.index, pd.DatetimeIndex):
        if 'timestamp' in data_df.columns:
            data_df = data_df.set_index('timestamp')
        elif 'datetime' in data_df.columns:
            data_df = data_df.set_index('datetime')

    results = []

    for period in periods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {period.description}")
        logger.info(f"Period: {period.test_start} to {period.test_end}")

        try:
            # Split data
            train_data = data_df[data_df.index < period.train_end]
            test_data = data_df[
                (data_df.index >= period.test_start) &
                (data_df.index < period.test_end)
            ]

            if len(train_data) < min_train_samples:
                logger.warning(f"Insufficient training data: {len(train_data)} < {min_train_samples}")
                results.append({
                    'period': period.name,
                    'description': period.description,
                    'status': 'SKIPPED',
                    'reason': 'Insufficient training data',
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                })
                continue

            if len(test_data) == 0:
                logger.warning(f"No test data for period")
                results.append({
                    'period': period.name,
                    'description': period.description,
                    'status': 'SKIPPED',
                    'reason': 'No test data',
                    'train_samples': len(train_data),
                    'test_samples': 0,
                })
                continue

            logger.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

            # Note: Actual training would go here
            # For now, we just report data availability
            results.append({
                'period': period.name,
                'description': period.description,
                'test_start': period.test_start,
                'test_end': period.test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'status': 'READY',
            })

        except Exception as e:
            logger.error(f"Error processing period {period.name}: {e}")
            results.append({
                'period': period.name,
                'description': period.description,
                'status': 'ERROR',
                'error': str(e),
            })

    return pd.DataFrame(results)


# =============================================================================
# 5. RISK MANAGEMENT SUGGESTIONS
# =============================================================================

def suggest_stop_loss(trades_df: pd.DataFrame, atr_column: str = None) -> Dict:
    """
    Suggest stop-loss levels based on historical trade data.

    Uses the apply_stop_loss_to_returns function for accurate simulation.

    Returns suggested stop-loss levels at various confidence intervals.
    """
    if len(trades_df) == 0:
        return {'error': 'No trades to analyze'}

    returns = trades_df['trade_return'].values
    mae_values = trades_df['trade_mae'].values if 'trade_mae' in trades_df.columns else None

    # Calculate percentile-based stop levels
    stop_levels = {
        'conservative': np.percentile(returns, 5),    # 5th percentile
        'moderate': np.percentile(returns, 2.5),      # 2.5th percentile
        'aggressive': np.percentile(returns, 1),      # 1st percentile
    }

    suggestions = {}

    for level_name, stop_pct in stop_levels.items():
        # Use the apply_stop_loss_to_returns function for accurate simulation
        sl_results = apply_stop_loss_to_returns(
            returns,
            stop_loss_pct=stop_pct,
            mae_values=mae_values,
        )

        suggestions[level_name] = {
            'stop_loss_pct': stop_pct * 100,
            'stop_loss_decimal': stop_pct,
            'trades_stopped': sl_results['pct_stopped'],
            'description': f'{100 - (5 if level_name == "conservative" else 2.5 if level_name == "moderate" else 1):.1f}% of trades unaffected',
            'original_total_return': sl_results['original_total_return'],
            'new_total_return': sl_results['adjusted_total_return'],
            'improvement': sl_results['improvement'],
            'trades_remaining': len(returns) - sl_results['n_stopped'],
            'n_stopped': sl_results['n_stopped'],
        }

    # Also add a "no stop" baseline for comparison
    suggestions['none'] = {
        'stop_loss_pct': None,
        'description': 'No stop-loss (baseline)',
        'original_total_return': np.sum(returns) * 100,
        'new_total_return': np.sum(returns) * 100,
        'improvement': 0,
        'trades_remaining': len(returns),
        'n_stopped': 0,
    }

    return suggestions


def suggest_stop_loss_old(trades_df: pd.DataFrame, atr_column: str = None) -> Dict:
    """
    [DEPRECATED] Old stop-loss suggestion function.

    Kept for backwards compatibility. Use suggest_stop_loss instead.
    """
    if len(trades_df) == 0:
        return {'error': 'No trades to analyze'}

    returns = trades_df['trade_return']

    # Calculate percentile-based stops
    suggestions = {
        'conservative': {
            'stop_loss_pct': np.percentile(returns, 5) * 100,  # 95% of trades better than this
            'trades_stopped': 5,
            'description': '5th percentile - very tight, 5% of trades stopped'
        },
        'moderate': {
            'stop_loss_pct': np.percentile(returns, 2.5) * 100,  # 97.5% better
            'trades_stopped': 2.5,
            'description': '2.5th percentile - moderate, 2.5% stopped'
        },
        'aggressive': {
            'stop_loss_pct': np.percentile(returns, 1) * 100,  # 99% better
            'trades_stopped': 1,
            'description': '1st percentile - loose, 1% stopped'
        }
    }

    # Calculate impact of each stop
    for level_name, level in suggestions.items():
        stop = level['stop_loss_pct'] / 100
        stopped_trades = trades_df[returns < stop]
        surviving_trades = trades_df[returns >= stop]

        if len(surviving_trades) > 0:
            level['new_avg_return'] = surviving_trades['trade_return'].mean() * 100
            level['new_total_return'] = surviving_trades['trade_return'].sum() * 100
            level['original_total_return'] = returns.sum() * 100
            level['improvement'] = level['new_total_return'] - level['original_total_return']
            level['trades_remaining'] = len(surviving_trades)

    return suggestions


# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_performance_review(results_df: pd.DataFrame, trading_days: int = 343) -> str:
    """
    Generate comprehensive performance review report.
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE PERFORMANCE REVIEW")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Filter to trades only
    trades = results_df[results_df['should_trade']].copy() if 'should_trade' in results_df.columns else results_df.copy()

    if len(trades) == 0:
        lines.append("ERROR: No trades found in results")
        return "\n".join(lines)

    # --- Section 1: Sharpe Ratio Verification ---
    lines.append("=" * 80)
    lines.append("1. SHARPE RATIO VERIFICATION")
    lines.append("=" * 80)
    lines.append("")

    sharpe_results = verify_sharpe_calculation(trades, trading_days)

    lines.append("ISSUE: Original Sharpe calculation may be overstated")
    lines.append("")
    lines.append("The original formula uses sqrt(35000) assuming trading every 15-min candle.")
    lines.append("With sparse trading, we should use actual trade frequency for annualization.")
    lines.append("")
    lines.append(f"Trade Statistics:")
    lines.append(f"  - Total Trades:        {sharpe_results['n_trades']}")
    lines.append(f"  - Trading Days:        {sharpe_results['trading_days']}")
    lines.append(f"  - Trades per Year:     {sharpe_results.get('trades_per_year', 'N/A'):.1f}")
    lines.append(f"  - Mean Return:         {sharpe_results['mean_return']*100:.4f}%")
    lines.append(f"  - Std Return:          {sharpe_results['std_return']*100:.4f}%")
    lines.append(f"  - Min Return:          {sharpe_results['min_return']*100:.4f}%")
    lines.append(f"  - Max Return:          {sharpe_results['max_return']*100:.4f}%")
    lines.append("")
    lines.append("Sharpe Ratio Comparison:")
    lines.append(f"  - Original (15-min):   {sharpe_results.get('sharpe_original_15min', 'N/A'):.2f}")
    lines.append(f"  - Trade-based:         {sharpe_results.get('sharpe_trade_based', 'N/A'):.2f}")
    lines.append(f"  - 95% CI:              [{sharpe_results.get('sharpe_95ci_low', 'N/A'):.2f}, {sharpe_results.get('sharpe_95ci_high', 'N/A'):.2f}]")
    lines.append("")
    lines.append("RECOMMENDATION: Use trade-based Sharpe for realistic assessment")
    lines.append("")

    # --- Section 2: Confidence Bucket Analysis ---
    lines.append("=" * 80)
    lines.append("2. CONFIDENCE BUCKET ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Does higher confidence correlate with better returns?")
    lines.append("")

    conf_buckets = analyze_confidence_buckets(trades)

    if len(conf_buckets) > 0:
        lines.append(f"{'Bucket':<20} {'N':>6} {'%Total':>7} {'Win%':>7} {'AvgRet':>8} {'TotRet':>9}")
        lines.append("-" * 70)
        for _, row in conf_buckets.iterrows():
            lines.append(
                f"{row['bucket']:<20} {row['n_trades']:>6} "
                f"{row['pct_of_total']:>7.1f} {row['win_rate']:>7.1f} "
                f"{row['avg_return']:>8.4f} {row['total_return']:>9.2f}"
            )
        lines.append("")

        # Analysis
        if len(conf_buckets) > 1:
            best_bucket = conf_buckets.loc[conf_buckets['avg_return'].idxmax()]
            worst_bucket = conf_buckets.loc[conf_buckets['avg_return'].idxmin()]
            lines.append(f"Best Performing:  {best_bucket['bucket']} ({best_bucket['avg_return']:.4f}% avg)")
            lines.append(f"Worst Performing: {worst_bucket['bucket']} ({worst_bucket['avg_return']:.4f}% avg)")

            # Correlation check
            if conf_buckets['n_trades'].sum() > 50:
                # Simple check: is higher confidence better?
                buckets_sorted = conf_buckets.sort_values('range')
                correlation = buckets_sorted['avg_return'].corr(pd.Series(range(len(buckets_sorted))))
                lines.append(f"Confidence-Return Correlation: {correlation:.3f}")
                if correlation > 0.3:
                    lines.append("FINDING: Higher confidence DOES correlate with better returns")
                elif correlation < -0.3:
                    lines.append("WARNING: Higher confidence correlates with WORSE returns!")
                else:
                    lines.append("FINDING: No strong correlation between confidence and returns")
    else:
        lines.append("Insufficient data for bucket analysis")
    lines.append("")

    # --- Section 3: Tail Risk Analysis ---
    lines.append("=" * 80)
    lines.append("3. TAIL RISK ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    tail_risk = analyze_tail_risk(trades)

    lines.append("Distribution Analysis:")
    lines.append(f"  - Mean Return:     {tail_risk['mean_return']:.4f}%")
    lines.append(f"  - Std Return:      {tail_risk['std_return']:.4f}%")
    lines.append(f"  - Skewness:        {tail_risk['skewness']:.3f}")
    lines.append(f"  - Excess Kurtosis: {tail_risk['kurtosis']:.3f}")
    lines.append("")

    if tail_risk['skewness'] < -0.5:
        lines.append("WARNING: Negative skew indicates fat left tail (large losses)")
    if tail_risk['kurtosis'] > 3:
        lines.append("WARNING: High kurtosis indicates fat tails (extreme events)")
    lines.append("")

    lines.append("Percentile Analysis:")
    lines.append(f"  - 1st percentile:  {tail_risk['pct_1']:.4f}%")
    lines.append(f"  - 5th percentile:  {tail_risk['pct_5']:.4f}%")
    lines.append(f"  - 25th percentile: {tail_risk['pct_25']:.4f}%")
    lines.append(f"  - 75th percentile: {tail_risk['pct_75']:.4f}%")
    lines.append(f"  - 95th percentile: {tail_risk['pct_95']:.4f}%")
    lines.append(f"  - 99th percentile: {tail_risk['pct_99']:.4f}%")
    lines.append("")

    lines.append("Risk Metrics:")
    lines.append(f"  - VaR (95%):       {tail_risk['var_95']:.4f}%")
    lines.append(f"  - VaR (99%):       {tail_risk['var_99']:.4f}%")
    lines.append(f"  - CVaR (95%):      {tail_risk.get('cvar_95', 'N/A')}%")
    lines.append(f"  - Max Drawdown:    {tail_risk['max_drawdown']:.4f}%")
    lines.append("")

    lines.append("Worst 5 Trades:")
    for i, loss in enumerate(tail_risk['worst_5_trades']):
        lines.append(f"  {i+1}. {loss:.4f}%")
    lines.append("")

    lines.append(f"Trades to recover from worst loss: {tail_risk['trades_to_recover_worst']:.1f}")
    lines.append(f"Profit Factor: {tail_risk.get('profit_factor', 'N/A'):.2f}")
    lines.append("")

    # Large loss patterns
    large_losses = identify_loss_patterns(trades, threshold=-0.01)
    if len(large_losses) > 0:
        lines.append(f"Large Losses (> -1%): {len(large_losses)} trades")
        lines.append("Analysis of large losses:")
        if 'direction' in large_losses.columns:
            direction_counts = large_losses['direction'].value_counts()
            for direction, count in direction_counts.items():
                lines.append(f"  - {direction}: {count} trades ({count/len(large_losses)*100:.1f}%)")
    lines.append("")

    # --- Section 4: Stop-Loss Suggestions ---
    lines.append("=" * 80)
    lines.append("4. STOP-LOSS RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    stop_suggestions = suggest_stop_loss(trades)

    for level_name, level in stop_suggestions.items():
        lines.append(f"{level_name.upper()}:")
        lines.append(f"  Stop Level: {level['stop_loss_pct']:.4f}%")
        lines.append(f"  Description: {level['description']}")
        if 'new_total_return' in level:
            lines.append(f"  Original Total Return: {level['original_total_return']:.2f}%")
            lines.append(f"  New Total Return:      {level['new_total_return']:.2f}%")
            lines.append(f"  Improvement:           {level['improvement']:+.2f}%")
            lines.append(f"  Trades Remaining:      {level['trades_remaining']}")
        lines.append("")

    # --- Section 5: Walk-Forward Readiness ---
    lines.append("=" * 80)
    lines.append("5. WALK-FORWARD VALIDATION READINESS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Recommended test periods for robust validation:")
    lines.append("")
    for period in MARKET_PERIODS:
        lines.append(f"  {period.name}:")
        lines.append(f"    - {period.description}")
        lines.append(f"    - Train until: {period.train_end}")
        lines.append(f"    - Test: {period.test_start} to {period.test_end}")
        lines.append("")

    lines.append("To run walk-forward validation, ensure you have data covering these periods.")
    lines.append("")

    # --- Summary ---
    lines.append("=" * 80)
    lines.append("SUMMARY & ACTION ITEMS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. SHARPE RATIO:")
    lines.append(f"   - Corrected Sharpe: {sharpe_results.get('sharpe_trade_based', 'N/A'):.2f}")
    lines.append(f"   - Still positive: {'YES' if sharpe_results.get('sharpe_trade_based', 0) > 0 else 'NO'}")
    lines.append("")
    lines.append("2. CONFIDENCE ANALYSIS:")
    if len(conf_buckets) > 0:
        high_conf_trades = conf_buckets[conf_buckets['range'].str.contains('0.6|0.7|0.8')]['n_trades'].sum()
        lines.append(f"   - High confidence (>0.60) trades: {high_conf_trades}")
    lines.append("")
    lines.append("3. TAIL RISK:")
    lines.append(f"   - Maximum single-trade loss: {tail_risk['pct_1']:.4f}%")
    lines.append(f"   - Consider stop-loss at: {stop_suggestions['moderate']['stop_loss_pct']:.4f}%")
    lines.append("")
    lines.append("4. NEXT STEPS:")
    lines.append("   - [ ] Implement stop-loss based on recommendations")
    lines.append("   - [ ] Consider filtering to higher confidence trades only")
    lines.append("   - [ ] Run walk-forward validation on historical periods")
    lines.append("   - [ ] Paper trade for at least 30 days before live trading")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Run performance review on existing results."""

    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    MODEL_DIR = Path("experiments/run_001")
    OUTPUT_DIR = MODEL_DIR / "review"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing calibrated results
    calibrated_results_path = MODEL_DIR / "calibrated_results" / "calibrated_predictions.csv"

    if calibrated_results_path.exists():
        logger.info(f"Loading existing results from {calibrated_results_path}")
        results_df = pd.read_csv(calibrated_results_path)

        # Generate review report
        report = generate_performance_review(results_df)
        print(report)

        # Save report
        with open(OUTPUT_DIR / "performance_review.txt", 'w') as f:
            f.write(report)

        logger.info(f"\nReview saved to: {OUTPUT_DIR / 'performance_review.txt'}")

    else:
        logger.warning(f"No calibrated results found at {calibrated_results_path}")
        logger.info("Run 'python scripts/run_calibrated.py' first to generate results")

        # If we have a model and test data, run inference
        if (MODEL_DIR / "best_model.pt").exists() and (DATA_DIR / "test.parquet").exists():
            logger.info("\nModel and test data found. Running inference...")

            # Import and run calibrated inference
            from run_calibrated import load_calibrated_model, run_inference, compute_performance

            # Load model
            calibrated_model, config = load_calibrated_model(MODEL_DIR / "best_model.pt")

            # Load test data
            test_df = pd.read_parquet(DATA_DIR / "test.parquet")
            with open(DATA_DIR / "feature_info.json") as f:
                feature_info = json.load(f)

            price_cols = feature_info['price_columns']
            eng_cols = [c for c in feature_info['engineered_columns'] if c in test_df.columns]

            test_dataset = TradingDataset(
                test_df, price_cols, eng_cols,
                window_size=config.window_size
            )
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

            # Run inference
            results_df = run_inference(calibrated_model, test_loader)

            # Generate review
            report = generate_performance_review(results_df)
            print(report)

            # Save
            with open(OUTPUT_DIR / "performance_review.txt", 'w') as f:
                f.write(report)

        else:
            logger.error("Model or test data not found. Cannot generate review.")
            logger.info("Required files:")
            logger.info(f"  - {MODEL_DIR / 'best_model.pt'}")
            logger.info(f"  - {DATA_DIR / 'test.parquet'}")


if __name__ == "__main__":
    main()
