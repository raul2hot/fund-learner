"""
Analyze and compare strategy performance:
1. Performance metrics table
2. Equity curves plot
3. Drawdown analysis
4. Trade distribution by regime
5. Statistical significance tests
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from .engine import BacktestResult


def create_summary_table(
    results: list[BacktestResult],
    output_path: str = "backtest/results/summary.csv"
) -> pd.DataFrame:
    """
    Create performance summary table.

    Args:
        results: List of BacktestResult objects
        output_path: Path to save CSV

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    for r in results:
        summary_data.append({
            'Strategy': r.strategy_name,
            'Return (%)': f"{r.total_return * 100:.2f}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'MaxDD (%)': f"{r.max_drawdown * 100:.2f}",
            'Win Rate (%)': f"{r.win_rate * 100:.1f}" if r.num_trades > 0 else "N/A",
            'Trades': r.num_trades,
            'Profit Factor': f"{r.profit_factor:.2f}" if r.profit_factor < float('inf') else "N/A",
            'HIGH Return (%)': f"{r.regime_returns.get('HIGH', 0) * 100:.2f}",
            'LOW Return (%)': f"{r.regime_returns.get('LOW', 0) * 100:.2f}"
        })

    df = pd.DataFrame(summary_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def plot_equity_curves(
    results: list[BacktestResult],
    predictions_df: Optional[pd.DataFrame] = None,
    output_path: str = "backtest/results/equity_curves.png"
):
    """
    Plot equity curves for all strategies.

    Args:
        results: List of BacktestResult objects
        predictions_df: DataFrame with regime predictions (for shading)
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette for strategies
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, result in enumerate(results):
        equity = np.array(result.equity_curve)
        normalized_equity = equity / equity[0] * 100  # Start at 100

        ax.plot(normalized_equity, label=result.strategy_name,
                color=colors[i], linewidth=2)

    # Add regime shading if predictions available
    if predictions_df is not None and 'predicted_regime' in predictions_df.columns:
        regimes = predictions_df['predicted_regime'].values
        for i in range(len(regimes)):
            if regimes[i] == 1:  # HIGH volatility
                ax.axvspan(i, i+1, alpha=0.1, color='red')

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Portfolio Value (indexed to 100)', fontsize=12)
    ax.set_title('Strategy Equity Curves', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at starting value
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regime_analysis(
    results: list[BacktestResult],
    output_path: str = "backtest/results/regime_analysis.png"
):
    """
    Plot performance breakdown by predicted regime.

    Args:
        results: List of BacktestResult objects
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = [r.strategy_name for r in results]
    high_returns = [r.regime_returns.get('HIGH', 0) * 100 for r in results]
    low_returns = [r.regime_returns.get('LOW', 0) * 100 for r in results]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, high_returns, width, label='HIGH Volatility',
                   color='coral', edgecolor='darkred')
    bars2 = ax.bar(x + width/2, low_returns, width, label='LOW Volatility',
                   color='steelblue', edgecolor='darkblue')

    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title('Returns by Volatility Regime', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trades_analysis(
    results: list[BacktestResult],
    output_path: str = "backtest/results/trades_analysis.png"
):
    """
    Plot trade analysis: P&L distribution, win/loss by regime, holding periods.

    Args:
        results: List of BacktestResult objects
        output_path: Path to save plot
    """
    # Filter results with trades
    results_with_trades = [r for r in results if r.num_trades > 0]

    if not results_with_trades:
        print("No trades to analyze")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trade P&L distribution (top left)
    ax1 = axes[0, 0]
    all_pnls = []
    labels = []
    for r in results_with_trades:
        pnls = [t.pnl_pct * 100 for t in r.trades]
        if pnls:
            all_pnls.append(pnls)
            labels.append(r.strategy_name)

    if all_pnls:
        ax1.boxplot(all_pnls, labels=labels)
        ax1.set_ylabel('Trade P&L (%)')
        ax1.set_title('Trade P&L Distribution')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Win rate comparison (top right)
    ax2 = axes[0, 1]
    strategies = [r.strategy_name for r in results_with_trades]
    win_rates = [r.win_rate * 100 for r in results_with_trades]

    colors = ['green' if wr >= 50 else 'red' for wr in win_rates]
    bars = ax2.bar(strategies, win_rates, color=colors, edgecolor='black')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Strategy Win Rates')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, wr in zip(bars, win_rates):
        ax2.annotate(f'{wr:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    # 3. Win/Loss by regime (bottom left)
    ax3 = axes[1, 0]

    regime_stats = {'HIGH_wins': [], 'HIGH_losses': [], 'LOW_wins': [], 'LOW_losses': []}

    for r in results_with_trades:
        high_trades = [t for t in r.trades if t.regime_at_entry == 'HIGH']
        low_trades = [t for t in r.trades if t.regime_at_entry == 'LOW']

        high_wins = len([t for t in high_trades if t.pnl > 0])
        high_losses = len([t for t in high_trades if t.pnl <= 0])
        low_wins = len([t for t in low_trades if t.pnl > 0])
        low_losses = len([t for t in low_trades if t.pnl <= 0])

        regime_stats['HIGH_wins'].append(high_wins)
        regime_stats['HIGH_losses'].append(high_losses)
        regime_stats['LOW_wins'].append(low_wins)
        regime_stats['LOW_losses'].append(low_losses)

    x = np.arange(len(results_with_trades))
    width = 0.2

    ax3.bar(x - 1.5*width, regime_stats['HIGH_wins'], width, label='HIGH Wins', color='lightcoral')
    ax3.bar(x - 0.5*width, regime_stats['HIGH_losses'], width, label='HIGH Losses', color='darkred')
    ax3.bar(x + 0.5*width, regime_stats['LOW_wins'], width, label='LOW Wins', color='lightblue')
    ax3.bar(x + 1.5*width, regime_stats['LOW_losses'], width, label='LOW Losses', color='darkblue')

    ax3.set_ylabel('Number of Trades')
    ax3.set_title('Win/Loss by Regime')
    ax3.set_xticks(x)
    ax3.set_xticklabels([r.strategy_name for r in results_with_trades], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Holding period distribution (bottom right)
    ax4 = axes[1, 1]
    holding_periods = []
    hp_labels = []

    for r in results_with_trades:
        periods = [t.holding_period for t in r.trades]
        if periods:
            holding_periods.append(periods)
            hp_labels.append(r.strategy_name)

    if holding_periods:
        ax4.boxplot(holding_periods, labels=hp_labels)
        ax4.set_ylabel('Holding Period (hours)')
        ax4.set_title('Trade Holding Periods')
        ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_drawdown(
    results: list[BacktestResult],
    output_path: str = "backtest/results/drawdown.png"
):
    """
    Plot drawdown curves for all strategies.

    Args:
        results: List of BacktestResult objects
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, result in enumerate(results):
        equity = np.array(result.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        ax.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color=colors[i])
        ax.plot(drawdown, label=f"{result.strategy_name} (Max: {result.max_drawdown*100:.1f}%)",
                color=colors[i], linewidth=1.5)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Strategy Drawdowns', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Drawdowns are negative

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(
    results: list[BacktestResult],
    predictions_df: Optional[pd.DataFrame] = None,
    output_dir: str = "backtest/results"
):
    """
    Generate complete analysis report.

    Args:
        results: List of BacktestResult objects
        predictions_df: DataFrame with predictions (for context)
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING ANALYSIS REPORT")
    print("="*60)

    # 1. Summary table
    print("\n1. Creating summary table...")
    summary = create_summary_table(results, f"{output_dir}/summary.csv")
    print(summary.to_string())

    # 2. Equity curves
    print("\n2. Plotting equity curves...")
    plot_equity_curves(results, predictions_df, f"{output_dir}/equity_curves.png")
    print(f"   Saved to {output_dir}/equity_curves.png")

    # 3. Regime analysis
    print("\n3. Plotting regime analysis...")
    plot_regime_analysis(results, f"{output_dir}/regime_analysis.png")
    print(f"   Saved to {output_dir}/regime_analysis.png")

    # 4. Trade analysis
    print("\n4. Plotting trade analysis...")
    plot_trades_analysis(results, f"{output_dir}/trades_analysis.png")
    print(f"   Saved to {output_dir}/trades_analysis.png")

    # 5. Drawdown
    print("\n5. Plotting drawdown curves...")
    plot_drawdown(results, f"{output_dir}/drawdown.png")
    print(f"   Saved to {output_dir}/drawdown.png")

    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    # Best strategy by Sharpe
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    print(f"\nBest Sharpe Ratio: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")

    # Best strategy by return
    best_return = max(results, key=lambda r: r.total_return)
    print(f"Best Total Return: {best_return.strategy_name} ({best_return.total_return*100:.2f}%)")

    # Lowest drawdown
    lowest_dd = min(results, key=lambda r: r.max_drawdown)
    print(f"Lowest Max Drawdown: {lowest_dd.strategy_name} ({lowest_dd.max_drawdown*100:.2f}%)")

    # Trading activity
    total_trades = sum(r.num_trades for r in results)
    print(f"\nTotal Trades Across All Strategies: {total_trades}")

    # Regime effectiveness
    print("\nRegime Performance Summary:")
    for r in results:
        if r.num_trades > 0:
            high_ret = r.regime_returns.get('HIGH', 0) * 100
            low_ret = r.regime_returns.get('LOW', 0) * 100
            print(f"  {r.strategy_name}: HIGH={high_ret:+.2f}%, LOW={low_ret:+.2f}%")

    print("\n" + "="*60)
    print(f"Report saved to {output_dir}/")
    print("="*60)
