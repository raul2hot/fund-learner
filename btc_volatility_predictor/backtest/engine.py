"""
Backtesting engine that:
1. Loads test predictions
2. Runs each strategy
3. Tracks positions, P&L, drawdown
4. Generates performance metrics
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .strategies.base import BaseStrategy, Position


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    regime_at_entry: str
    holding_period: int = 0


@dataclass
class BacktestResult:
    """Results from running a backtest."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float
    profit_factor: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    regime_returns: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'avg_trade_pnl': self.avg_trade_pnl,
            'profit_factor': self.profit_factor,
            'regime_returns': self.regime_returns
        }


class BacktestEngine:
    """Engine for running strategy backtests."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade (Binance taker fee)
        slippage: float = 0.0005,  # 0.05%
        position_size: float = 1.0  # 100% of capital per trade
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        regime_column: str = 'predicted_regime'
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy: Strategy instance to test
            data: DataFrame with OHLCV + predictions
            regime_column: Column name for regime predictions

        Returns:
            BacktestResult with all metrics
        """
        strategy.reset()

        # Initialize tracking
        capital = self.initial_capital
        equity_curve = [capital]
        trades: list[Trade] = []
        history: list[dict] = []

        position: Optional[Position] = None
        peak_equity = capital

        # Track regime-specific returns
        regime_pnl = {'HIGH': 0.0, 'LOW': 0.0}

        # Convert DataFrame to list of dicts for efficient access
        data_records = data.to_dict('records')

        for i, row in enumerate(data_records):
            close = row['close']
            predicted_regime = 'HIGH' if row.get(regime_column, 0) == 1 else 'LOW'

            # Update strategy position reference
            strategy.set_position(position)

            # Generate signal
            signal = strategy.generate_signal(row, predicted_regime, history)

            # Process signal
            if signal == 'CLOSE' and position is not None:
                # Close existing position
                exit_price = self._apply_slippage(close, position.direction, closing=True)
                trade = self._close_position(position, exit_price, i, capital)
                trades.append(trade)

                # Update regime P&L
                regime_pnl[position.entry_regime] += trade.pnl

                capital += trade.pnl
                position = None
                strategy.set_position(None)

            elif signal == 'BUY' and position is None:
                # Open long position
                entry_price = self._apply_slippage(close, 'LONG', closing=False)
                entry_price = self._apply_transaction_cost(entry_price, 'LONG', opening=True)
                position = Position(
                    entry_price=entry_price,
                    entry_time=i,
                    direction='LONG',
                    size=self.position_size,
                    entry_regime=predicted_regime
                )

                # Calculate initial stop loss if strategy supports it
                atr = row.get('atr_14', 0) * close
                if hasattr(strategy, 'calculate_stop_loss'):
                    position.stop_loss = strategy.calculate_stop_loss(entry_price, atr, 'LONG')

                strategy.set_position(position)

            elif signal == 'SELL' and position is None:
                # Open short position
                entry_price = self._apply_slippage(close, 'SHORT', closing=False)
                entry_price = self._apply_transaction_cost(entry_price, 'SHORT', opening=True)
                position = Position(
                    entry_price=entry_price,
                    entry_time=i,
                    direction='SHORT',
                    size=self.position_size,
                    entry_regime=predicted_regime
                )

                # Calculate initial stop loss if strategy supports it
                atr = row.get('atr_14', 0) * close
                if hasattr(strategy, 'calculate_stop_loss'):
                    position.stop_loss = strategy.calculate_stop_loss(entry_price, atr, 'SHORT')

                strategy.set_position(position)

            # Update equity (mark-to-market)
            if position is not None:
                unrealized_pnl = self._calculate_pnl(position, close, capital)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # Track peak for drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity

            # Add to history
            history.append(row)

        # Close any remaining position at end
        if position is not None:
            final_close = data_records[-1]['close']
            exit_price = self._apply_slippage(final_close, position.direction, closing=True)
            trade = self._close_position(position, exit_price, len(data_records) - 1, capital)
            trades.append(trade)
            regime_pnl[position.entry_regime] += trade.pnl
            capital += trade.pnl

        # Calculate metrics
        result = self._calculate_metrics(
            strategy.name,
            trades,
            equity_curve,
            regime_pnl
        )

        return result

    def _apply_slippage(self, price: float, direction: str, closing: bool) -> float:
        """Apply slippage to price."""
        if direction == 'LONG':
            if closing:
                return price * (1 - self.slippage)  # Worse price when selling
            else:
                return price * (1 + self.slippage)  # Worse price when buying
        else:  # SHORT
            if closing:
                return price * (1 + self.slippage)  # Worse price when buying back
            else:
                return price * (1 - self.slippage)  # Better price when selling

    def _apply_transaction_cost(self, price: float, direction: str, opening: bool) -> float:
        """Apply transaction cost to effective entry/exit price."""
        if direction == 'LONG':
            if opening:
                return price * (1 + self.transaction_cost)
            else:
                return price * (1 - self.transaction_cost)
        else:  # SHORT
            if opening:
                return price * (1 - self.transaction_cost)
            else:
                return price * (1 + self.transaction_cost)

    def _calculate_pnl(self, position: Position, current_price: float, capital: float) -> float:
        """Calculate unrealized P&L for a position."""
        position_value = capital * position.size

        if position.direction == 'LONG':
            return position_value * (current_price / position.entry_price - 1)
        else:  # SHORT
            return position_value * (1 - current_price / position.entry_price)

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: int,
        capital: float
    ) -> Trade:
        """Close a position and return the Trade."""
        # Apply transaction cost to exit
        if position.direction == 'LONG':
            effective_exit = exit_price * (1 - self.transaction_cost)
            pnl_pct = (effective_exit / position.entry_price) - 1
        else:  # SHORT
            effective_exit = exit_price * (1 + self.transaction_cost)
            pnl_pct = 1 - (effective_exit / position.entry_price)

        position_value = capital * position.size
        pnl = position_value * pnl_pct

        return Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            regime_at_entry=position.entry_regime,
            holding_period=exit_time - position.entry_time
        )

    def _calculate_metrics(
        self,
        strategy_name: str,
        trades: list[Trade],
        equity_curve: list[float],
        regime_pnl: dict
    ) -> BacktestResult:
        """Calculate performance metrics from trades and equity curve."""
        equity = np.array(equity_curve)

        # Total return
        total_return = (equity[-1] / equity[0]) - 1

        # Sharpe ratio (annualized, assuming hourly data)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]

            win_rate = len(wins) / num_trades
            avg_trade_pnl = sum(t.pnl for t in trades) / num_trades

            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            profit_factor = 0.0

        # Regime-specific returns
        regime_returns = {
            'HIGH': regime_pnl['HIGH'] / self.initial_capital,
            'LOW': regime_pnl['LOW'] / self.initial_capital
        }

        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=num_trades,
            avg_trade_pnl=avg_trade_pnl,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
            regime_returns=regime_returns
        )

    def save_trades(self, result: BacktestResult, output_dir: str = "backtest/results/trades"):
        """Save individual trade log to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        trades_data = []
        for t in result.trades:
            trades_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'direction': t.direction,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'regime_at_entry': t.regime_at_entry,
                'holding_period': t.holding_period
            })

        df = pd.DataFrame(trades_data)
        filename = f"{output_dir}/{result.strategy_name.lower().replace(' ', '_')}_trades.csv"
        df.to_csv(filename, index=False)
        return filename
