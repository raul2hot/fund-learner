"""
Base class for all trading strategies.

Each strategy should implement:
- generate_signal(row, prediction, history) -> 'BUY', 'SELL', 'HOLD', or 'CLOSE'
- get_params() -> dict of strategy parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


Signal = Literal['BUY', 'SELL', 'HOLD', 'CLOSE']


@dataclass
class Position:
    """Represents an open trading position."""
    entry_price: float
    entry_time: int  # Index in the data
    direction: Literal['LONG', 'SHORT']
    size: float = 1.0
    entry_regime: str = 'UNKNOWN'
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.position: Optional[Position] = None

    @abstractmethod
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate trading signal based on current data and prediction.

        Args:
            row: Current bar data (OHLCV + indicators)
            predicted_regime: 'HIGH' or 'LOW' volatility prediction
            history: List of previous bars (most recent last)

        Returns:
            Signal: 'BUY', 'SELL', 'HOLD', or 'CLOSE'
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return strategy parameters for logging."""
        pass

    def set_position(self, position: Optional[Position]):
        """Set current position."""
        self.position = position

    def has_position(self) -> bool:
        """Check if strategy has an open position."""
        return self.position is not None

    def get_position_direction(self) -> Optional[str]:
        """Get direction of current position."""
        if self.position:
            return self.position.direction
        return None

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.position and self.position.stop_loss:
            if self.position.direction == 'LONG':
                return current_price <= self.position.stop_loss
            else:  # SHORT
                return current_price >= self.position.stop_loss
        return False

    def update_trailing_stop(self, current_price: float, atr: float, multiplier: float = 2.0):
        """Update trailing stop based on current price."""
        if self.position and self.position.direction == 'LONG':
            new_stop = current_price - (atr * multiplier)
            if self.position.trailing_stop is None or new_stop > self.position.trailing_stop:
                self.position.trailing_stop = new_stop
                self.position.stop_loss = new_stop
        elif self.position and self.position.direction == 'SHORT':
            new_stop = current_price + (atr * multiplier)
            if self.position.trailing_stop is None or new_stop < self.position.trailing_stop:
                self.position.trailing_stop = new_stop
                self.position.stop_loss = new_stop

    def reset(self):
        """Reset strategy state."""
        self.position = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
