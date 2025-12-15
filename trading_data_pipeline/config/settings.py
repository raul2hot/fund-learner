"""
NON-FRAGILE TIME CONFIGURATION

Design Principles:
1. Use relative dates where possible (e.g., "last 5 years")
2. Hard floor based on data availability (Binance Futures launch)
3. Dynamic end date (always "now" unless specified)
4. Graceful handling of future dates
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class TimeConfig:
    """
    NON-FRAGILE TIME CONFIGURATION

    Design Principles:
    1. Use relative dates where possible (e.g., "last 5 years")
    2. Hard floor based on data availability (Binance Futures launch)
    3. Dynamic end date (always "now" unless specified)
    4. Graceful handling of future dates
    """

    # === HARD CONSTRAINTS (Data Availability) ===
    # Binance Futures launched Sept 13, 2019
    # We use Sept 15 to ensure stable data
    BINANCE_FUTURES_LAUNCH = "2019-09-15"

    # Minimum required for full feature set
    DATA_FLOOR = BINANCE_FUTURES_LAUNCH

    @classmethod
    def get_default_start(cls) -> str:
        """
        Default: Maximum available history with full features.
        This automatically grows as time passes.
        """
        return cls.DATA_FLOOR

    @classmethod
    def get_default_end(cls) -> str:
        """
        Default: Today's date.
        Always fresh, never stale.
        """
        return datetime.utcnow().strftime("%Y-%m-%d")

    @classmethod
    def get_relative_start(cls, years_back: int = 5) -> str:
        """
        Alternative: Rolling window (e.g., last 5 years).
        Useful for strategies that don't need ancient data.
        """
        start = datetime.utcnow() - timedelta(days=365 * years_back)

        # Never go before data floor
        floor = datetime.strptime(cls.DATA_FLOOR, "%Y-%m-%d")
        if start < floor:
            start = floor

        return start.strftime("%Y-%m-%d")

    @classmethod
    def validate_dates(cls, start: str, end: str) -> Tuple[str, str]:
        """
        Validate and adjust dates to ensure data availability.
        Returns adjusted (start, end) with warnings if modified.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        floor_dt = datetime.strptime(cls.DATA_FLOOR, "%Y-%m-%d")
        now = datetime.utcnow()

        warnings: List[str] = []

        # Adjust start if before data floor
        if start_dt < floor_dt:
            warnings.append(
                f"Start date {start} is before Binance Futures launch. "
                f"Adjusted to {cls.DATA_FLOOR}"
            )
            start_dt = floor_dt

        # Adjust end if in future
        if end_dt > now:
            new_end = now.strftime("%Y-%m-%d")
            warnings.append(
                f"End date {end} is in the future. "
                f"Adjusted to {new_end}"
            )
            end_dt = now

        # Ensure start < end
        if start_dt >= end_dt:
            raise ValueError(
                f"Start date ({start_dt.date()}) must be before "
                f"end date ({end_dt.date()})"
            )

        for w in warnings:
            print(f"WARNING: {w}")

        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# === EXPORTED DEFAULTS ===
# These are used throughout the codebase

DEFAULT_START_DATE = TimeConfig.get_default_start()  # "2019-09-15"
DEFAULT_END_DATE = TimeConfig.get_default_end()      # Dynamic: today
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"

# For strategies that want rolling window instead of full history
ROLLING_WINDOW_YEARS = 3  # Configurable

# IMPORTANT: Walk-forward split (chronological, NO shuffling)
TRAIN_RATIO = 0.70   # 70% for training (earliest data)
VAL_RATIO = 0.15     # 15% for validation (middle)
TEST_RATIO = 0.15    # 15% for testing (most recent)
