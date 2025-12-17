"""
Regime Filter Module

Detects market stress regimes and gates trading decisions.

The regime filter addresses the critical issue identified in walk-forward validation:
- May 2021 had 1,285 trades vs FTX crash's 109 trades (12x over-trading)
- May 2021 returned -36.79% while FTX returned +0.72%
- Seed 1337 was profitable (+15.91%) in May 2021, proving the architecture CAN work

This module provides:
1. MarketRegime detection (NORMAL, ELEVATED, EXTREME)
2. Trade gating based on regime
3. Position scaling based on regime
4. Integration with existing backtesting pipeline

Usage:
    from regime_filter import RegimeFilter, RegimeConfig, apply_regime_filter_to_predictions

    # Basic usage
    filter = RegimeFilter(config)
    regime, metrics = filter.detect_regime(prices, funding_rates)
    should_trade, reason = filter.should_allow_trade(regime, timestamp)

    # Integration with backtest predictions
    filtered_predictions = apply_regime_filter_to_predictions(
        predictions, prices, funding_rates, config
    )
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """
    Market regime classification.

    NORMAL: Standard market conditions - trade normally
    ELEVATED: Increased stress - reduce position sizes, limit trades
    EXTREME: Crisis conditions - no new trades, protect capital
    """
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


@dataclass
class RegimeConfig:
    """
    Configuration for regime detection thresholds.

    Thresholds are calibrated based on historical crypto market behavior:
    - Volatility thresholds based on rolling std of log returns
    - Price drop thresholds based on 24h percentage change
    - Trade limits to prevent over-trading during stress

    Default values are set to be conservative - better to miss trades
    than to over-trade during crises.
    """
    # Volatility thresholds (rolling std of returns relative to baseline)
    vol_lookback: int = 24  # hours for current volatility calculation
    vol_baseline_lookback: int = 168  # 7 days for baseline volatility
    vol_elevated_threshold: float = 1.5  # 1.5x normal vol triggers ELEVATED
    vol_extreme_threshold: float = 2.5   # 2.5x normal vol triggers EXTREME

    # NEW: Absolute volatility thresholds (annualized %)
    # These trigger regardless of relative comparison
    # BTC historical: normal ~40-60%, elevated ~80-100%, extreme >120%
    vol_absolute_elevated: float = 0.04  # 4% daily std (~75% annualized)
    vol_absolute_extreme: float = 0.06   # 6% daily std (~115% annualized)
    use_absolute_vol: bool = True  # Enable absolute vol detection

    # NEW: Historical percentile thresholds (0-1)
    # Compare current vol to ALL available history, not just recent window
    vol_percentile_elevated: float = 0.80  # Top 20% of historical vol
    vol_percentile_extreme: float = 0.95   # Top 5% of historical vol
    use_percentile_vol: bool = True  # Enable percentile-based detection

    # Price drop thresholds (percentage change over lookback)
    price_drop_lookback: int = 24  # hours
    price_drop_elevated: float = -0.05  # -5% in 24h triggers ELEVATED
    price_drop_extreme: float = -0.10   # -10% in 24h triggers EXTREME

    # NEW: Drawdown from recent high
    drawdown_lookback: int = 72  # 3 days
    drawdown_elevated: float = -0.10   # -10% from 72h high
    drawdown_extreme: float = -0.20    # -20% from 72h high
    use_drawdown: bool = True

    # Funding rate thresholds (if available)
    funding_extreme_threshold: float = 0.001  # |0.1%| per 8h triggers EXTREME

    # Trade frequency control per regime
    max_trades_per_day_normal: int = 50  # Normal regime
    max_trades_per_day_elevated: int = 10  # Reduced during elevated
    max_trades_per_day_extreme: int = 0   # No trading in extreme

    # Position scaling per regime
    position_scale_normal: float = 1.0
    position_scale_elevated: float = 0.5
    position_scale_extreme: float = 0.0

    # Additional safeguards
    consecutive_drops_threshold: int = 3  # Number of consecutive -2% hours for EXTREME
    require_recovery_hours: int = 6  # Hours of calm before exiting EXTREME

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RegimeConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RegimePresets:
    """Pre-configured regime filter settings."""

    @staticmethod
    def conservative() -> RegimeConfig:
        """
        Conservative settings - triggers regime changes earlier.
        Best for risk-averse strategies or uncertain market conditions.
        """
        return RegimeConfig(
            vol_elevated_threshold=1.3,
            vol_extreme_threshold=2.0,
            vol_absolute_elevated=0.03,  # 3% daily std
            vol_absolute_extreme=0.045,  # 4.5% daily std
            vol_percentile_elevated=0.70,
            vol_percentile_extreme=0.90,
            price_drop_elevated=-0.03,
            price_drop_extreme=-0.07,
            drawdown_elevated=-0.07,
            drawdown_extreme=-0.15,
            max_trades_per_day_elevated=5,
            max_trades_per_day_extreme=0,
            position_scale_elevated=0.3,
        )

    @staticmethod
    def moderate() -> RegimeConfig:
        """
        Moderate settings - balanced approach.
        Default configuration suitable for most strategies.
        """
        return RegimeConfig()  # Uses default values

    @staticmethod
    def aggressive() -> RegimeConfig:
        """
        Aggressive settings - allows more trading during stress.
        Use only if strategy has proven resilience to volatility.
        """
        return RegimeConfig(
            vol_elevated_threshold=2.0,
            vol_extreme_threshold=3.0,
            vol_absolute_elevated=0.05,  # 5% daily std
            vol_absolute_extreme=0.08,   # 8% daily std
            vol_percentile_elevated=0.90,
            vol_percentile_extreme=0.98,
            price_drop_elevated=-0.07,
            price_drop_extreme=-0.15,
            drawdown_elevated=-0.15,
            drawdown_extreme=-0.30,
            max_trades_per_day_elevated=20,
            max_trades_per_day_extreme=5,
            position_scale_elevated=0.7,
            position_scale_extreme=0.2,
        )


class RegimeFilter:
    """
    Market regime detector and trade filter.

    This class provides:
    1. Real-time regime detection from price/funding data
    2. Trade gating based on current regime
    3. Position size scaling
    4. Daily trade counting with regime-specific limits

    Example:
        config = RegimeConfig(vol_extreme_threshold=2.5)
        filter = RegimeFilter(config)

        # Detect regime
        regime, metrics = filter.detect_regime(prices, funding_rates)

        # Check if trade is allowed
        allowed, reason = filter.should_allow_trade(regime, current_timestamp)

        if allowed:
            # Execute trade with scaled position
            scale = filter.get_position_scale(regime)
            position_size = base_size * scale
            filter.record_trade()
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize the regime filter.

        Args:
            config: Regime configuration. Uses defaults if not provided.
        """
        self.config = config or RegimeConfig()
        self.trades_today = 0
        self.last_trade_date = None
        self.current_regime = MarketRegime.NORMAL
        self.regime_start_time = None
        self.extreme_cooldown_remaining = 0

    def calculate_volatility_ratio(self, prices: pd.Series) -> float:
        """
        Calculate current volatility relative to baseline.

        Uses rolling standard deviation of log returns to measure volatility.
        Returns the ratio of current (short-term) volatility to baseline
        (longer-term) volatility.

        Args:
            prices: Price series (should have at least vol_baseline_lookback periods)

        Returns:
            Ratio of current volatility to baseline (1.0 = normal)
        """
        min_periods = self.config.vol_baseline_lookback
        if len(prices) < min_periods:
            return 1.0  # Not enough data, assume normal

        # Calculate log returns
        returns = np.log(prices / prices.shift(1)).dropna()

        if len(returns) < min_periods:
            return 1.0

        # Current volatility (recent window)
        current_vol = returns.tail(self.config.vol_lookback).std()

        # Baseline volatility (longer window)
        baseline_vol = returns.tail(self.config.vol_baseline_lookback).std()

        if baseline_vol == 0 or np.isnan(baseline_vol):
            return 1.0

        ratio = current_vol / baseline_vol
        return float(ratio) if not np.isnan(ratio) else 1.0

    def calculate_price_change(self, prices: pd.Series) -> float:
        """
        Calculate price change over lookback period.

        Args:
            prices: Price series

        Returns:
            Percentage change (e.g., -0.10 for -10%)
        """
        lookback = self.config.price_drop_lookback
        if len(prices) < lookback:
            return 0.0

        current_price = prices.iloc[-1]
        lookback_price = prices.iloc[-lookback]

        if lookback_price == 0 or np.isnan(lookback_price):
            return 0.0

        change = (current_price - lookback_price) / lookback_price
        return float(change) if not np.isnan(change) else 0.0

    def count_consecutive_drops(self, prices: pd.Series, threshold: float = -0.02) -> int:
        """
        Count consecutive hourly drops below threshold.

        Args:
            prices: Price series
            threshold: Drop threshold per hour (default -2%)

        Returns:
            Number of consecutive drops
        """
        if len(prices) < 2:
            return 0

        returns = prices.pct_change().dropna()

        if len(returns) == 0:
            return 0

        # Count from most recent going backwards
        count = 0
        for ret in returns.iloc[::-1]:
            if ret < threshold:
                count += 1
            else:
                break
        return count

    def detect_regime(
        self,
        prices: pd.Series,
        funding_rates: Optional[pd.Series] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[MarketRegime, Dict[str, Any]]:
        """
        Detect current market regime.

        Evaluates multiple indicators and returns the most severe regime
        triggered by any of them (worst signal wins).

        Args:
            prices: Historical price series ending at current time
            funding_rates: Optional funding rate series
            timestamp: Current timestamp (for logging and tracking)

        Returns:
            Tuple of (regime, metrics_dict) where metrics_dict contains:
            - volatility_ratio: Current vol / baseline vol
            - price_change_24h: 24-hour price change
            - funding_rate: Current funding rate (if available)
            - regime: Detected regime value
            - trigger: Which indicator triggered the regime
        """
        metrics: Dict[str, Any] = {
            'timestamp': timestamp,
            'trigger': 'none',
        }

        # Calculate volatility ratio
        vol_ratio = self.calculate_volatility_ratio(prices)
        metrics['volatility_ratio'] = vol_ratio

        # Calculate price change
        price_change = self.calculate_price_change(prices)
        metrics['price_change_24h'] = price_change

        # Count consecutive drops
        consecutive_drops = self.count_consecutive_drops(prices)
        metrics['consecutive_drops'] = consecutive_drops

        # Check funding rates if available
        funding_signal = False
        if funding_rates is not None and len(funding_rates) > 0:
            current_funding = funding_rates.iloc[-1]
            if not np.isnan(current_funding):
                metrics['funding_rate'] = float(current_funding)
                funding_signal = abs(current_funding) > self.config.funding_extreme_threshold

        # Determine regime (worst signal wins)
        regime = MarketRegime.NORMAL

        # Check for EXTREME conditions
        if vol_ratio >= self.config.vol_extreme_threshold:
            regime = MarketRegime.EXTREME
            metrics['trigger'] = 'volatility_extreme'
        elif price_change <= self.config.price_drop_extreme:
            regime = MarketRegime.EXTREME
            metrics['trigger'] = 'price_drop_extreme'
        elif funding_signal:
            regime = MarketRegime.EXTREME
            metrics['trigger'] = 'funding_extreme'
        elif consecutive_drops >= self.config.consecutive_drops_threshold:
            regime = MarketRegime.EXTREME
            metrics['trigger'] = 'consecutive_drops'

        # Check for ELEVATED conditions (only if not already EXTREME)
        elif vol_ratio >= self.config.vol_elevated_threshold:
            regime = MarketRegime.ELEVATED
            metrics['trigger'] = 'volatility_elevated'
        elif price_change <= self.config.price_drop_elevated:
            regime = MarketRegime.ELEVATED
            metrics['trigger'] = 'price_drop_elevated'

        metrics['regime'] = regime.value

        # Track regime changes
        if regime != self.current_regime:
            logger.info(f"Regime changed: {self.current_regime.value} -> {regime.value} "
                       f"(trigger: {metrics['trigger']})")
            self.current_regime = regime
            self.regime_start_time = timestamp

            # Set cooldown when entering extreme
            if regime == MarketRegime.EXTREME:
                self.extreme_cooldown_remaining = self.config.require_recovery_hours

        return regime, metrics

    def should_allow_trade(
        self,
        regime: MarketRegime,
        current_date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be allowed given current regime.

        Considers:
        1. Regime-specific trade limits
        2. Daily trade count
        3. Cooldown periods after extreme regimes

        Args:
            regime: Current market regime
            current_date: Current timestamp for trade counting

        Returns:
            Tuple of (should_trade: bool, reason: str)
        """
        # Reset daily counter if new day
        current_day = current_date.date() if hasattr(current_date, 'date') else current_date
        if self.last_trade_date != current_day:
            self.trades_today = 0
            self.last_trade_date = current_day

        # Get max trades for this regime
        if regime == MarketRegime.EXTREME:
            max_trades = self.config.max_trades_per_day_extreme
            if max_trades == 0:
                return False, "BLOCKED: Extreme regime - no trading allowed"
        elif regime == MarketRegime.ELEVATED:
            max_trades = self.config.max_trades_per_day_elevated
        else:
            max_trades = self.config.max_trades_per_day_normal

        # Check cooldown
        if self.extreme_cooldown_remaining > 0:
            return False, f"BLOCKED: Cooldown after extreme regime ({self.extreme_cooldown_remaining}h remaining)"

        # Check trade count
        if self.trades_today >= max_trades:
            return False, f"BLOCKED: Max trades ({max_trades}) reached for {regime.value} regime"

        return True, f"ALLOWED: {regime.value} regime, {self.trades_today}/{max_trades} trades today"

    def record_trade(self):
        """Record that a trade was executed."""
        self.trades_today += 1

    def update_cooldown(self, hours: int = 1):
        """
        Update cooldown timer (call once per hour).

        Args:
            hours: Number of hours to decrement from cooldown
        """
        if self.extreme_cooldown_remaining > 0:
            self.extreme_cooldown_remaining = max(0, self.extreme_cooldown_remaining - hours)

    def get_position_scale(self, regime: MarketRegime) -> float:
        """
        Get position size multiplier for current regime.

        Args:
            regime: Current market regime

        Returns:
            Position scale factor (0.0 to 1.0)
        """
        if regime == MarketRegime.EXTREME:
            return self.config.position_scale_extreme
        elif regime == MarketRegime.ELEVATED:
            return self.config.position_scale_elevated
        else:
            return self.config.position_scale_normal

    def reset_state(self):
        """Reset internal state (for new backtest runs)."""
        self.trades_today = 0
        self.last_trade_date = None
        self.current_regime = MarketRegime.NORMAL
        self.regime_start_time = None
        self.extreme_cooldown_remaining = 0


def apply_regime_filter_to_predictions(
    predictions: pd.DataFrame,
    prices: pd.Series,
    funding_rates: Optional[pd.Series] = None,
    config: Optional[RegimeConfig] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply regime filter to model predictions.

    This is the main integration function for backtesting. It takes a DataFrame
    of model predictions and filters them based on detected market regimes.

    Args:
        predictions: DataFrame with at minimum a 'timestamp' column and a signal column.
                    Expected columns: ['timestamp', 'should_trade', 'is_long', ...]
        prices: Price series with DatetimeIndex, used for regime detection
        funding_rates: Optional funding rate series with DatetimeIndex
        config: Regime filter configuration
        verbose: If True, log regime changes and blocked trades

    Returns:
        predictions DataFrame with additional columns:
        - 'regime': detected regime for each row
        - 'original_should_trade': preserved original trade signal
        - 'should_trade': filtered signal (may be set to False)
        - 'position_scale': position size multiplier
        - 'filter_reason': why signal was modified
        - 'regime_blocked': True if trade was blocked by regime filter
    """
    regime_filter = RegimeFilter(config)

    # Make a copy to avoid modifying original
    result = predictions.copy()

    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in result.columns:
        raise ValueError("predictions DataFrame must have 'timestamp' column")

    result['timestamp'] = pd.to_datetime(result['timestamp'])

    # Preserve original signals
    if 'should_trade' in result.columns:
        result['original_should_trade'] = result['should_trade'].copy()
    else:
        # If no should_trade column, assume all rows are potential trades
        result['should_trade'] = True
        result['original_should_trade'] = True

    # Initialize new columns
    result['regime'] = MarketRegime.NORMAL.value
    result['position_scale'] = 1.0
    result['filter_reason'] = ''
    result['regime_blocked'] = False

    # Ensure prices has DatetimeIndex for alignment
    if not isinstance(prices.index, pd.DatetimeIndex):
        if 'timestamp' in prices.index.names or isinstance(prices.index[0], (pd.Timestamp, str)):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)

    # Track statistics
    stats = {
        'total_predictions': len(result),
        'original_trades': result['original_should_trade'].sum(),
        'blocked_by_regime': 0,
        'regime_normal': 0,
        'regime_elevated': 0,
        'regime_extreme': 0,
    }

    for idx in result.index:
        timestamp = result.loc[idx, 'timestamp']

        # Get price history up to this point
        try:
            # Handle different index types
            if isinstance(prices.index, pd.DatetimeIndex):
                price_history = prices[prices.index <= timestamp]
            else:
                price_history = prices.iloc[:idx+1] if idx < len(prices) else prices
        except (KeyError, TypeError):
            # Fallback: use position-based slicing
            price_history = prices.iloc[:min(idx+1, len(prices))]

        # Get funding history if available
        funding_history = None
        if funding_rates is not None:
            try:
                if isinstance(funding_rates.index, pd.DatetimeIndex):
                    funding_history = funding_rates[funding_rates.index <= timestamp]
                else:
                    funding_history = funding_rates.iloc[:idx+1] if idx < len(funding_rates) else funding_rates
            except (KeyError, TypeError):
                funding_history = None

        # Skip if insufficient price history
        if len(price_history) < 24:
            continue

        # Detect regime
        regime, metrics = regime_filter.detect_regime(price_history, funding_history, timestamp)
        result.loc[idx, 'regime'] = regime.value

        # Update regime stats
        if regime == MarketRegime.NORMAL:
            stats['regime_normal'] += 1
        elif regime == MarketRegime.ELEVATED:
            stats['regime_elevated'] += 1
        else:
            stats['regime_extreme'] += 1

        # Only check trade gating if this was originally a trade
        if result.loc[idx, 'original_should_trade']:
            # Check if trade should be allowed
            should_trade, reason = regime_filter.should_allow_trade(regime, timestamp)
            result.loc[idx, 'filter_reason'] = reason

            # Get position scale
            scale = regime_filter.get_position_scale(regime)
            result.loc[idx, 'position_scale'] = scale

            # Block trade if not allowed or scale is 0
            if not should_trade or scale == 0:
                result.loc[idx, 'should_trade'] = False
                result.loc[idx, 'regime_blocked'] = True
                stats['blocked_by_regime'] += 1

                if verbose:
                    logger.debug(f"Trade blocked at {timestamp}: {reason}")
            else:
                # Trade allowed - record it
                regime_filter.record_trade()

    # Log summary statistics
    if verbose:
        logger.info(f"Regime Filter Summary:")
        logger.info(f"  Total predictions: {stats['total_predictions']}")
        logger.info(f"  Original trades: {stats['original_trades']}")
        logger.info(f"  Blocked by regime: {stats['blocked_by_regime']}")
        logger.info(f"  Regime distribution: Normal={stats['regime_normal']}, "
                   f"Elevated={stats['regime_elevated']}, Extreme={stats['regime_extreme']}")

    return result


def apply_regime_filter_vectorized(
    predictions: pd.DataFrame,
    prices: pd.Series,
    funding_rates: Optional[pd.Series] = None,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """
    Vectorized regime filter for large datasets.

    This is a faster implementation that pre-computes regime indicators
    for all timestamps, then applies thresholds vectorized.

    Uses multiple detection methods:
    1. Relative volatility ratio (current vs baseline)
    2. Absolute volatility thresholds
    3. Historical volatility percentiles
    4. Price drops over lookback period
    5. Drawdown from recent high
    6. Extreme funding rates (if available)

    Trade count limits are NOT enforced in this version (for speed).
    Use apply_regime_filter_to_predictions for full functionality.

    Args:
        predictions: DataFrame with 'timestamp' and 'should_trade' columns
        prices: Price series aligned with predictions
        funding_rates: Optional funding rate series
        config: Regime filter configuration

    Returns:
        predictions DataFrame with regime columns added
    """
    cfg = config or RegimeConfig()
    result = predictions.copy()

    # Ensure timestamp is datetime
    result['timestamp'] = pd.to_datetime(result['timestamp'])

    # Preserve original signals
    if 'should_trade' in result.columns:
        result['original_should_trade'] = result['should_trade'].copy()
    else:
        result['should_trade'] = True
        result['original_should_trade'] = True

    # Compute log returns
    log_returns = np.log(prices / prices.shift(1))

    # 1. Compute rolling volatility and ratio
    current_vol = log_returns.rolling(window=cfg.vol_lookback, min_periods=cfg.vol_lookback).std()
    baseline_vol = log_returns.rolling(window=cfg.vol_baseline_lookback, min_periods=cfg.vol_baseline_lookback).std()
    vol_ratio = (current_vol / baseline_vol).fillna(1.0)

    # 2. Compute absolute volatility (already in current_vol, it's daily std of log returns)
    abs_vol = current_vol.fillna(0.0)

    # 3. Compute historical volatility percentile (expanding window)
    # This compares current vol to ALL historical vol, not just recent
    vol_percentile = current_vol.expanding(min_periods=720).apply(
        lambda x: (x[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # 4. Compute price change
    price_change = prices.pct_change(periods=cfg.price_drop_lookback).fillna(0.0)

    # 5. Compute drawdown from recent high
    if cfg.use_drawdown:
        rolling_high = prices.rolling(window=cfg.drawdown_lookback, min_periods=1).max()
        drawdown = (prices - rolling_high) / rolling_high
        drawdown = drawdown.fillna(0.0)
    else:
        drawdown = pd.Series(0.0, index=prices.index)

    # Initialize regime as NORMAL
    result['regime'] = MarketRegime.NORMAL.value
    result['position_scale'] = cfg.position_scale_normal
    result['regime_blocked'] = False
    result['filter_reason'] = 'ALLOWED: normal regime'

    # Align indices - create mapping from prediction timestamps to price indices
    def align_series(series, timestamps, default_value):
        try:
            aligned = series.reindex(timestamps, method='ffill').fillna(default_value)
            return aligned
        except Exception:
            # Fallback: use position-based alignment
            vals = series.values[:len(timestamps)] if len(series) >= len(timestamps) else \
                   np.pad(series.values, (0, len(timestamps) - len(series)), constant_values=default_value)
            return pd.Series(vals, index=result.index)

    vol_ratio_aligned = align_series(vol_ratio, result['timestamp'], 1.0)
    abs_vol_aligned = align_series(abs_vol, result['timestamp'], 0.0)
    vol_pctl_aligned = align_series(vol_percentile, result['timestamp'], 0.5)
    price_change_aligned = align_series(price_change, result['timestamp'], 0.0)
    drawdown_aligned = align_series(drawdown, result['timestamp'], 0.0)

    # Build EXTREME detection mask (any of these conditions)
    extreme_conditions = []

    # Relative volatility ratio
    extreme_conditions.append(vol_ratio_aligned >= cfg.vol_extreme_threshold)

    # Absolute volatility
    if cfg.use_absolute_vol:
        extreme_conditions.append(abs_vol_aligned >= cfg.vol_absolute_extreme)

    # Historical percentile
    if cfg.use_percentile_vol:
        extreme_conditions.append(vol_pctl_aligned >= cfg.vol_percentile_extreme)

    # Price drop
    extreme_conditions.append(price_change_aligned <= cfg.price_drop_extreme)

    # Drawdown from high
    if cfg.use_drawdown:
        extreme_conditions.append(drawdown_aligned <= cfg.drawdown_extreme)

    # Combine all extreme conditions with OR
    extreme_mask = pd.Series(False, index=result.index)
    for cond in extreme_conditions:
        extreme_mask = extreme_mask | cond

    result.loc[extreme_mask, 'regime'] = MarketRegime.EXTREME.value
    result.loc[extreme_mask, 'position_scale'] = cfg.position_scale_extreme
    result.loc[extreme_mask, 'filter_reason'] = 'BLOCKED: extreme regime'

    # Block trades in extreme regime
    extreme_trade_mask = extreme_mask & result['original_should_trade']
    result.loc[extreme_trade_mask, 'should_trade'] = False
    result.loc[extreme_trade_mask, 'regime_blocked'] = True

    # Build ELEVATED detection mask (where not already EXTREME)
    elevated_conditions = []

    # Relative volatility ratio
    elevated_conditions.append(vol_ratio_aligned >= cfg.vol_elevated_threshold)

    # Absolute volatility
    if cfg.use_absolute_vol:
        elevated_conditions.append(abs_vol_aligned >= cfg.vol_absolute_elevated)

    # Historical percentile
    if cfg.use_percentile_vol:
        elevated_conditions.append(vol_pctl_aligned >= cfg.vol_percentile_elevated)

    # Price drop
    elevated_conditions.append(price_change_aligned <= cfg.price_drop_elevated)

    # Drawdown from high
    if cfg.use_drawdown:
        elevated_conditions.append(drawdown_aligned <= cfg.drawdown_elevated)

    # Combine all elevated conditions with OR
    elevated_cond = pd.Series(False, index=result.index)
    for cond in elevated_conditions:
        elevated_cond = elevated_cond | cond

    elevated_mask = ~extreme_mask & elevated_cond

    result.loc[elevated_mask, 'regime'] = MarketRegime.ELEVATED.value
    result.loc[elevated_mask, 'position_scale'] = cfg.position_scale_elevated
    result.loc[elevated_mask, 'filter_reason'] = 'ALLOWED: elevated regime, reduced position'

    # Add funding rate check if available
    if funding_rates is not None and len(funding_rates) > 0:
        try:
            funding_aligned = funding_rates.reindex(result['timestamp'], method='ffill').fillna(0.0)
        except Exception:
            funding_aligned = pd.Series(0.0, index=result.index)

        funding_extreme = funding_aligned.abs() > cfg.funding_extreme_threshold
        funding_extreme_mask = funding_extreme & ~extreme_mask  # Only where not already extreme

        result.loc[funding_extreme_mask, 'regime'] = MarketRegime.EXTREME.value
        result.loc[funding_extreme_mask, 'position_scale'] = cfg.position_scale_extreme
        result.loc[funding_extreme_mask, 'filter_reason'] = 'BLOCKED: extreme funding rate'

        # Block trades
        funding_trade_mask = funding_extreme_mask & result['original_should_trade']
        result.loc[funding_trade_mask, 'should_trade'] = False
        result.loc[funding_trade_mask, 'regime_blocked'] = True

    return result
