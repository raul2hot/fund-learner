"""
edge_cases.py - Robust handling of data quality issues
Version: 2.0
Date: December 2025

Handles:
- Missing OHLCV candles (exchange downtime)
- Missing funding rate records
- Gaps in Fear & Greed data
- Extreme/invalid values
- Data validation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """
    Handles edge cases in crypto market data.
    
    Common issues:
    1. Missing OHLCV candles (exchange downtime)
    2. Missing funding rate records
    3. Gaps in Fear & Greed data
    4. Extreme/invalid values
    5. Timezone inconsistencies
    """
    
    MAX_OHLCV_GAP_HOURS = 4
    MAX_FUNDING_GAP_HOURS = 24
    MAX_FNG_GAP_DAYS = 3
    
    def __init__(self):
        self.issues_found = []
    
    def detect_gaps(self, df: pd.DataFrame, expected_freq: str = '1h') -> List[Dict]:
        """
        Detect gaps in time series data.
        
        Returns list of gaps with start, end, and duration.
        """
        if df.empty or 'timestamp' not in df.columns:
            return []
        
        df = df.sort_values('timestamp')
        
        freq_map = {
            '1h': pd.Timedelta(hours=1),
            '8h': pd.Timedelta(hours=8),
            '1d': pd.Timedelta(days=1)
        }
        expected_diff = freq_map.get(expected_freq, pd.Timedelta(hours=1))
        
        gaps = []
        timestamps = df['timestamp'].values
        
        for i in range(1, len(timestamps)):
            actual_diff = pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i-1])
            
            if actual_diff > expected_diff * 1.5:
                gap = {
                    'start': pd.Timestamp(timestamps[i-1]),
                    'end': pd.Timestamp(timestamps[i]),
                    'duration': actual_diff,
                    'missing_periods': int(actual_diff / expected_diff) - 1
                }
                gaps.append(gap)
        
        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in data")
            for gap in gaps[:5]:
                logger.warning(f"  Gap: {gap['start']} to {gap['end']} ({gap['missing_periods']} periods)")
        
        return gaps
    
    def fill_ohlcv_gaps(self, df: pd.DataFrame, max_gap_hours: int = 4) -> pd.DataFrame:
        """
        Fill small gaps in OHLCV data.
        
        Strategy:
        - Forward-fill OHLC prices
        - Zero-fill volumes during gaps
        """
        if df.empty:
            return df
        
        df = df.copy().sort_values('timestamp')
        
        full_index = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='1h',
            tz='UTC'
        )
        
        df = df.set_index('timestamp').reindex(full_index)
        
        was_missing = df['close'].isna()
        
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        volume_cols = ['volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in volume_cols:
            if col in df.columns:
                df.loc[was_missing, col] = 0
        
        if 'trades' in df.columns:
            df.loc[was_missing, 'trades'] = 0
        
        filled = was_missing.sum()
        if filled > 0:
            logger.info(f"Filled {filled} missing OHLCV rows ({filled/len(df)*100:.2f}%)")
        
        return df.reset_index().rename(columns={'index': 'timestamp'})
    
    def handle_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle extreme/invalid values.
        
        Checks:
        - Negative prices (invalid)
        - Zero prices (invalid)
        - Negative volumes (invalid)
        - Extreme funding rates
        - Fear & Greed out of range
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Price validation
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_mask = df[col] <= 0
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid values in {col}")
                    df.loc[invalid_mask, col] = np.nan
        
        # Volume validation
        volume_cols = ['volume', 'taker_buy_volume']
        for col in volume_cols:
            if col in df.columns:
                invalid_mask = df[col] < 0
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} negative values in {col}")
                    df.loc[invalid_mask, col] = np.nan
        
        # Extreme price change detection (informational)
        if 'close' in df.columns:
            pct_change = df['close'].pct_change().abs()
            extreme_changes = pct_change > 0.5
            if extreme_changes.any():
                logger.warning(f"Found {extreme_changes.sum()} extreme price changes (>50% in 1h)")
        
        # Funding rate validation
        if 'funding_rate' in df.columns:
            invalid_mask = (df['funding_rate'].abs() > 1)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} extreme funding rates")
        
        # Fear & Greed validation
        if 'fear_greed_value' in df.columns:
            invalid_mask = (df['fear_greed_value'] < 0) | (df['fear_greed_value'] > 100)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} invalid Fear & Greed values")
                df.loc[invalid_mask, 'fear_greed_value'] = np.nan
        
        return df
    
    def interpolate_missing(
        self,
        df: pd.DataFrame,
        method: str = 'ffill',
        limit: int = 4
    ) -> pd.DataFrame:
        """
        Interpolate remaining NaN values.
        
        Parameters:
        -----------
        method : str - 'linear', 'ffill', or 'bfill'
        limit : int - Maximum consecutive NaNs to fill
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'timestamp':
                continue
            
            before_nans = df[col].isna().sum()
            
            if method == 'linear':
                df[col] = df[col].interpolate(method='linear', limit=limit)
            elif method == 'ffill':
                df[col] = df[col].ffill(limit=limit)
            elif method == 'bfill':
                df[col] = df[col].bfill(limit=limit)
            
            after_nans = df[col].isna().sum()
            
            if before_nans > after_nans:
                logger.debug(f"Interpolated {before_nans - after_nans} values in {col}")
        
        return df
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Final validation of data integrity.
        
        Returns:
        --------
        Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check timestamp is sorted
        if 'timestamp' in df.columns and not df['timestamp'].is_monotonic_increasing:
            issues.append("Timestamps are not sorted")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate timestamps")
        
        # Check for remaining NaNs in critical columns
        for col in ['close', 'volume']:
            if col in df.columns:
                nans = df[col].isna().sum()
                if nans > 0:
                    issues.append(f"Column {col} has {nans} NaN values ({nans/len(df)*100:.2f}%)")
        
        # Check OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            inconsistent = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            if inconsistent > 0:
                issues.append(f"Found {inconsistent} OHLC consistency violations")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues


def run_all_validations(df: pd.DataFrame) -> dict:
    """
    Run comprehensive data quality validations.
    
    Returns dict with validation results.
    """
    results = {
        'passed': True,
        'checks': []
    }
    
    # Check 1: No duplicate timestamps
    check = {
        'name': 'No duplicate timestamps',
        'passed': not df['timestamp'].duplicated().any(),
        'details': f"Duplicates found: {df['timestamp'].duplicated().sum()}"
    }
    results['checks'].append(check)
    
    # Check 2: Timestamps are sorted
    check = {
        'name': 'Timestamps are sorted',
        'passed': df['timestamp'].is_monotonic_increasing,
        'details': None
    }
    results['checks'].append(check)
    
    # Check 3: No large gaps
    time_diffs = df['timestamp'].diff()
    max_gap = time_diffs.max()
    check = {
        'name': 'No large gaps (> 4 hours)',
        'passed': pd.isna(max_gap) or max_gap <= pd.Timedelta(hours=4),
        'details': f"Max gap: {max_gap}"
    }
    results['checks'].append(check)
    
    # Check 4: OHLC consistency
    ohlc_valid = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ).all()
    check = {
        'name': 'OHLC consistency',
        'passed': ohlc_valid,
        'details': None
    }
    results['checks'].append(check)
    
    # Check 5: No negative volumes
    check = {
        'name': 'No negative volumes',
        'passed': (df['volume'] >= 0).all(),
        'details': f"Negative values: {(df['volume'] < 0).sum()}"
    }
    results['checks'].append(check)
    
    # Check 6: Funding rate within bounds
    if 'funding_rate' in df.columns:
        check = {
            'name': 'Funding rate within [-0.75%, 0.75%]',
            'passed': ((df['funding_rate'].abs() <= 0.0075) | df['funding_rate'].isna()).all(),
            'details': f"Out of bounds: {(df['funding_rate'].abs() > 0.0075).sum()}"
        }
        results['checks'].append(check)
    
    # Check 7: Fear & Greed within range
    if 'fear_greed_value' in df.columns:
        check = {
            'name': 'Fear & Greed within [0, 100]',
            'passed': ((df['fear_greed_value'] >= 0) & (df['fear_greed_value'] <= 100) | df['fear_greed_value'].isna()).all(),
            'details': None
        }
        results['checks'].append(check)
    
    # Check 8: Minimum data points
    min_rows = 8760  # 1 year
    check = {
        'name': f'Minimum rows ({min_rows:,})',
        'passed': len(df) >= min_rows,
        'details': f"Actual rows: {len(df):,}"
    }
    results['checks'].append(check)
    
    # Update overall status
    results['passed'] = all(c['passed'] for c in results['checks'])
    
    return results


def print_validation_report(results: dict):
    """Print formatted validation report."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    for check in results['checks']:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"{status}: {check['name']}")
        if check['details'] and not check['passed']:
            print(f"       {check['details']}")
    
    print("-" * 60)
    overall = "✓ ALL CHECKS PASSED" if results['passed'] else "✗ SOME CHECKS FAILED"
    print(f"{overall}")
    print("=" * 60)
