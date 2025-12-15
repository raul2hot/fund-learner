"""
data_processor.py - Data Alignment and Preprocessing
Version: 2.0
Date: December 2025

Handles:
- Timestamp alignment across different data frequencies
- Forward-filling for lower-frequency data
- Derived feature computation
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles alignment, merging, and preprocessing of multi-source data.
    
    Key responsibilities:
    - Align all data to hourly UTC timestamps
    - Forward-fill lower-frequency data (funding, fear/greed)
    - Handle missing values and gaps
    - Validate data integrity
    """
    
    def create_hourly_index(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Create complete hourly datetime index."""
        start = pd.Timestamp(start_date, tz='UTC').normalize()
        end = pd.Timestamp(end_date, tz='UTC').normalize() + pd.Timedelta(days=1)
        return pd.date_range(start=start, end=end, freq='1h', tz='UTC')[:-1]
    
    def align_funding_rate(
        self,
        funding_df: pd.DataFrame,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align 8-hourly funding rate to hourly timestamps.
        
        Strategy: Forward-fill
        - Funding rate at 00:00 applies to hours 00-07
        - Funding rate at 08:00 applies to hours 08-15
        - Funding rate at 16:00 applies to hours 16-23
        """
        if funding_df.empty:
            return pd.DataFrame(index=target_index, columns=['funding_rate'])
        
        funding_indexed = funding_df.set_index('timestamp')
        aligned = funding_indexed[['funding_rate']].reindex(target_index)
        aligned = aligned.ffill()
        
        # Handle leading NaNs
        first_valid = aligned['funding_rate'].first_valid_index()
        if first_valid is not None:
            aligned.loc[:first_valid, 'funding_rate'] = aligned.loc[first_valid, 'funding_rate']
        
        logger.info(f"Aligned funding rate: {aligned['funding_rate'].notna().sum()}/{len(aligned)} valid values")
        
        return aligned.reset_index().rename(columns={'index': 'timestamp'})
    
    def align_fear_greed(
        self,
        fng_df: pd.DataFrame,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align daily Fear & Greed index to hourly timestamps.
        
        Strategy: Forward-fill
        - Daily value at 00:00 UTC applies to all 24 hours of that day
        """
        if fng_df.empty:
            return pd.DataFrame(index=target_index, columns=['fear_greed_value'])
        
        fng_indexed = fng_df.set_index('timestamp')
        aligned = fng_indexed[['fear_greed_value']].reindex(target_index)
        aligned = aligned.ffill()
        
        # Handle leading NaNs
        first_valid = aligned['fear_greed_value'].first_valid_index()
        if first_valid is not None:
            aligned.loc[:first_valid, 'fear_greed_value'] = aligned.loc[first_valid, 'fear_greed_value']
        
        logger.info(f"Aligned Fear & Greed: {aligned['fear_greed_value'].notna().sum()}/{len(aligned)} valid values")
        
        return aligned.reset_index().rename(columns={'index': 'timestamp'})
    
    def merge_all_data(
        self,
        ohlcv_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        fng_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Merge all data sources into a single aligned DataFrame.
        
        Parameters:
        -----------
        ohlcv_df : DataFrame with OHLCV + taker volume
        funding_df : DataFrame with funding rate
        fng_df : DataFrame with Fear & Greed index
        start_date : Start date (inclusive)
        end_date : End date (inclusive)
        
        Returns:
        --------
        Merged DataFrame with all features aligned to hourly timestamps
        """
        target_index = self.create_hourly_index(start_date, end_date)
        
        if ohlcv_df.empty:
            raise ValueError("OHLCV data is required")
        
        ohlcv_indexed = ohlcv_df.set_index('timestamp')
        merged = ohlcv_indexed.reindex(target_index)
        
        # Align and merge funding rate
        aligned_funding = self.align_funding_rate(funding_df, target_index)
        aligned_funding = aligned_funding.set_index('timestamp')
        merged = merged.join(aligned_funding)
        
        # Align and merge Fear & Greed
        aligned_fng = self.align_fear_greed(fng_df, target_index)
        aligned_fng = aligned_fng.set_index('timestamp')
        merged = merged.join(aligned_fng)
        
        merged = merged.reset_index().rename(columns={'index': 'timestamp'})
        
        self._log_merge_stats(merged)
        
        return merged
    
    def _log_merge_stats(self, df: pd.DataFrame):
        """Log statistics about the merged data."""
        total_rows = len(df)
        
        for col in df.columns:
            if col == 'timestamp':
                continue
            missing = df[col].isna().sum()
            pct = (missing / total_rows) * 100
            if missing > 0:
                logger.warning(f"Column '{col}': {missing} missing values ({pct:.2f}%)")
            else:
                logger.info(f"Column '{col}': Complete (0 missing)")
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features for the 10 filters.
        
        Features computed:
        - taker_buy_ratio: Taker buy volume / Total volume
        - taker_sell_volume: Total - Taker buy
        - order_flow_imbalance: (Buy - Sell) / Total
        - spread_proxy: (High - Low) / Close
        - session: Trading session (asian/london/new_york)
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        """
        df = df.copy()
        
        # Order Flow Imbalance features
        df['taker_buy_ratio'] = np.where(
            df['volume'] > 0,
            df['taker_buy_volume'] / df['volume'],
            0.5  # Default to neutral if no volume
        )
        df['taker_sell_volume'] = df['volume'] - df['taker_buy_volume']
        df['order_flow_imbalance'] = np.where(
            df['volume'] > 0,
            (df['taker_buy_volume'] - df['taker_sell_volume']) / df['volume'],
            0.0
        )
        
        # Liquidity/Spread proxy
        df['spread_proxy'] = np.where(
            df['close'] > 0,
            (df['high'] - df['low']) / df['close'],
            0.0
        )
        
        # Session/Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Session classification
        def classify_session(hour):
            if 0 <= hour < 8:
                return 'asian'
            elif 8 <= hour < 16:
                return 'london'
            else:
                return 'new_york'
        
        df['session'] = df['hour'].apply(classify_session)
        
        # Validate derived features
        self._validate_derived_features(df)
        
        return df
    
    def _validate_derived_features(self, df: pd.DataFrame):
        """Validate derived features are within expected ranges."""
        validations = [
            ('taker_buy_ratio', 0, 1, 'Taker buy ratio should be between 0 and 1'),
            ('order_flow_imbalance', -1, 1, 'Order flow imbalance should be between -1 and 1'),
        ]
        
        for col, min_val, max_val, msg in validations:
            if col in df.columns:
                invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid > 0:
                    logger.warning(f"{msg}: {invalid} violations found")
