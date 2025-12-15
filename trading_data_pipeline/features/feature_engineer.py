"""
Feature engineering for all 10 filter categories.
All features are computed using only past data (no future leakage).

Filter Categories:
1. Volatility Features
2. K-Means Support/Resistance Levels
3. HMM Regime Detection
4. Order Flow Imbalance
5. Funding Rate Features
6. Liquidity Score
7. Fear & Greed Features
8. Session/Time Features
9. Trend Strength Features
10. Momentum Exhaustion Features
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings


class FeatureEngineer:
    """
    Compute all filter features from aligned data.

    All features are computed using only past data to prevent look-ahead bias.
    """

    def __init__(self, aligned_data: pd.DataFrame):
        """
        Initialize FeatureEngineer.

        Args:
            aligned_data: DataFrame from PointInTimeDatabase.build_aligned_dataset()
        """
        self.data = aligned_data.copy()

        # Ensure timestamp is index for easier manipulation
        if 'timestamp' in self.data.columns:
            self.data = self.data.set_index('timestamp')

    def compute_all_features(self) -> pd.DataFrame:
        """
        Compute all feature categories.

        Returns:
            DataFrame with all features added
        """
        print("\nComputing features...")

        # 1. Volatility Features
        print("  1/10: Volatility features...")
        self._add_volatility_features()

        # 2. K-Means S/R Levels
        print("  2/10: Support/Resistance features...")
        self._add_support_resistance_features()

        # 3. HMM Regime Detection
        print("  3/10: HMM Regime features...")
        self._add_regime_features()

        # 4. Order Flow Imbalance (proxy from taker data)
        print("  4/10: Order flow features...")
        self._add_order_flow_features()

        # 5. Funding Rate Features
        print("  5/10: Funding rate features...")
        self._add_funding_features()

        # 6. Liquidity Score (proxy)
        print("  6/10: Liquidity features...")
        self._add_liquidity_features()

        # 7. Fear & Greed Features
        print("  7/10: Sentiment features...")
        self._add_sentiment_features()

        # 8. Session/Time Features
        print("  8/10: Session/time features...")
        self._add_session_features()

        # 9. Trend Strength Features
        print("  9/10: Trend strength features...")
        self._add_trend_features()

        # 10. Momentum Exhaustion Features
        print("  10/10: Momentum features...")
        self._add_momentum_features()

        print(f"Feature engineering complete. Total columns: {len(self.data.columns)}")

        return self.data.reset_index()

    # =========================================================================
    # 1. VOLATILITY FEATURES
    # =========================================================================
    def _add_volatility_features(self):
        """ATR, Bollinger Width, Realized Volatility"""
        df = self.data

        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # ATR (14 period)
        df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']  # Normalized

        # Bollinger Bands Width
        df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['std_20'] = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']

        # Realized Volatility (hourly returns std)
        df['returns'] = df['close'].pct_change()
        df['realized_vol_24h'] = df['returns'].rolling(24, min_periods=1).std() * np.sqrt(24)
        df['realized_vol_168h'] = df['returns'].rolling(168, min_periods=1).std() * np.sqrt(168)

        # Volatility regime (percentile-based)
        df['vol_percentile'] = df['atr_ratio'].rolling(168, min_periods=24).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )

    # =========================================================================
    # 2. K-MEANS SUPPORT/RESISTANCE
    # =========================================================================
    def _add_support_resistance_features(self, lookback: int = 168, n_clusters: int = 5):
        """Dynamic S/R levels using K-Means clustering"""
        df = self.data

        # Initialize columns
        df['nearest_sr_distance'] = np.nan
        df['nearest_sr_strength'] = np.nan
        df['price_vs_sr'] = np.nan  # Above (+1) or below (-1) nearest level

        try:
            from sklearn.cluster import KMeans

            for i in range(lookback, len(df)):
                # Use only past data for clustering
                window = df.iloc[i-lookback:i]

                # Cluster on highs, lows, closes (pivot points)
                pivots = np.concatenate([
                    window['high'].values,
                    window['low'].values,
                    window['close'].values
                ]).reshape(-1, 1)

                # Remove NaN values
                pivots = pivots[~np.isnan(pivots)].reshape(-1, 1)

                if len(pivots) < n_clusters:
                    continue

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pivots)

                # Get cluster centers (S/R levels)
                levels = kmeans.cluster_centers_.flatten()

                # Current price
                current_price = df.iloc[i]['close']

                # Distance to nearest level
                distances = np.abs(levels - current_price)
                nearest_idx = np.argmin(distances)
                nearest_level = levels[nearest_idx]

                # Strength = how many points in that cluster
                labels = kmeans.labels_
                strength = np.sum(labels == nearest_idx) / len(labels)

                df.iloc[i, df.columns.get_loc('nearest_sr_distance')] = \
                    (current_price - nearest_level) / current_price
                df.iloc[i, df.columns.get_loc('nearest_sr_strength')] = strength
                df.iloc[i, df.columns.get_loc('price_vs_sr')] = \
                    1 if current_price > nearest_level else -1

        except ImportError:
            warnings.warn("sklearn not available, skipping K-Means S/R features")

    # =========================================================================
    # 3. HMM REGIME DETECTION
    # =========================================================================
    def _add_regime_features(self, lookback: int = 168, n_regimes: int = 3):
        """Hidden Markov Model for regime detection"""
        df = self.data

        # Initialize
        df['hmm_regime'] = np.nan
        df['hmm_regime_prob_bull'] = np.nan
        df['hmm_regime_prob_bear'] = np.nan
        df['hmm_regime_prob_sideways'] = np.nan

        try:
            from hmmlearn import hmm

            for i in range(lookback, len(df)):
                # Features for HMM: returns and volatility
                window = df.iloc[i-lookback:i]

                returns = window['returns'].fillna(0).values
                atr_ratio = window['atr_ratio'].fillna(window['atr_ratio'].mean()).values

                features = np.column_stack([returns, atr_ratio])

                # Skip if not enough valid data
                if np.isnan(features).any():
                    continue

                try:
                    model = hmm.GaussianHMM(
                        n_components=n_regimes,
                        covariance_type="full",
                        n_iter=100,
                        random_state=42
                    )
                    model.fit(features)

                    # Get regime probabilities for current state
                    probs = model.predict_proba(features)[-1]
                    regime = model.predict(features)[-1]

                    df.iloc[i, df.columns.get_loc('hmm_regime')] = regime

                    # Sort regimes by mean return to identify bull/bear/sideways
                    regime_returns = []
                    predictions = model.predict(features)
                    for r in range(n_regimes):
                        mask = predictions == r
                        if mask.any():
                            regime_returns.append(returns[mask].mean())
                        else:
                            regime_returns.append(0)

                    sorted_regimes = np.argsort(regime_returns)
                    bear_idx, sideways_idx, bull_idx = sorted_regimes

                    df.iloc[i, df.columns.get_loc('hmm_regime_prob_bull')] = probs[bull_idx]
                    df.iloc[i, df.columns.get_loc('hmm_regime_prob_bear')] = probs[bear_idx]
                    df.iloc[i, df.columns.get_loc('hmm_regime_prob_sideways')] = probs[sideways_idx]

                except Exception:
                    # HMM failed to converge, use NaN
                    pass

        except ImportError:
            warnings.warn("hmmlearn not available, skipping HMM regime features")

    # =========================================================================
    # 4. ORDER FLOW IMBALANCE (Proxy)
    # =========================================================================
    def _add_order_flow_features(self):
        """Order flow features from taker buy/sell data"""
        df = self.data

        # If we have taker volume data from hybrid fetcher
        if 'taker_volume_buySellRatio' in df.columns:
            df['ofi_proxy'] = df['taker_volume_buySellRatio'] - 1  # >0 = buy pressure
            df['ofi_proxy_ma'] = df['ofi_proxy'].rolling(6, min_periods=1).mean()
            df['ofi_proxy_std'] = df['ofi_proxy'].rolling(24, min_periods=1).std()

        # Alternative: use taker buy from OHLCV
        if 'taker_buy_base' in df.columns and 'volume' in df.columns:
            df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, np.nan)
            df['buy_pressure'] = df['taker_buy_ratio'] - 0.5  # Center around 0
            df['buy_pressure_ma'] = df['buy_pressure'].rolling(6, min_periods=1).mean()

    # =========================================================================
    # 5. FUNDING RATE FEATURES
    # =========================================================================
    def _add_funding_features(self):
        """Funding rate sentiment features"""
        df = self.data

        # Look for funding rate column (may have different prefixes)
        fr_col = None
        for col in df.columns:
            if 'fundingRate' in col or 'funding_rate' in col.lower():
                fr_col = col
                break

        if fr_col is not None:
            fr = df[fr_col]

            # Raw and normalized
            df['funding_rate'] = fr
            df['funding_rate_abs'] = fr.abs()

            # Rolling averages
            df['funding_rate_ma_24h'] = fr.rolling(24, min_periods=1).mean()

            # Extreme funding detection
            df['funding_extreme_long'] = (fr > 0.001).astype(int)  # >0.1%
            df['funding_extreme_short'] = (fr < -0.001).astype(int)

            # Z-score of funding
            rolling_mean = fr.rolling(168, min_periods=24).mean()
            rolling_std = fr.rolling(168, min_periods=24).std()
            df['funding_zscore'] = (fr - rolling_mean) / rolling_std.replace(0, np.nan)

    # =========================================================================
    # 6. LIQUIDITY SCORE (Proxy)
    # =========================================================================
    def _add_liquidity_features(self):
        """Liquidity proxy from available data"""
        df = self.data

        # Volume-based liquidity proxy
        df['volume_ma_24h'] = df['volume'].rolling(24, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_24h'].replace(0, np.nan)

        # Spread proxy: (high - low) / close normalized by ATR
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_vs_atr'] = df['spread_proxy'] / df['atr_ratio'].replace(0, np.nan)

        # Composite liquidity score
        df['liquidity_score'] = df['volume_ratio'] / (df['spread_vs_atr'].replace(0, np.nan) + 1)

        # Low liquidity flag
        rolling_quantile = df['liquidity_score'].rolling(168, min_periods=24).quantile(0.25)
        df['low_liquidity'] = (df['liquidity_score'] < rolling_quantile).astype(int)

    # =========================================================================
    # 7. FEAR & GREED FEATURES
    # =========================================================================
    def _add_sentiment_features(self):
        """Fear & Greed index features"""
        df = self.data

        # Look for fear greed column
        fg_col = None
        for col in df.columns:
            if 'fear_greed' in col.lower() and 'value' in col.lower():
                fg_col = col
                break

        if fg_col is not None:
            fg = pd.to_numeric(df[fg_col], errors='coerce')

            df['fear_greed'] = fg
            df['fear_greed_ma_7d'] = fg.rolling(168, min_periods=24).mean()  # 7 days

            # Sentiment zones
            df['extreme_fear'] = (fg < 25).astype(int)
            df['extreme_greed'] = (fg > 75).astype(int)

            # Rate of change
            df['fear_greed_roc'] = fg.diff(24)  # 24h change

    # =========================================================================
    # 8. SESSION/TIME FEATURES
    # =========================================================================
    def _add_session_features(self):
        """Time-based features"""
        df = self.data

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        # Hour of day (cyclical encoding)
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical encoding)
        dow = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # Trading sessions (UTC)
        df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_london'] = ((hour >= 8) & (hour < 14)).astype(int)
        df['session_newyork'] = ((hour >= 14) & (hour < 21)).astype(int)
        df['session_overlap'] = ((hour >= 14) & (hour < 17)).astype(int)  # London/NY

        # Weekend
        df['is_weekend'] = (dow >= 5).astype(int)

    # =========================================================================
    # 9. TREND STRENGTH FEATURES
    # =========================================================================
    def _add_trend_features(self):
        """ADX and trend strength indicators"""
        df = self.data

        # Directional Movement
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()

        df['plus_dm'] = np.where(
            (high_diff > low_diff.abs()) & (high_diff > 0),
            high_diff, 0
        )
        df['minus_dm'] = np.where(
            (low_diff.abs() > high_diff) & (low_diff < 0),
            low_diff.abs(), 0
        )

        # Smoothed DI
        atr = df['atr_14']
        df['plus_di'] = 100 * (pd.Series(df['plus_dm']).rolling(14, min_periods=1).sum() / (atr * 14).replace(0, np.nan))
        df['minus_di'] = 100 * (pd.Series(df['minus_dm']).rolling(14, min_periods=1).sum() / (atr * 14).replace(0, np.nan))

        # ADX
        di_diff = (df['plus_di'] - df['minus_di']).abs()
        di_sum = df['plus_di'] + df['minus_di']
        df['dx'] = 100 * di_diff / di_sum.replace(0, np.nan)
        df['adx'] = df['dx'].rolling(14, min_periods=1).mean()

        # MA slopes
        df['sma_slope_20'] = df['sma_20'].diff(5) / df['sma_20'].shift(5).replace(0, np.nan)
        sma_50 = df['close'].rolling(50, min_periods=1).mean()
        df['sma_slope_50'] = sma_50.diff(10) / sma_50.shift(10).replace(0, np.nan)

        # Higher highs / lower lows count
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['hh_count_10'] = df['higher_high'].rolling(10, min_periods=1).sum()
        df['ll_count_10'] = df['lower_low'].rolling(10, min_periods=1).sum()

        # Trend score
        df['trend_score'] = (df['hh_count_10'] - df['ll_count_10']) / 10

    # =========================================================================
    # 10. MOMENTUM EXHAUSTION FEATURES
    # =========================================================================
    def _add_momentum_features(self):
        """RSI, MACD, and exhaustion signals"""
        df = self.data

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Price-Volume Divergence (exhaustion signal)
        price_slope = df['close'].diff(5).rolling(5, min_periods=1).mean()
        volume_slope = df['volume'].diff(5).rolling(5, min_periods=1).mean()

        df['bullish_exhaustion'] = ((price_slope > 0) & (volume_slope < 0)).astype(int)
        df['bearish_exhaustion'] = ((price_slope < 0) & (volume_slope < 0)).astype(int)

        # Momentum (rate of change)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_24'] = df['close'].pct_change(24)
