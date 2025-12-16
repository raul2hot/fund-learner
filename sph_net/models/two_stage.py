"""
Two-Stage Model for Trading Classification

Architecture designed to handle the distribution shift and class imbalance problems:

Stage 1: TRADEABLE DETECTOR
    - Binary classification: TRADE vs NO_TRADE
    - Target: (label in {0, 4}) -> TRADE, else NO_TRADE
    - Handles the ~12% vs ~88% imbalance as a simpler binary problem

Stage 2: DIRECTION CLASSIFIER (only when Stage 1 predicts TRADE)
    - Binary classification: LONG vs SHORT
    - Target: (label == 0) -> LONG, (label == 4) -> SHORT
    - Balanced problem (~50% each in tradeable samples)

Benefits:
1. Each model does ONE thing well
2. Stage 1 can learn "is this a good setup?" without direction confusion
3. Stage 2 is perfectly balanced
4. Can tune thresholds independently for each stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion


class TwoStageModel(nn.Module):
    """
    Two-Stage Trading Model.

    Stage 1: Is this candle tradeable?
    Stage 2: If tradeable, what direction?
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Support both old and new config attribute names
        n_price_features = getattr(config, 'n_price_features', getattr(config, 'price_features', 5))
        n_engineered_features = getattr(config, 'n_engineered_features', getattr(config, 'engineered_features', 10))

        # Shared encoders (feature extraction is shared)
        self.temporal_encoder = TemporalEncoder(
            input_dim=n_price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=n_engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Post-fusion transformer layer
        d_feedforward = getattr(config, 'd_feedforward', config.d_model * 4)
        self.decoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=d_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        # Stage 1: Tradeable Detector (binary)
        self.tradeable_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2)  # TRADE vs NO_TRADE
        )

        # Stage 2: Direction Classifier (binary)
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2)  # LONG vs SHORT
        )

        # Auxiliary regression head for return prediction
        self.aux_regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )

    def _get_features(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Extract shared features from inputs."""
        temporal_tokens = self.temporal_encoder(prices)
        feature_tokens = self.feature_encoder(features)
        fused = self.co_attention(temporal_tokens, feature_tokens)
        decoded = self.decoder(fused)
        pooled = decoded[:, -1, :]  # Use last token
        return pooled

    def forward(
        self,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> dict:
        """
        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]

        Returns:
            dict with:
            - tradeable_logits: [batch, 2] - Stage 1 output
            - direction_logits: [batch, 2] - Stage 2 output
            - return_pred: [batch] - Auxiliary regression
            - logits: [batch, 5] - Combined 5-class logits for compatibility
        """
        # Get shared features
        pooled = self._get_features(prices, features)

        # Stage 1: Is it tradeable?
        tradeable_logits = self.tradeable_head(pooled)

        # Stage 2: What direction?
        direction_logits = self.direction_head(pooled)

        # Auxiliary return prediction
        return_pred = self.aux_regressor(pooled).squeeze(-1)

        # Convert to 5-class logits for compatibility with existing evaluation
        combined_logits = self._combine_to_5class(tradeable_logits, direction_logits)

        return {
            'tradeable_logits': tradeable_logits,
            'direction_logits': direction_logits,
            'return_pred': return_pred,
            'logits': combined_logits
        }

    def _combine_to_5class(
        self,
        tradeable_logits: torch.Tensor,
        direction_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine two-stage outputs into 5-class logits.

        Classes: 0=HIGH_BULL, 1=BULL, 2=RANGE_BOUND, 3=BEAR, 4=LOW_BEAR

        Logic:
        - If NOT tradeable: high prob for class 2 (RANGE_BOUND), lower for 1 and 3
        - If tradeable + LONG: high prob for class 0 (HIGH_BULL)
        - If tradeable + SHORT: high prob for class 4 (LOW_BEAR)
        """
        batch_size = tradeable_logits.shape[0]
        device = tradeable_logits.device

        # Get probabilities
        tradeable_probs = F.softmax(tradeable_logits, dim=-1)  # [batch, 2]
        direction_probs = F.softmax(direction_logits, dim=-1)  # [batch, 2]

        p_tradeable = tradeable_probs[:, 1]  # P(trade)
        p_no_trade = tradeable_probs[:, 0]   # P(no trade)
        p_long = direction_probs[:, 0]       # P(long | trade)
        p_short = direction_probs[:, 1]      # P(short | trade)

        # Build 5-class logits
        # We use log probabilities to stay in logit space
        logits = torch.zeros(batch_size, 5, device=device)

        # Class 0 (HIGH_BULL): P(trade) * P(long)
        logits[:, 0] = torch.log(p_tradeable * p_long + 1e-8)

        # Class 1 (BULL): P(no_trade) * 0.2 (mild bullish tendency)
        logits[:, 1] = torch.log(p_no_trade * 0.2 + 1e-8)

        # Class 2 (RANGE_BOUND): P(no_trade) * 0.6 (main no-trade class)
        logits[:, 2] = torch.log(p_no_trade * 0.6 + 1e-8)

        # Class 3 (BEAR): P(no_trade) * 0.2 (mild bearish tendency)
        logits[:, 3] = torch.log(p_no_trade * 0.2 + 1e-8)

        # Class 4 (LOW_BEAR): P(trade) * P(short)
        logits[:, 4] = torch.log(p_tradeable * p_short + 1e-8)

        return logits

    def predict(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0-4)."""
        outputs = self.forward(prices, features)
        return torch.argmax(outputs['logits'], dim=-1)

    def predict_two_stage(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        tradeable_threshold: float = 0.5,
        long_threshold: float = 0.5
    ) -> dict:
        """
        Get predictions using two-stage logic with custom thresholds.

        Returns:
            dict with:
            - is_tradeable: [batch] bool
            - direction: [batch] 0=LONG, 1=SHORT (only valid where is_tradeable)
            - tradeable_prob: [batch] probability of being tradeable
            - direction_probs: [batch, 2] LONG/SHORT probabilities
        """
        outputs = self.forward(prices, features)

        tradeable_probs = F.softmax(outputs['tradeable_logits'], dim=-1)
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)

        is_tradeable = tradeable_probs[:, 1] >= tradeable_threshold
        direction = (direction_probs[:, 1] >= long_threshold).long()  # 0=LONG, 1=SHORT

        return {
            'is_tradeable': is_tradeable,
            'direction': direction,
            'tradeable_prob': tradeable_probs[:, 1],
            'direction_probs': direction_probs
        }


class CalibratedTwoStageModel(nn.Module):
    """
    Wrapper for TwoStageModel with calibrated thresholds, regime filtering, and risk management.

    This class wraps a trained TwoStageModel and adds:
    1. Calibrated trade threshold (reduces over-trading)
    2. Volatility regime filtering (avoid high volatility)
    3. Optional confidence-based position sizing
    4. Stop-loss and take-profit risk management
    5. ADAPTIVE THRESHOLD based on regime features (NEW)
    6. CONFIDENCE CAPPING in choppy markets (NEW)

    CRITICAL FINDING (May 2021 analysis):
    The model can be "confidently wrong" - in choppy markets, HIGH confidence
    trades LOSE money while LOW confidence trades are profitable. This is an
    INVERSE confidence pattern that requires:
    - Capping maximum confidence in choppy markets
    - Increasing minimum threshold when trend_efficiency is low

    Optimal settings based on backtesting (2025 data, 343 days):
    - trade_threshold=0.55: Best risk-adjusted returns
    - ~320 trades, ~3.9% trade frequency
    - +32.69% total return without stop-loss
    - +52.45% total return with -1.78% stop-loss (moderate)
    - Avoid high volatility regime: loses money (-0.015% avg)
    - Position sizing disabled: confidence doesn't correlate with returns

    Stop-Loss Analysis Results:
    - Conservative (-1.32%): +65.05% return, stops 5% of trades
    - Moderate (-1.78%): +52.45% return, stops 2.5% of trades (RECOMMENDED)
    - Aggressive (-2.27%): +44.78% return, stops 1% of trades

    Note on Sharpe Ratio:
    - Use trade-frequency annualization, not candle-frequency
    - With 320 trades over 343 days: ~340 trades/year
    - Sharpe = (mean/std) * sqrt(340), NOT sqrt(35000)

    Usage:
        calibrated = CalibratedTwoStageModel(
            model,
            trade_threshold=0.55,
            stop_loss_pct=-0.0178,  # -1.78% stop-loss
            use_adaptive_threshold=True,  # Enable regime-aware filtering
            trend_efficiency_col_idx=2,   # Index of trend_efficiency in features
            vol_ratio_col_idx=1,          # Index of vol_ratio in features
        )
        results = calibrated.predict_with_sizing(prices, features)
    """

    # Volatility regime thresholds (percentiles from training data)
    VOL_LOW_THRESHOLD = 0.33   # Below this = low volatility
    VOL_HIGH_THRESHOLD = 0.66  # Above this = high volatility (avoid!)

    # Default risk management settings (from analysis)
    DEFAULT_STOP_LOSS = -0.02     # -2.0% stop-loss
    DEFAULT_TAKE_PROFIT = None    # Let winners run (positive skew)

    # Regime detection thresholds for adaptive threshold (NEW)
    TREND_EFFICIENCY_LOW = 0.3      # Below this = very choppy market
    TREND_EFFICIENCY_MED = 0.5      # Below this = somewhat choppy
    VOL_RATIO_HIGH = 1.3            # Above this = elevated short-term volatility

    # Threshold adjustments (NEW)
    BASE_THRESHOLD = 0.55
    CHOPPY_ADJUSTMENT = 0.05        # Add this per choppy condition
    MAX_THRESHOLD = 0.70            # Never go above this

    # Confidence capping for inverse regime (NEW)
    MAX_CONFIDENCE_CHOPPY = 0.65    # Cap confidence in very choppy markets
    # CRITICAL: Only apply confidence cap when batch trade frequency is HIGH
    # High trade frequency indicates model is overconfident/inverted
    CONFIDENCE_CAP_FREQ_THRESHOLD = 0.10  # Only cap when >10% of batch would trade

    def __init__(
        self,
        model: 'TwoStageModel',
        trade_threshold: float = 0.55,  # Base threshold (may be adjusted dynamically)
        direction_threshold: float = 0.5,
        max_position: float = 1.0,
        use_position_sizing: bool = False,  # Disabled - doesn't improve returns
        filter_high_volatility: bool = True,  # Avoid high vol regime
        vol_threshold: float = None,  # Custom high vol threshold (auto if None)
        stop_loss_pct: float = None,  # Stop-loss as decimal (e.g., -0.0178 for -1.78%)
        take_profit_pct: float = None,  # Take-profit as decimal (None = let winners run)
        # NEW: Adaptive threshold settings
        use_adaptive_threshold: bool = False,
        trend_efficiency_col_idx: int = None,  # Index of trend_efficiency in features
        vol_ratio_col_idx: int = None,          # Index of vol_ratio in features
        use_confidence_capping: bool = False,   # Cap confidence in choppy markets
        confidence_cap_freq_threshold: float = 0.10,  # Only cap when batch freq > this
    ):
        super().__init__()
        self.model = model
        self.base_threshold = trade_threshold
        self.trade_threshold = trade_threshold
        self.direction_threshold = direction_threshold
        self.max_position = max_position
        self.use_position_sizing = use_position_sizing
        self.filter_high_volatility = filter_high_volatility
        self.vol_threshold = vol_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Adaptive threshold config (NEW)
        self.use_adaptive_threshold = use_adaptive_threshold
        self.trend_efficiency_col_idx = trend_efficiency_col_idx
        self.vol_ratio_col_idx = vol_ratio_col_idx
        self.use_confidence_capping = use_confidence_capping
        self.confidence_cap_freq_threshold = confidence_cap_freq_threshold

    def forward(self, prices: torch.Tensor, features: torch.Tensor) -> dict:
        """Pass through to underlying model."""
        return self.model(prices, features)

    def compute_adaptive_threshold(
        self,
        features: torch.Tensor,
        trend_efficiency: torch.Tensor = None,
        vol_ratio: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute per-sample adaptive threshold based on regime features.

        In choppy markets (low trend_efficiency, high vol_ratio), the model
        tends to be confidently wrong. We increase the threshold to filter
        out trades where confidence is unreliable.

        Args:
            features: [batch, window_size, n_features] - full feature tensor
            trend_efficiency: [batch] - optional pre-computed values
            vol_ratio: [batch] - optional pre-computed values

        Returns:
            threshold: [batch] - adaptive threshold per sample
        """
        batch_size = features.shape[0]
        device = features.device

        # Start with base threshold
        threshold = torch.full((batch_size,), self.base_threshold, device=device)

        if not self.use_adaptive_threshold:
            return threshold

        # Extract regime features from last timestep
        last_features = features[:, -1, :]  # [batch, n_features]

        # Get trend efficiency
        if trend_efficiency is None and self.trend_efficiency_col_idx is not None:
            trend_efficiency = last_features[:, self.trend_efficiency_col_idx]

        # Get vol ratio
        if vol_ratio is None and self.vol_ratio_col_idx is not None:
            vol_ratio = last_features[:, self.vol_ratio_col_idx]

        # Apply adjustments based on regime
        if trend_efficiency is not None:
            # Low trend efficiency = choppy market = higher threshold
            # Very choppy (< 0.3): add 0.10 to threshold
            very_choppy_mask = trend_efficiency < self.TREND_EFFICIENCY_LOW
            # Somewhat choppy (0.3-0.5): add 0.05 to threshold
            somewhat_choppy_mask = (
                (trend_efficiency >= self.TREND_EFFICIENCY_LOW) &
                (trend_efficiency < self.TREND_EFFICIENCY_MED)
            )

            threshold = torch.where(very_choppy_mask, threshold + 0.10, threshold)
            threshold = torch.where(somewhat_choppy_mask, threshold + self.CHOPPY_ADJUSTMENT, threshold)

        if vol_ratio is not None:
            # High vol ratio = elevated uncertainty = higher threshold
            high_vol_mask = vol_ratio > self.VOL_RATIO_HIGH
            threshold = torch.where(high_vol_mask, threshold + self.CHOPPY_ADJUSTMENT, threshold)

        # Cap at maximum
        threshold = torch.clamp(threshold, max=self.MAX_THRESHOLD)

        return threshold

    def compute_confidence_cap(
        self,
        features: torch.Tensor,
        trend_efficiency: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute per-sample maximum confidence cap for inverse regime detection.

        In choppy markets, HIGH confidence trades often LOSE money (inverse
        correlation). This method computes a cap on confidence to filter out
        trades where the model is confidently wrong.

        Args:
            features: [batch, window_size, n_features] - full feature tensor
            trend_efficiency: [batch] - optional pre-computed values

        Returns:
            max_confidence: [batch] - maximum confidence to accept per sample
        """
        batch_size = features.shape[0]
        device = features.device

        # Start with no cap (1.0)
        max_confidence = torch.ones(batch_size, device=device)

        if not self.use_confidence_capping:
            return max_confidence

        # Extract regime features from last timestep
        last_features = features[:, -1, :]

        # Get trend efficiency
        if trend_efficiency is None and self.trend_efficiency_col_idx is not None:
            trend_efficiency = last_features[:, self.trend_efficiency_col_idx]

        if trend_efficiency is not None:
            # In very choppy markets, cap confidence to avoid inverse correlation
            very_choppy_mask = trend_efficiency < self.TREND_EFFICIENCY_LOW
            max_confidence = torch.where(
                very_choppy_mask,
                torch.full_like(max_confidence, self.MAX_CONFIDENCE_CHOPPY),
                max_confidence
            )

        return max_confidence

    @torch.no_grad()
    def predict_with_sizing(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None,
        trend_efficiency: torch.Tensor = None,
        vol_ratio: torch.Tensor = None,
    ) -> dict:
        """
        Get predictions with ADAPTIVE thresholds and regime filtering.

        MODIFIED: Now uses per-sample adaptive threshold based on regime features.
        Also supports confidence capping to handle inverse confidence patterns.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]
            volatility: [batch] optional volatility values for regime filtering
            trend_efficiency: [batch] optional pre-computed trend efficiency values
            vol_ratio: [batch] optional pre-computed vol ratio values

        Returns:
            dict with:
            - should_trade: [batch] bool - whether to trade
            - is_long: [batch] bool - direction (True=long, False=short)
            - position_size: [batch] 0.0-1.0 position size
            - trade_prob: [batch] probability of trade
            - direction_confidence: [batch] confidence in direction
            - regime_filtered: [batch] bool - True if filtered by regime
            - adaptive_threshold: [batch] - threshold used per sample (NEW)
            - confidence_capped: [batch] bool - True if confidence was capped (NEW)
        """
        outputs = self.model(prices, features)

        # Get probabilities
        tradeable_probs = F.softmax(outputs['tradeable_logits'], dim=-1)
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)

        trade_prob = tradeable_probs[:, 1]  # P(trade)
        long_prob = direction_probs[:, 0]   # P(long)
        short_prob = direction_probs[:, 1]  # P(short)

        # ADAPTIVE THRESHOLD (NEW)
        # Compute per-sample threshold based on regime features
        adaptive_threshold = self.compute_adaptive_threshold(
            features, trend_efficiency, vol_ratio
        )

        # Apply adaptive threshold (now per-sample)
        should_trade = trade_prob >= adaptive_threshold

        # FREQUENCY-CONDITIONAL CONFIDENCE CAPPING (NEW)
        # Only apply confidence cap when batch trade frequency is HIGH
        # High frequency indicates model is overconfident (inverted confidence pattern)
        confidence_capped = torch.zeros_like(should_trade)
        if self.use_confidence_capping:
            # Compute batch trade frequency (what % of batch would trade)
            batch_trade_freq = should_trade.float().mean().item()

            # Only apply cap if frequency exceeds threshold (default 10%)
            if batch_trade_freq > self.confidence_cap_freq_threshold:
                max_confidence = self.compute_confidence_cap(features, trend_efficiency)
                # Filter trades where confidence exceeds cap
                exceeds_cap = trade_prob > max_confidence
                confidence_capped = exceeds_cap & should_trade
                should_trade = should_trade & ~exceeds_cap

        # Volatility regime filtering (unchanged)
        regime_filtered = torch.zeros_like(should_trade)
        if self.filter_high_volatility and volatility is not None:
            # Estimate high volatility threshold if not set
            if self.vol_threshold is None:
                # Use 66th percentile as high vol threshold
                vol_thresh = torch.quantile(volatility, self.VOL_HIGH_THRESHOLD)
            else:
                vol_thresh = self.vol_threshold

            is_high_vol = volatility > vol_thresh
            regime_filtered = is_high_vol & should_trade
            should_trade = should_trade & ~is_high_vol

        # Determine direction
        is_long = long_prob > short_prob
        direction_confidence = torch.abs(long_prob - 0.5) * 2  # 0-1 scale

        # Calculate position size if enabled
        # Use base threshold for position sizing calculation
        if self.use_position_sizing:
            # Scale trade_prob from [base_threshold, 1] to [0, 1]
            scaled_prob = (trade_prob - self.base_threshold) / (1.0 - self.base_threshold)
            scaled_prob = scaled_prob.clamp(0, 1)

            # Combine with direction confidence
            position_size = scaled_prob * (0.7 + 0.3 * direction_confidence)
            position_size = position_size.clamp(0, self.max_position)

            # Zero position for non-trades
            position_size = torch.where(should_trade, position_size, torch.zeros_like(position_size))
        else:
            # Equal sizing: full position for all trades
            position_size = torch.where(
                should_trade,
                torch.full_like(trade_prob, self.max_position),
                torch.zeros_like(trade_prob)
            )

        return {
            'should_trade': should_trade,
            'is_long': is_long,
            'position_size': position_size,
            'trade_prob': trade_prob,
            'direction_confidence': direction_confidence,
            'long_prob': long_prob,
            'short_prob': short_prob,
            'regime_filtered': regime_filtered,
            'adaptive_threshold': adaptive_threshold,  # NEW: threshold used per sample
            'confidence_capped': confidence_capped,    # NEW: True if filtered by cap
        }

    def get_trade_signal(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None
    ) -> dict:
        """
        Get a single trade signal (convenience method for single samples).

        Returns:
            dict with human-readable trade signal
        """
        results = self.predict_with_sizing(prices, features, volatility)

        signals = []
        for i in range(len(results['should_trade'])):
            if results['should_trade'][i]:
                direction = 'LONG' if results['is_long'][i] else 'SHORT'
                signals.append({
                    'action': direction,
                    'position_size': results['position_size'][i].item(),
                    'confidence': results['trade_prob'][i].item(),
                    'direction_confidence': results['direction_confidence'][i].item(),
                })
            elif results['regime_filtered'][i]:
                signals.append({
                    'action': 'HOLD_HIGH_VOL',
                    'position_size': 0.0,
                    'confidence': results['trade_prob'][i].item(),
                    'direction_confidence': 0.0,
                    'reason': 'Filtered due to high volatility regime',
                })
            else:
                signals.append({
                    'action': 'HOLD',
                    'position_size': 0.0,
                    'confidence': results['trade_prob'][i].item(),
                    'direction_confidence': 0.0,
                })

        return signals if len(signals) > 1 else signals[0]

    def apply_stop_loss(
        self,
        trade_returns: torch.Tensor,
        mae_values: torch.Tensor = None,
    ) -> dict:
        """
        Apply stop-loss and take-profit to trade returns.

        In backtesting, we use the Maximum Adverse Excursion (MAE) to determine
        if a stop-loss would have been triggered. MAE represents the worst
        drawdown during the trade before it closed.

        Args:
            trade_returns: [batch] actual trade returns (signed based on direction)
            mae_values: [batch] maximum adverse excursion during the trade
                       (how far price moved against the position before close)

        Returns:
            dict with:
            - adjusted_returns: [batch] returns after stop-loss application
            - stopped_out: [batch] bool - True if stop-loss was triggered
            - took_profit: [batch] bool - True if take-profit was triggered
        """
        if self.stop_loss_pct is None and self.take_profit_pct is None:
            # No risk management, return original returns
            return {
                'adjusted_returns': trade_returns,
                'stopped_out': torch.zeros_like(trade_returns, dtype=torch.bool),
                'took_profit': torch.zeros_like(trade_returns, dtype=torch.bool),
            }

        adjusted_returns = trade_returns.clone()
        stopped_out = torch.zeros_like(trade_returns, dtype=torch.bool)
        took_profit = torch.zeros_like(trade_returns, dtype=torch.bool)

        # Apply stop-loss
        if self.stop_loss_pct is not None:
            # If MAE is provided, use it to determine if stop would have triggered
            if mae_values is not None:
                # Stop triggered if MAE exceeded stop-loss threshold
                stop_triggered = mae_values > abs(self.stop_loss_pct)
                stopped_out = stop_triggered

                # For stopped trades, cap the loss at stop-loss level
                # (in reality, slippage might make it slightly worse)
                adjusted_returns = torch.where(
                    stop_triggered,
                    torch.full_like(trade_returns, self.stop_loss_pct),
                    trade_returns
                )
            else:
                # Without MAE, use simple return-based stop
                # (less accurate - assumes stop triggers at worst point)
                stop_triggered = trade_returns < self.stop_loss_pct
                stopped_out = stop_triggered
                adjusted_returns = torch.where(
                    stop_triggered,
                    torch.full_like(trade_returns, self.stop_loss_pct),
                    trade_returns
                )

        # Apply take-profit (after stop-loss)
        if self.take_profit_pct is not None:
            # Take profit if return exceeded target
            tp_triggered = adjusted_returns > self.take_profit_pct
            took_profit = tp_triggered
            adjusted_returns = torch.where(
                tp_triggered,
                torch.full_like(adjusted_returns, self.take_profit_pct),
                adjusted_returns
            )

        return {
            'adjusted_returns': adjusted_returns,
            'stopped_out': stopped_out,
            'took_profit': took_profit,
        }

    def get_risk_parameters(self) -> dict:
        """
        Get current risk management parameters.

        Returns human-readable risk configuration.
        """
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'stop_loss_display': f"{self.stop_loss_pct * 100:.2f}%" if self.stop_loss_pct else "None",
            'take_profit_pct': self.take_profit_pct,
            'take_profit_display': f"{self.take_profit_pct * 100:.2f}%" if self.take_profit_pct else "None (let winners run)",
            'max_position': self.max_position,
            'trade_threshold': self.trade_threshold,
        }


def apply_stop_loss_to_returns(
    trade_returns: torch.Tensor,
    stop_loss_pct: float,
    mae_values: torch.Tensor = None,
    take_profit_pct: float = None,
) -> dict:
    """
    Standalone function to apply stop-loss to trade returns.

    Useful for backtesting different stop-loss levels without
    instantiating a new model.

    Args:
        trade_returns: Trade returns (can be numpy array or tensor)
        stop_loss_pct: Stop-loss as decimal (e.g., -0.0178 for -1.78%)
        mae_values: Optional MAE values for more accurate simulation
        take_profit_pct: Optional take-profit level

    Returns:
        dict with adjusted returns and statistics
    """
    import numpy as np

    # Convert to numpy if tensor
    if isinstance(trade_returns, torch.Tensor):
        returns = trade_returns.numpy()
    else:
        returns = np.array(trade_returns)

    if mae_values is not None:
        if isinstance(mae_values, torch.Tensor):
            mae = mae_values.numpy()
        else:
            mae = np.array(mae_values)
    else:
        mae = None

    adjusted = returns.copy()
    stopped_out = np.zeros(len(returns), dtype=bool)
    took_profit = np.zeros(len(returns), dtype=bool)

    # Apply stop-loss
    if stop_loss_pct is not None:
        if mae is not None:
            # Use MAE for accurate stop detection
            stop_triggered = mae > abs(stop_loss_pct)
        else:
            # Fallback: use return-based detection
            stop_triggered = returns < stop_loss_pct

        stopped_out = stop_triggered
        adjusted[stop_triggered] = stop_loss_pct

    # Apply take-profit
    if take_profit_pct is not None:
        tp_triggered = adjusted > take_profit_pct
        took_profit = tp_triggered
        adjusted[tp_triggered] = take_profit_pct

    # Calculate statistics
    original_total = returns.sum()
    adjusted_total = adjusted.sum()
    n_stopped = stopped_out.sum()
    n_took_profit = took_profit.sum()

    return {
        'adjusted_returns': adjusted,
        'stopped_out': stopped_out,
        'took_profit': took_profit,
        'original_total_return': original_total * 100,
        'adjusted_total_return': adjusted_total * 100,
        'improvement': (adjusted_total - original_total) * 100,
        'n_stopped': int(n_stopped),
        'n_took_profit': int(n_took_profit),
        'pct_stopped': n_stopped / len(returns) * 100 if len(returns) > 0 else 0,
    }


class TwoStageLoss(nn.Module):
    """
    Loss function for Two-Stage Model.

    Components:
    1. Stage 1 loss: Binary cross-entropy for tradeable detection
    2. Stage 2 loss: Binary cross-entropy for direction (only on tradeable samples)
    3. Auxiliary MSE loss for return prediction
    """

    def __init__(
        self,
        tradeable_weight: float = 1.0,
        direction_weight: float = 1.0,
        aux_weight: float = 0.1,
        tradeable_pos_weight: float = 7.0,  # ~88%/12% imbalance
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.tradeable_weight = tradeable_weight
        self.direction_weight = direction_weight
        self.aux_weight = aux_weight
        self.focal_gamma = focal_gamma

        # Positive class weight for tradeable (handles imbalance)
        self.tradeable_pos_weight = torch.tensor([tradeable_pos_weight])

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        next_return: torch.Tensor
    ) -> dict:
        """
        Args:
            outputs: dict with 'tradeable_logits', 'direction_logits', 'return_pred'
            targets: [batch] original 5-class labels (0-4)
            next_return: [batch] actual returns

        Returns:
            dict with individual losses and total
        """
        device = targets.device

        # Convert 5-class targets to binary targets
        # Tradeable: labels 0 and 4
        is_tradeable = ((targets == 0) | (targets == 4)).float()

        # Direction: 0=LONG (from label 0), 1=SHORT (from label 4)
        # Only valid where is_tradeable
        direction_target = (targets == 4).float()

        # Stage 1: Tradeable detection loss with focal component
        tradeable_logits = outputs['tradeable_logits']
        pos_weight = self.tradeable_pos_weight.to(device)

        # Focal loss for tradeable detection
        tradeable_probs = F.softmax(tradeable_logits, dim=-1)[:, 1]
        tradeable_ce = F.binary_cross_entropy_with_logits(
            tradeable_logits[:, 1] - tradeable_logits[:, 0],
            is_tradeable,
            pos_weight=pos_weight,
            reduction='none'
        )
        pt = torch.where(is_tradeable == 1, tradeable_probs, 1 - tradeable_probs)
        tradeable_loss = ((1 - pt) ** self.focal_gamma * tradeable_ce).mean()

        # Stage 2: Direction loss (only on actually tradeable samples)
        tradeable_mask = is_tradeable.bool()
        if tradeable_mask.sum() > 0:
            direction_logits = outputs['direction_logits'][tradeable_mask]
            direction_targets = direction_target[tradeable_mask]

            # Binary cross-entropy for direction (balanced, so no pos_weight)
            direction_loss = F.binary_cross_entropy_with_logits(
                direction_logits[:, 1] - direction_logits[:, 0],
                direction_targets,
                reduction='mean'
            )
        else:
            direction_loss = torch.tensor(0.0, device=device)

        # Auxiliary regression loss
        aux_loss = self.mse_loss(outputs['return_pred'], next_return)

        # Total loss
        total_loss = (
            self.tradeable_weight * tradeable_loss +
            self.direction_weight * direction_loss +
            self.aux_weight * aux_loss
        )

        return {
            'total': total_loss,
            'tradeable': tradeable_loss,
            'direction': direction_loss,
            'auxiliary': aux_loss
        }
