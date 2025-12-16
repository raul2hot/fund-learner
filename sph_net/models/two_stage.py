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
    Wrapper for TwoStageModel with calibrated thresholds and regime filtering.

    This class wraps a trained TwoStageModel and adds:
    1. Calibrated trade threshold (reduces over-trading)
    2. Volatility regime filtering (avoid high volatility)
    3. Optional confidence-based position sizing

    Optimal settings based on backtesting (2025 data, 343 days):
    - trade_threshold=0.55: Best risk-adjusted returns
    - ~320 trades, ~3.9% trade frequency
    - +32.69% total return, +0.10% per trade
    - Avoid high volatility regime: loses money (-0.015% avg)
    - Position sizing disabled: confidence doesn't correlate with returns

    Note on Sharpe Ratio:
    - Use trade-frequency annualization, not candle-frequency
    - With 320 trades over 343 days: ~340 trades/year
    - Sharpe = (mean/std) * sqrt(340), NOT sqrt(35000)

    Usage:
        calibrated = CalibratedTwoStageModel(model, trade_threshold=0.55)
        results = calibrated.predict_with_sizing(prices, features)
    """

    # Volatility regime thresholds (percentiles from training data)
    VOL_LOW_THRESHOLD = 0.33   # Below this = low volatility
    VOL_HIGH_THRESHOLD = 0.66  # Above this = high volatility (avoid!)

    def __init__(
        self,
        model: 'TwoStageModel',
        trade_threshold: float = 0.55,  # Optimal from analysis
        direction_threshold: float = 0.5,
        max_position: float = 1.0,
        use_position_sizing: bool = False,  # Disabled - doesn't improve returns
        filter_high_volatility: bool = True,  # Avoid high vol regime
        vol_threshold: float = None  # Custom high vol threshold (auto if None)
    ):
        super().__init__()
        self.model = model
        self.trade_threshold = trade_threshold
        self.direction_threshold = direction_threshold
        self.max_position = max_position
        self.use_position_sizing = use_position_sizing
        self.filter_high_volatility = filter_high_volatility
        self.vol_threshold = vol_threshold

    def forward(self, prices: torch.Tensor, features: torch.Tensor) -> dict:
        """Pass through to underlying model."""
        return self.model(prices, features)

    @torch.no_grad()
    def predict_with_sizing(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None
    ) -> dict:
        """
        Get predictions with calibrated thresholds and regime filtering.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]
            volatility: [batch] optional volatility values for regime filtering

        Returns:
            dict with:
            - should_trade: [batch] bool - whether to trade
            - is_long: [batch] bool - direction (True=long, False=short)
            - position_size: [batch] 0.0-1.0 position size
            - trade_prob: [batch] probability of trade
            - direction_confidence: [batch] confidence in direction
            - regime_filtered: [batch] bool - True if filtered by regime
        """
        outputs = self.model(prices, features)

        # Get probabilities
        tradeable_probs = F.softmax(outputs['tradeable_logits'], dim=-1)
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)

        trade_prob = tradeable_probs[:, 1]  # P(trade)
        long_prob = direction_probs[:, 0]   # P(long)
        short_prob = direction_probs[:, 1]  # P(short)

        # Apply calibrated threshold
        should_trade = trade_prob >= self.trade_threshold

        # Volatility regime filtering
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
        if self.use_position_sizing:
            # Scale trade_prob from [threshold, 1] to [0, 1]
            scaled_prob = (trade_prob - self.trade_threshold) / (1.0 - self.trade_threshold)
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
