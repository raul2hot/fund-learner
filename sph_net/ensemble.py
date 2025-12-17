"""
Ensemble Model Module

Combines predictions from multiple trained models (different seeds)
to improve robustness and reduce variance.

Key insight: Seed 1337 was profitable in May 2021 (+15.91%) while
other seeds lost -47% to -78%. Averaging includes crash-resistant
signals in every prediction.

Architecture:
    - ModelLoader: Loads trained models from different seeds
    - EnsemblePredictor: Combines predictions using various methods
    - PerformanceWeightedEnsemble: Weights by validation performance

Usage:
    from sph_net.ensemble import (
        EnsemblePredictor,
        ModelLoader,
        create_ensemble_from_walk_forward
    )

    # Load and create ensemble
    ensemble = create_ensemble_from_walk_forward(
        results_dir="experiments/walk_forward",
        method=EnsembleMethod.WEIGHTED,
        weight_by_period="period_1_may2021"  # Weight by crash performance
    )

    # Generate predictions
    result = ensemble.predict(prices, features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from .config import SPHNetConfig
from .models.two_stage import TwoStageModel

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    MEAN = "mean"           # Simple average of logits
    WEIGHTED = "weighted"   # Performance-weighted average
    MEDIAN = "median"       # Median (robust to outliers)
    VOTING = "voting"       # Majority vote for direction


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictions."""
    method: EnsembleMethod = EnsembleMethod.MEAN
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    weights: Dict[int, float] = None  # Manual weights per seed
    confidence_threshold: float = 0.5  # Min agreement for voting
    normalize_predictions: bool = True  # Normalize before combining
    trade_threshold: float = 0.55  # From calibrated model
    stop_loss_pct: float = -0.02  # -2.0% stop-loss
    # Volatility filtering (matches CalibratedTwoStageModel)
    filter_high_volatility: bool = True  # Filter trades during high volatility
    vol_high_threshold: float = 0.66  # 66th percentile = high volatility regime

    def __post_init__(self):
        if self.weights is None:
            self.weights = {}


class ModelLoader:
    """
    Utility class to load trained models from different seeds.

    Handles the checkpoint format used by walk_forward_validation.py:
    - checkpoint['model_state_dict']: The model weights
    - checkpoint['config']: SPHNetConfig instance
    - checkpoint['metrics']: Validation metrics
    """

    def __init__(self, base_dir: Union[str, Path]):
        """
        Args:
            base_dir: Base directory containing seed folders (e.g., experiments/walk_forward)
        """
        self.base_dir = Path(base_dir)
        self.loaded_models: Dict[int, nn.Module] = {}
        self.model_configs: Dict[int, SPHNetConfig] = {}
        self.model_metrics: Dict[int, dict] = {}

    def load_model(
        self,
        seed: int,
        period: str = "period_5_full",
        checkpoint: str = "best_model.pt",
        device: str = None
    ) -> nn.Module:
        """
        Load a trained model from a specific seed.

        Args:
            seed: Random seed used during training
            period: Training period folder name
            checkpoint: Checkpoint filename
            device: Device to load model on (auto-detect if None)

        Returns:
            Loaded PyTorch model in eval mode
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Construct path
        model_path = self.base_dir / f"seed_{seed}" / period / checkpoint

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)

        # Extract config
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            self.model_configs[seed] = config
        else:
            # Create default config if not saved
            logger.warning(f"Seed {seed}: No config in checkpoint, using defaults")
            config = SPHNetConfig()
            self.model_configs[seed] = config

        # Extract metrics if available
        if 'metrics' in checkpoint_data:
            self.model_metrics[seed] = checkpoint_data['metrics']

        # Instantiate model
        model = TwoStageModel(config)

        # Load state dict
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            # Assume checkpoint is just state_dict
            model.load_state_dict(checkpoint_data)

        model = model.to(device)
        model.eval()

        self.loaded_models[seed] = model
        return model

    def load_all_seeds(
        self,
        seeds: List[int] = None,
        period: str = "period_5_full",
        checkpoint: str = "best_model.pt",
        device: str = None
    ) -> Dict[int, nn.Module]:
        """
        Load models from all specified seeds.

        Args:
            seeds: List of seeds to load (default: [42, 123, 456, 789, 1337])
            period: Training period folder name
            checkpoint: Checkpoint filename
            device: Device to load models on

        Returns:
            Dictionary mapping seed -> loaded model
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 1337]

        logger.info(f"Loading models from {self.base_dir}")

        for seed in seeds:
            try:
                self.load_model(seed, period, checkpoint, device)
                logger.info(f"  [OK] Loaded seed {seed}")
            except FileNotFoundError as e:
                logger.warning(f"  [SKIP] Seed {seed}: {e}")
            except Exception as e:
                logger.error(f"  [ERROR] Seed {seed}: {e}")

        logger.info(f"Successfully loaded {len(self.loaded_models)}/{len(seeds)} models")
        return self.loaded_models

    def get_available_periods(self, seed: int = 42) -> List[str]:
        """List available periods for a given seed."""
        seed_dir = self.base_dir / f"seed_{seed}"
        if not seed_dir.exists():
            return []
        return [d.name for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith("period_")]

    def get_expected_dimensions(self) -> Dict[str, int]:
        """
        Get expected input dimensions from loaded model configs.

        Returns:
            Dict with n_price_features, n_engineered_features, window_size
        """
        if not self.model_configs:
            return {
                'n_price_features': 5,
                'n_engineered_features': 34,
                'window_size': 64
            }

        # Use first available config
        config = next(iter(self.model_configs.values()))

        return {
            'n_price_features': getattr(config, 'n_price_features', 5),
            'n_engineered_features': getattr(config, 'n_engineered_features', 34),
            'window_size': getattr(config, 'window_size', 64)
        }


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models.

    Supports multiple combination methods:
    - MEAN: Simple average of logits
    - WEIGHTED: Performance-weighted average
    - MEDIAN: Median (robust to outliers)
    - VOTING: Majority vote with probability weighting

    Usage:
        loader = ModelLoader("experiments/walk_forward")
        loader.load_all_seeds([42, 123, 456, 789, 1337])

        ensemble = EnsemblePredictor(loader.loaded_models)
        predictions = ensemble.predict(prices, features)
    """

    def __init__(
        self,
        models: Dict[int, nn.Module],
        config: Optional[EnsembleConfig] = None
    ):
        """
        Args:
            models: Dictionary mapping seed -> loaded model
            config: Ensemble configuration
        """
        if not models:
            raise ValueError("At least one model required")

        self.models = models
        self.config = config or EnsembleConfig()

        # Get device from first model
        first_model = next(iter(models.values()))
        self.device = next(first_model.parameters()).device

        # Initialize equal weights if not provided
        if not self.config.weights:
            self.config.weights = {seed: 1.0 for seed in models.keys()}

        # Normalize weights to sum to 1
        self._normalize_weights()

        logger.info(f"Ensemble initialized with {len(models)} models, method={self.config.method.value}")

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.config.weights.get(seed, 1.0) for seed in self.models.keys())
        if total > 0:
            for seed in self.models.keys():
                w = self.config.weights.get(seed, 1.0)
                self.config.weights[seed] = w / total

    @torch.no_grad()
    def predict_single_model(
        self,
        model: nn.Module,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from a single TwoStageModel.

        Args:
            model: The TwoStageModel to use
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]

        Returns:
            Dict with tradeable_logits, direction_logits, logits, return_pred
        """
        model.eval()
        output = model(prices, features)

        return {
            'tradeable_logits': output['tradeable_logits'],
            'direction_logits': output['direction_logits'],
            'logits': output['logits'],  # Combined 5-class
            'return_pred': output['return_pred']
        }

    @torch.no_grad()
    def predict(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble predictions.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]
            volatility: [batch] optional volatility values for regime filtering
            return_individual: If True, also return individual model predictions

        Returns:
            Dictionary with keys:
            - 'tradeable_logits': Combined tradeable logits [batch, 2]
            - 'direction_logits': Combined direction logits [batch, 2]
            - 'logits': Combined 5-class logits [batch, 5]
            - 'tradeable_prob': Probability of trade [batch]
            - 'is_long': Direction prediction (True=long) [batch]
            - 'should_trade': Trade signal based on threshold [batch]
            - 'confidence': Ensemble confidence [batch]
            - 'agreement': Fraction of models agreeing [batch]
            - 'volatility_filtered': [batch] bool - True if filtered by volatility
            - 'individual_predictions': (optional) Dict of per-seed predictions
        """
        prices = prices.to(self.device)
        features = features.to(self.device)
        if volatility is not None:
            volatility = volatility.to(self.device)

        # Collect predictions from all models
        all_tradeable_logits = []
        all_direction_logits = []
        all_logits = []
        individual_preds = {}

        seeds_used = []
        weights_used = []

        for seed, model in self.models.items():
            if self.config.weights and seed not in self.config.weights:
                continue

            output = self.predict_single_model(model, prices, features)

            all_tradeable_logits.append(output['tradeable_logits'])
            all_direction_logits.append(output['direction_logits'])
            all_logits.append(output['logits'])

            seeds_used.append(seed)
            weights_used.append(self.config.weights.get(seed, 1.0))

            if return_individual:
                tradeable_probs = F.softmax(output['tradeable_logits'], dim=-1)
                direction_probs = F.softmax(output['direction_logits'], dim=-1)

                individual_preds[seed] = {
                    'tradeable_prob': tradeable_probs[:, 1].cpu(),
                    'long_prob': direction_probs[:, 0].cpu(),
                    'should_trade': (tradeable_probs[:, 1] >= self.config.trade_threshold).cpu(),
                    'is_long': (direction_probs[:, 0] > 0.5).cpu(),
                    'logits': output['logits'].cpu(),
                }

        # Stack predictions
        stacked_tradeable = torch.stack(all_tradeable_logits, dim=0)  # [n_models, batch, 2]
        stacked_direction = torch.stack(all_direction_logits, dim=0)  # [n_models, batch, 2]
        stacked_logits = torch.stack(all_logits, dim=0)  # [n_models, batch, 5]

        # Convert weights to tensor
        weights_tensor = torch.tensor(weights_used, device=self.device).view(-1, 1, 1)

        # Combine based on method
        if self.config.method == EnsembleMethod.MEAN:
            combined_tradeable = stacked_tradeable.mean(dim=0)
            combined_direction = stacked_direction.mean(dim=0)
            combined_logits = stacked_logits.mean(dim=0)

        elif self.config.method == EnsembleMethod.WEIGHTED:
            combined_tradeable = (stacked_tradeable * weights_tensor).sum(dim=0)
            combined_direction = (stacked_direction * weights_tensor).sum(dim=0)
            combined_logits = (stacked_logits * weights_tensor).sum(dim=0)

        elif self.config.method == EnsembleMethod.MEDIAN:
            combined_tradeable = stacked_tradeable.median(dim=0).values
            combined_direction = stacked_direction.median(dim=0).values
            combined_logits = stacked_logits.median(dim=0).values

        elif self.config.method == EnsembleMethod.VOTING:
            # Voting based on trade decisions
            combined_tradeable, combined_direction, combined_logits = self._voting_combine(
                stacked_tradeable, stacked_direction, stacked_logits, weights_tensor
            )

        # Calculate probabilities
        tradeable_probs = F.softmax(combined_tradeable, dim=-1)
        direction_probs = F.softmax(combined_direction, dim=-1)

        tradeable_prob = tradeable_probs[:, 1]  # P(trade)
        long_prob = direction_probs[:, 0]  # P(long)

        # Trade decisions
        should_trade = tradeable_prob >= self.config.trade_threshold
        is_long = long_prob > 0.5

        # Volatility regime filtering (matches CalibratedTwoStageModel)
        batch_size = prices.shape[0]
        volatility_filtered = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if self.config.filter_high_volatility and volatility is not None:
            # Use configurable percentile as high volatility threshold
            vol_thresh = torch.quantile(volatility, self.config.vol_high_threshold)
            is_high_vol = volatility > vol_thresh

            # Track which trades would be filtered
            volatility_filtered = is_high_vol & should_trade

            # Filter out trades during high volatility
            should_trade = should_trade & ~is_high_vol

        # Calculate agreement (how many models agree with ensemble decision)
        individual_should_trade = F.softmax(stacked_tradeable, dim=-1)[:, :, 1] >= self.config.trade_threshold
        individual_is_long = F.softmax(stacked_direction, dim=-1)[:, :, 0] > 0.5

        trade_agreement = (individual_should_trade == should_trade.unsqueeze(0)).float().mean(dim=0)
        direction_agreement = (individual_is_long == is_long.unsqueeze(0)).float().mean(dim=0)

        # Combined agreement (for trades: agree on both trade and direction)
        agreement = torch.where(
            should_trade,
            trade_agreement * direction_agreement,
            trade_agreement
        )

        # Confidence = trade probability * agreement
        confidence = tradeable_prob * agreement

        result = {
            'tradeable_logits': combined_tradeable,
            'direction_logits': combined_direction,
            'logits': combined_logits,
            'tradeable_prob': tradeable_prob,
            'long_prob': long_prob,
            'should_trade': should_trade,
            'is_long': is_long,
            'confidence': confidence,
            'agreement': agreement,
            'trade_agreement': trade_agreement,
            'direction_agreement': direction_agreement,
            'volatility_filtered': volatility_filtered,
        }

        if return_individual:
            result['individual_predictions'] = individual_preds
            result['seeds_used'] = seeds_used
            result['weights_used'] = weights_used

        return result

    def _voting_combine(
        self,
        stacked_tradeable: torch.Tensor,
        stacked_direction: torch.Tensor,
        stacked_logits: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voting-based combination."""
        # Get individual decisions
        tradeable_probs = F.softmax(stacked_tradeable, dim=-1)  # [n_models, batch, 2]
        direction_probs = F.softmax(stacked_direction, dim=-1)  # [n_models, batch, 2]

        # Weighted vote for tradeable
        should_trade = (tradeable_probs[:, :, 1] >= self.config.trade_threshold).float()
        weighted_trade_vote = (should_trade * weights.squeeze(-1)).sum(dim=0)  # [batch]

        # Weighted vote for direction (only from models that say trade)
        is_long = (direction_probs[:, :, 0] > 0.5).float()
        trade_mask = should_trade

        # Avoid division by zero
        trade_weight_sum = (trade_mask * weights.squeeze(-1)).sum(dim=0).clamp(min=1e-8)
        weighted_long_vote = ((is_long * trade_mask) * weights.squeeze(-1)).sum(dim=0) / trade_weight_sum

        # Convert votes back to logits
        # Trade: if weighted vote >= 0.5, predict trade
        trade_prob = weighted_trade_vote.clamp(0.01, 0.99)
        combined_tradeable = torch.stack([
            torch.log(1 - trade_prob + 1e-8),  # P(no trade)
            torch.log(trade_prob + 1e-8)  # P(trade)
        ], dim=-1)

        # Direction
        long_prob = weighted_long_vote.clamp(0.01, 0.99)
        combined_direction = torch.stack([
            torch.log(long_prob + 1e-8),  # P(long)
            torch.log(1 - long_prob + 1e-8)  # P(short)
        ], dim=-1)

        # For 5-class logits, use weighted average
        combined_logits = (stacked_logits * weights).sum(dim=0)

        return combined_tradeable, combined_direction, combined_logits

    def get_trade_signal(
        self,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> List[Dict]:
        """
        Get human-readable trade signals.

        Returns list of dicts with action, confidence, etc.
        """
        result = self.predict(prices, features)

        signals = []
        batch_size = result['should_trade'].shape[0]

        for i in range(batch_size):
            if result['should_trade'][i]:
                direction = 'LONG' if result['is_long'][i] else 'SHORT'
                signals.append({
                    'action': direction,
                    'confidence': result['confidence'][i].item(),
                    'tradeable_prob': result['tradeable_prob'][i].item(),
                    'agreement': result['agreement'][i].item(),
                })
            else:
                signals.append({
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'tradeable_prob': result['tradeable_prob'][i].item(),
                    'agreement': result['agreement'][i].item(),
                })

        return signals


class PerformanceWeightedEnsemble(EnsemblePredictor):
    """
    Ensemble that weights models by their validation performance.

    Better-performing seeds get higher weight in the ensemble.
    This is particularly useful when some seeds learned crash-resistant
    features (like seed 1337) while others didn't.

    Weight Calculation:
        Uses z-score normalization followed by softmax to convert returns
        into weights. This ensures:
        1. All models contribute meaningfully (no zero weights)
        2. Better performers get 2-3x weight, not 100x
        3. The weighting is statistically principled

    Usage:
        # Weight by May 2021 performance (crash resistance)
        validation_returns = {
            42: -73.13,
            123: -46.64,
            456: -77.52,
            789: -2.58,
            1337: +15.91
        }

        ensemble = PerformanceWeightedEnsemble(
            models,
            validation_returns,
            temperature=2.0  # Controls weight concentration (higher = more uniform)
        )
    """

    def __init__(
        self,
        models: Dict[int, nn.Module],
        validation_returns: Dict[int, float],
        config: Optional[EnsembleConfig] = None,
        temperature: float = 2.0
    ):
        """
        Args:
            models: Dictionary mapping seed -> loaded model
            validation_returns: Dictionary mapping seed -> validation return (%)
            config: Ensemble configuration
            temperature: Softmax temperature for weight calculation
                        Higher = more uniform weights
                        Lower = more concentrated on best performer
                        Recommended: 2.0 (gives best performer ~2-3x weight of worst)
        """
        # Calculate weights from validation performance
        seeds = list(models.keys())
        returns = np.array([validation_returns.get(seed, 0) for seed in seeds])

        # Z-score normalize returns to handle different scales
        # This ensures the softmax operates on a standardized scale
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            z_returns = (returns - mean_return) / std_return
        else:
            # All returns equal - use uniform weights
            z_returns = np.zeros_like(returns)

        # Apply softmax with temperature to normalized returns
        # Temperature=2.0 with z-scores gives reasonable weight spread:
        # Best performer gets ~2-3x the weight of worst performer
        exp_returns = np.exp(z_returns / temperature)
        weights = exp_returns / exp_returns.sum()

        # Create weights dict
        weight_dict = {seed: float(w) for seed, w in zip(seeds, weights)}

        # Initialize config
        if config is None:
            config = EnsembleConfig(method=EnsembleMethod.WEIGHTED)
        config.weights = weight_dict
        config.method = EnsembleMethod.WEIGHTED

        super().__init__(models, config)

        # Log weight distribution
        logger.info("Performance-weighted ensemble initialized:")
        logger.info(f"  Returns: mean={mean_return:.2f}%, std={std_return:.2f}%")
        for seed in sorted(seeds):
            ret = validation_returns.get(seed, 0)
            w = weight_dict[seed]
            logger.info(f"  Seed {seed}: return={ret:+.2f}%, weight={w:.3f}")


def load_validation_results(results_dir: Path) -> Dict[int, Dict[str, float]]:
    """
    Load validation results for all seeds.

    Args:
        results_dir: Directory containing walk-forward results

    Returns:
        Dictionary mapping seed -> {period -> return}
    """
    results_dir = Path(results_dir)
    results = {}

    for seed_dir in results_dir.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        results[seed] = {}

        # Try to load from seed summary (structure: {"seed": X, "results": {period: {metrics: {...}}}})
        summary_path = seed_dir / "seed_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                    # Handle structure from walk_forward_validation.py
                    if 'results' in summary:
                        for period, data in summary['results'].items():
                            if isinstance(data, dict) and 'metrics' in data:
                                results[seed][period] = data['metrics'].get('total_return', 0)
                            elif isinstance(data, dict):
                                results[seed][period] = data.get('total_return', 0)
                        if results[seed]:
                            continue
                    # Also try 'periods' key for backwards compatibility
                    if 'periods' in summary:
                        for period, metrics in summary['periods'].items():
                            results[seed][period] = metrics.get('total_return', 0)
                        continue
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading {summary_path}: {e}")

        # Fallback: load from individual period results
        for period_dir in seed_dir.glob("period_*"):
            results_path = period_dir / "test_results.json"
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        period_results = json.load(f)
                        results[seed][period_dir.name] = period_results.get('total_return', 0)
                except (json.JSONDecodeError, KeyError):
                    pass

    return results


def create_ensemble_from_walk_forward(
    results_dir: Union[str, Path],
    method: EnsembleMethod = EnsembleMethod.WEIGHTED,
    period: str = "period_5_full",
    weight_by_period: str = "period_1_may2021",
    seeds: List[int] = None,
    device: str = None,
    temperature: float = 2.0
) -> EnsemblePredictor:
    """
    Create an ensemble from walk-forward validation results.

    This is the main factory function for creating ensemble models.

    Args:
        results_dir: Directory containing walk-forward results
        method: Ensemble combination method
        period: Which period's models to load
        weight_by_period: Which period to use for performance weighting
        seeds: Seeds to include (default: [42, 123, 456, 789, 1337])
        device: Device to load models on (auto-detect if None)
        temperature: Temperature for performance weighting softmax

    Returns:
        Configured EnsemblePredictor

    Example:
        ensemble = create_ensemble_from_walk_forward(
            "experiments/walk_forward",
            method=EnsembleMethod.WEIGHTED,
            weight_by_period="period_1_may2021"  # Weight by crash performance!
        )
    """
    results_dir = Path(results_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seeds is None:
        seeds = [42, 123, 456, 789, 1337]

    # Load models
    logger.info(f"Creating ensemble from {results_dir}")
    loader = ModelLoader(results_dir)
    loader.load_all_seeds(seeds, period=period, device=device)

    if not loader.loaded_models:
        raise ValueError(f"No models loaded from {results_dir}")

    if method == EnsembleMethod.WEIGHTED:
        # Load validation results for weighting
        logger.info(f"Loading validation results for weighting by {weight_by_period}")
        validation_results = load_validation_results(results_dir)

        # Extract returns for the weighting period
        weight_returns = {}
        for seed in loader.loaded_models.keys():
            if seed in validation_results and weight_by_period in validation_results[seed]:
                weight_returns[seed] = validation_results[seed][weight_by_period]
            else:
                weight_returns[seed] = 0.0
                logger.warning(f"Seed {seed}: No results for {weight_by_period}, using weight=0")

        ensemble = PerformanceWeightedEnsemble(
            loader.loaded_models,
            weight_returns,
            temperature=temperature
        )
    else:
        config = EnsembleConfig(method=method)
        ensemble = EnsemblePredictor(loader.loaded_models, config)

    return ensemble


def ensemble_predict_dataframe(
    ensemble: EnsemblePredictor,
    data: pd.DataFrame,
    price_columns: List[str],
    feature_columns: List[str],
    window_size: int = 64,
    batch_size: int = 256,
    return_individual: bool = False
) -> pd.DataFrame:
    """
    Generate ensemble predictions for a DataFrame.

    Useful for backtesting and analysis.

    Args:
        ensemble: Configured EnsemblePredictor
        data: DataFrame with price and feature columns
        price_columns: List of price column names (OHLCV)
        feature_columns: List of feature column names
        window_size: Lookback window size
        batch_size: Batch size for prediction
        return_individual: Include individual model predictions

    Returns:
        DataFrame with predictions aligned to original timestamps
    """
    device = ensemble.device
    predictions = []

    # Validate columns exist
    missing_price = [c for c in price_columns if c not in data.columns]
    missing_features = [c for c in feature_columns if c not in data.columns]

    if missing_price:
        raise ValueError(f"Missing price columns in data: {missing_price}")
    if missing_features:
        logger.warning(f"Missing feature columns in data: {missing_features[:5]}... ({len(missing_features)} total)")
        # Filter to only existing columns
        feature_columns = [c for c in feature_columns if c in data.columns]

    # Log dimensions
    logger.info(f"Input dimensions: {len(price_columns)} price, {len(feature_columns)} features")

    # Check for NaN values
    prices_arr = data[price_columns].values.astype(np.float32)
    features_arr = data[feature_columns].values.astype(np.float32)

    nan_count = np.isnan(features_arr).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
        features_arr = np.nan_to_num(features_arr, nan=0.0)

    timestamps = data['timestamp'].values

    n_samples = len(prices_arr) - window_size + 1

    if n_samples <= 0:
        logger.warning(f"Not enough data: {len(prices_arr)} rows < window_size {window_size}")
        return pd.DataFrame()

    logger.info(f"Generating predictions for {n_samples} samples...")

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        # Create batch of sequences
        batch_prices = []
        batch_features = []

        for i in range(start_idx, end_idx):
            batch_prices.append(prices_arr[i:i + window_size])
            batch_features.append(features_arr[i:i + window_size])

        # Convert to tensors
        prices_tensor = torch.tensor(np.array(batch_prices), dtype=torch.float32, device=device)
        features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32, device=device)

        # Compute volatility from features (matches CalibratedTwoStageModel)
        # Volatility = std of features at the last timestep across all features
        volatility = features_tensor[:, -1, :].std(dim=-1)

        # Get predictions with volatility filtering
        result = ensemble.predict(
            prices_tensor, features_tensor,
            volatility=volatility,
            return_individual=return_individual
        )

        # Store results
        for j in range(end_idx - start_idx):
            idx = start_idx + j
            pred = {
                'timestamp': timestamps[idx + window_size - 1],
                'should_trade': result['should_trade'][j].item(),
                'is_long': result['is_long'][j].item(),
                'tradeable_prob': result['tradeable_prob'][j].item(),
                'long_prob': result['long_prob'][j].item(),
                'confidence': result['confidence'][j].item(),
                'agreement': result['agreement'][j].item(),
                'volatility_filtered': result['volatility_filtered'][j].item(),
            }

            # Add individual predictions if requested
            if return_individual and 'individual_predictions' in result:
                for seed, ind_pred in result['individual_predictions'].items():
                    pred[f'should_trade_seed_{seed}'] = ind_pred['should_trade'][j].item()
                    pred[f'is_long_seed_{seed}'] = ind_pred['is_long'][j].item()
                    pred[f'tradeable_prob_seed_{seed}'] = ind_pred['tradeable_prob'][j].item()

            predictions.append(pred)

    logger.info(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


def calculate_ensemble_trading_returns(
    predictions: pd.DataFrame,
    price_data: pd.DataFrame,
    stop_loss_pct: float = -0.02,
    use_regime_filter: bool = True,
    regime_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Calculate trading returns from ensemble predictions.

    IMPORTANT: Uses open-to-close returns (same as individual seed methodology)
    to ensure apples-to-apples comparison.

    Args:
        predictions: DataFrame with should_trade, is_long columns
        price_data: DataFrame with timestamp, open, close, next_return,
                   next_mae_long, next_mae_short columns (from labeling)
        stop_loss_pct: Stop-loss as decimal (e.g., -0.02 for -2%)
        use_regime_filter: Whether to apply regime filtering
        regime_config: Optional regime filter configuration

    Returns:
        Dictionary with trading metrics
    """
    # Ensure timestamp columns have matching timezone
    pred_df = predictions.copy()

    # Get required columns from price_data
    available_cols = [c for c in price_data.columns if c in
                      ['timestamp', 'open', 'close', 'next_return',
                       'next_mae_long', 'next_mae_short']]
    price_df = price_data[available_cols].copy()

    # Convert both to UTC-aware if needed
    if pred_df['timestamp'].dt.tz is None:
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.tz_localize('UTC')
    if price_df['timestamp'].dt.tz is None:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp']).dt.tz_localize('UTC')

    # CRITICAL: Filter price_df to only include timestamps present in predictions
    # Predictions have fewer rows than price_data due to windowing
    pred_timestamps = set(pred_df['timestamp'])
    price_df_filtered = price_df[price_df['timestamp'].isin(pred_timestamps)].copy()

    logger.info(f"Price data: {len(price_df)} rows, Predictions: {len(pred_df)} rows, "
                f"Filtered price data: {len(price_df_filtered)} rows")

    # Merge predictions with filtered price data
    merged = pred_df.merge(
        price_df_filtered,
        on='timestamp',
        how='left'
    )

    # DIAGNOSTIC: Check if next_return column exists and has valid values
    if 'next_return' in merged.columns:
        n_valid = merged['next_return'].notna().sum()
        mean_return = merged['next_return'].mean()
        logger.info(f"next_return column: {n_valid}/{len(merged)} valid values, mean={mean_return:.6f}")
        merged['fwd_return'] = merged['next_return']
    else:
        # Fallback: calculate close-to-close returns
        logger.warning(f"next_return NOT FOUND - available columns: {list(merged.columns)[:15]}")
        if 'close' in merged.columns:
            merged['fwd_return'] = merged['close'].pct_change().shift(-1)
            logger.warning("Using close-to-close returns (not comparable to individual seeds)")
        else:
            logger.error("No return data available")
            return {'total_return': 0, 'n_trades': 0, 'sharpe': 0, 'win_rate': 0}

    # Apply regime filter if enabled
    n_regime_blocked = 0
    if use_regime_filter and 'close' in price_df_filtered.columns:
        try:
            from regime_filter import apply_regime_filter_vectorized, RegimePresets

            # Create a temp predictions df with required format
            temp_pred = merged[['timestamp', 'should_trade', 'is_long']].copy()

            # CRITICAL FIX: Filter price series to match prediction timestamps
            price_series = price_df_filtered.set_index('timestamp')['close']

            # Verify shapes match
            logger.info(f"Regime filter input shapes: predictions={len(temp_pred)}, price_series={len(price_series)}")

            # Get funding rates if available (also filtered)
            funding_series = None
            if 'funding_rate' in price_df_filtered.columns:
                funding_series = price_df_filtered.set_index('timestamp')['funding_rate']

            # Apply regime filter
            config = regime_config if regime_config else RegimePresets.moderate()
            filtered = apply_regime_filter_vectorized(
                temp_pred,
                price_series,
                funding_series,
                config=config
            )

            # Update should_trade based on regime filter
            if 'regime_blocked' in filtered.columns:
                n_regime_blocked = int(filtered['regime_blocked'].sum())
                # Merge back the regime_blocked column
                merged = merged.merge(
                    filtered[['timestamp', 'regime_blocked', 'regime']],
                    on='timestamp',
                    how='left'
                )
                merged['regime_blocked'] = merged['regime_blocked'].fillna(False)
            else:
                merged['regime_blocked'] = False

        except ImportError:
            logger.warning("regime_filter module not available, skipping regime filtering")
            merged['regime_blocked'] = False
        except Exception as e:
            logger.warning(f"Regime filter failed: {e}, skipping")
            import traceback
            traceback.print_exc()
            merged['regime_blocked'] = False
    else:
        merged['regime_blocked'] = False

    # Calculate trade returns based on position
    # Position: +1 for long (return = next_return), -1 for short (return = -next_return)
    merged['position'] = 0.0
    # Only count non-blocked trades
    trade_mask = merged['should_trade'] & ~merged['regime_blocked']
    merged.loc[trade_mask & merged['is_long'], 'position'] = 1.0
    merged.loc[trade_mask & ~merged['is_long'], 'position'] = -1.0

    # Trade returns: position * forward return
    merged['trade_return'] = merged['position'] * merged['fwd_return']

    # Apply MAE-aware stop-loss (same as individual seed methodology)
    n_stopped_out = 0
    if stop_loss_pct is not None:
        has_mae = 'next_mae_long' in merged.columns and 'next_mae_short' in merged.columns

        if has_mae:
            # MAE-aware stop-loss: check if intra-candle move hit stop
            # For long trades: check if mae_long (downside) > stop_loss threshold
            # For short trades: check if mae_short (upside) > stop_loss threshold
            long_mask = merged['position'] == 1.0
            short_mask = merged['position'] == -1.0

            # Long trades stopped if MAE exceeded stop-loss
            long_stopped = long_mask & (merged['next_mae_long'] > abs(stop_loss_pct))
            # Short trades stopped if MAE exceeded stop-loss
            short_stopped = short_mask & (merged['next_mae_short'] > abs(stop_loss_pct))

            stopped_mask = long_stopped | short_stopped
            n_stopped_out = int(stopped_mask.sum())

            # Set stopped trades to stop-loss return
            merged.loc[stopped_mask, 'trade_return'] = stop_loss_pct

            logger.info(f"MAE-aware stop-loss: {n_stopped_out} trades stopped out of {int(long_mask.sum() + short_mask.sum())}")
        else:
            # Fallback: simple clip (less accurate)
            logger.warning("MAE columns not available, using simple clip for stop-loss")
            before_clip = merged['trade_return'].copy()
            merged['trade_return'] = merged['trade_return'].clip(lower=stop_loss_pct)
            n_stopped_out = int((before_clip < stop_loss_pct).sum())

    # Remove NaN
    valid = merged.dropna(subset=['trade_return'])

    if len(valid) == 0:
        return {'total_return': 0, 'n_trades': 0, 'sharpe': 0, 'win_rate': 0}

    # Metrics
    trade_returns = valid[valid['position'] != 0]['trade_return']

    total_return = trade_returns.sum() * 100  # As percentage
    n_trades = int((valid['position'] != 0).sum())

    # DIAGNOSTIC: Log trade breakdown
    n_long = int((valid['position'] == 1.0).sum())
    n_short = int((valid['position'] == -1.0).sum())
    logger.info(f"Trades: {n_trades} total ({n_long} long, {n_short} short), "
                f"Return: {total_return:.2f}%")

    if len(trade_returns) > 0 and trade_returns.std() > 0:
        # Annualized Sharpe using trade frequency
        days = (valid['timestamp'].max() - valid['timestamp'].min()).days
        if days > 0:
            trades_per_year = n_trades / days * 365
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
        win_rate = (trade_returns > 0).mean() * 100
    else:
        sharpe = 0
        win_rate = 0

    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_return_per_trade': trade_returns.mean() * 100 if len(trade_returns) > 0 else 0,
        'n_stopped_out': n_stopped_out,
        'n_regime_blocked': n_regime_blocked,
    }
