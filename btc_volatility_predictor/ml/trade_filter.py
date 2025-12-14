#!/usr/bin/env python3
"""
Trade quality filter using trained XGBoost model.

Provides a simple interface for strategies to check
if a potential trade is predicted to be a WIN.

Usage:
    filter = TradeQualityFilter()

    # In strategy:
    if filter.should_trade(row, history, threshold=0.6):
        return 'BUY'
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

from .feature_engineering import TRADE_FEATURES, compute_additional_features


class TradeQualityFilter:
    """
    Filter trades based on XGBoost WIN/LOSS prediction.

    This class loads a trained XGBoost model and uses it to predict
    whether a potential trade will be a WIN or LOSS. Strategies can
    use this to filter out low-quality trades.

    Attributes:
        default_threshold: Default probability threshold for WIN prediction
        model: Loaded XGBoost Booster model
        feature_names: List of feature names in order used during training
    """

    def __init__(
        self,
        model_path: str = "checkpoints/trade_classifier.json",
        features_path: str = "checkpoints/trade_classifier_features.json",
        default_threshold: float = 0.5,
    ):
        """
        Initialize filter with trained model.

        Args:
            model_path: Path to XGBoost model file
            features_path: Path to feature names JSON
            default_threshold: Default probability threshold for WIN prediction
        """
        self.default_threshold = default_threshold
        self.model = None
        self.feature_names = None
        self._model_path = model_path
        self._features_path = features_path

        if not HAS_XGBOOST:
            print("WARNING: xgboost not installed. TradeQualityFilter will be disabled.")
            return

        # Load model
        if os.path.exists(model_path):
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                print(f"Loaded trade classifier from {model_path}")
            except Exception as e:
                print(f"WARNING: Could not load trade classifier from {model_path}: {e}")
        else:
            print(f"WARNING: Trade classifier not found at {model_path}")

        # Load feature names
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
            except Exception as e:
                print(f"WARNING: Could not load feature names from {features_path}: {e}")
        else:
            # Use default feature names if file not found
            self.feature_names = TRADE_FEATURES
            print(f"WARNING: Feature names not found at {features_path}, using defaults")

    def is_ready(self) -> bool:
        """
        Check if filter is properly initialized.

        Returns:
            True if model and feature names are loaded
        """
        return self.model is not None and self.feature_names is not None

    def extract_features(self, row: dict, history: List[dict]) -> np.ndarray:
        """
        Extract features from current row and history.

        Args:
            row: Current bar data (dict with feature values)
            history: List of previous bars

        Returns:
            Feature array in correct order for model prediction
        """
        features = {}

        # Core features from row
        for feat in self.feature_names:
            if feat in row:
                features[feat] = row[feat]
            else:
                features[feat] = 0.0

        # Compute additional features that need history
        computed = compute_additional_features(row, history)
        features.update(computed)

        # Convert to array in correct order
        return np.array([features.get(f, 0.0) for f in self.feature_names])

    def predict_win_probability(
        self,
        row: dict,
        history: List[dict],
    ) -> float:
        """
        Predict probability that trade will be a WIN.

        Args:
            row: Current bar data
            history: Previous bars

        Returns:
            Probability [0, 1] that trade will be WIN
        """
        if not self.is_ready():
            return 0.5  # Neutral if model not loaded

        features = self.extract_features(row, history)
        dmatrix = xgb.DMatrix(features.reshape(1, -1), feature_names=self.feature_names)

        prob = self.model.predict(dmatrix)[0]
        return float(prob)

    def should_trade(
        self,
        row: dict,
        history: List[dict],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if trade should be taken.

        Args:
            row: Current bar data
            history: Previous bars
            threshold: Probability threshold (default: self.default_threshold)

        Returns:
            True if model predicts WIN with probability >= threshold
        """
        if not self.is_ready():
            return True  # Allow trade if model not loaded

        threshold = threshold if threshold is not None else self.default_threshold
        prob = self.predict_win_probability(row, history)

        return prob >= threshold

    def get_trade_decision(
        self,
        row: dict,
        history: List[dict],
        threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Get detailed trade decision with probability and reasoning.

        Args:
            row: Current bar data
            history: Previous bars
            threshold: Probability threshold

        Returns:
            Dictionary with decision details:
            - should_trade: bool
            - win_probability: float
            - threshold_used: float
            - model_ready: bool
        """
        threshold = threshold if threshold is not None else self.default_threshold

        if not self.is_ready():
            return {
                'should_trade': True,
                'win_probability': 0.5,
                'threshold_used': threshold,
                'model_ready': False,
                'reason': 'Model not loaded, allowing trade'
            }

        prob = self.predict_win_probability(row, history)
        should_trade = prob >= threshold

        return {
            'should_trade': should_trade,
            'win_probability': prob,
            'threshold_used': threshold,
            'model_ready': True,
            'reason': f"P(WIN)={prob:.2f} {'>='}  threshold={threshold}" if should_trade
                     else f"P(WIN)={prob:.2f} < threshold={threshold}"
        }

    def __repr__(self) -> str:
        status = "ready" if self.is_ready() else "not ready"
        return f"TradeQualityFilter(threshold={self.default_threshold}, status={status})"


# Singleton instance for easy import
_default_filter: Optional[TradeQualityFilter] = None


def get_trade_filter(
    threshold: float = 0.5,
    model_path: str = "checkpoints/trade_classifier.json",
    features_path: str = "checkpoints/trade_classifier_features.json",
) -> TradeQualityFilter:
    """
    Get or create default trade filter instance.

    This provides a singleton pattern for easy access to the filter
    across the codebase without reloading the model each time.

    Args:
        threshold: Default probability threshold
        model_path: Path to model file
        features_path: Path to feature names file

    Returns:
        TradeQualityFilter instance
    """
    global _default_filter
    if _default_filter is None:
        _default_filter = TradeQualityFilter(
            model_path=model_path,
            features_path=features_path,
            default_threshold=threshold
        )
    return _default_filter


def reset_trade_filter() -> None:
    """Reset the singleton filter instance (useful for testing)."""
    global _default_filter
    _default_filter = None


if __name__ == "__main__":
    # Simple test
    print("Testing TradeQualityFilter...")

    filter = TradeQualityFilter()
    print(f"Filter status: {filter}")
    print(f"Is ready: {filter.is_ready()}")

    if filter.is_ready():
        # Create dummy test data
        test_row = {
            'close': 50000,
            'rsi_14': 40,
            'adx_14': 25,
        }
        test_history = [{'close': 49000, 'rsi_14': 45}] * 200

        decision = filter.get_trade_decision(test_row, test_history)
        print(f"Decision: {decision}")
