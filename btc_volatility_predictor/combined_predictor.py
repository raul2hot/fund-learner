"""
Combined volatility regime + direction predictor.

Combines predictions from:
1. Volatility regime model (HIGH/LOW)
2. Direction model (UP/DOWN)

Returns combined predictions like: LOW_UP, LOW_DOWN, HIGH_UP, HIGH_DOWN
"""

import torch
import numpy as np
from typing import Optional

from config import Config
from models import SPHNet


class CombinedPredictor:
    """
    Predicts both volatility regime and price direction.

    Returns: {
        'vol_regime': 'HIGH' or 'LOW',
        'direction': 'UP' or 'DOWN',
        'combined': 'LOW_UP', 'LOW_DOWN', 'HIGH_UP', or 'HIGH_DOWN',
        'vol_prob': float,
        'dir_prob': float,
        'vol_confidence': float (0-1),
        'dir_confidence': float (0-1)
    }
    """

    def __init__(
        self,
        vol_checkpoint: str = "checkpoints/best_regime_model_90d.pt",
        dir_checkpoint: Optional[str] = "checkpoints/best_direction_model.pt",
        device: str = 'cpu'
    ):
        self.device = torch.device(device)

        # Load volatility model
        self.vol_model, self.vol_config = self._load_model(vol_checkpoint)

        # Load direction model (optional)
        self.dir_model = None
        self.dir_config = None
        if dir_checkpoint:
            try:
                self.dir_model, self.dir_config = self._load_model(dir_checkpoint)
            except FileNotFoundError:
                print(f"Warning: Direction model not found at {dir_checkpoint}")
                print("Direction prediction will be disabled.")

    def _load_model(self, checkpoint_path: str):
        """Load a model from checkpoint."""
        config = Config()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if 'config' in checkpoint:
            for k, v in checkpoint['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)

        model = SPHNet(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, config

    @torch.no_grad()
    def predict(self, prices: torch.Tensor, features: torch.Tensor) -> dict:
        """
        Predict both regime and direction.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_eng_features]

        Returns:
            dict with predictions and probabilities
        """
        if prices.dim() == 2:
            prices = prices.unsqueeze(0)
            features = features.unsqueeze(0)

        prices = prices.to(self.device)
        features = features.to(self.device)

        # Volatility prediction
        vol_out = self.vol_model(prices, features)
        vol_prob = torch.sigmoid(vol_out['direction_pred']).cpu().numpy().flatten()[0]
        vol_regime = 'HIGH' if vol_prob > 0.5 else 'LOW'

        # Direction prediction (if model available)
        if self.dir_model is not None:
            dir_out = self.dir_model(prices, features)
            dir_prob = torch.sigmoid(dir_out['direction_pred']).cpu().numpy().flatten()[0]
            direction = 'UP' if dir_prob > 0.5 else 'DOWN'
        else:
            dir_prob = 0.5
            direction = 'UNKNOWN'

        return {
            'vol_regime': vol_regime,
            'direction': direction,
            'combined': f"{vol_regime}_{direction}",
            'vol_prob': float(vol_prob),
            'dir_prob': float(dir_prob),
            'vol_confidence': float(abs(vol_prob - 0.5) * 2),
            'dir_confidence': float(abs(dir_prob - 0.5) * 2) if self.dir_model else 0.0
        }

    @torch.no_grad()
    def predict_batch(self, prices: torch.Tensor, features: torch.Tensor) -> list[dict]:
        """
        Predict for a batch of samples.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_eng_features]

        Returns:
            list of prediction dicts
        """
        prices = prices.to(self.device)
        features = features.to(self.device)

        # Volatility prediction
        vol_out = self.vol_model(prices, features)
        vol_probs = torch.sigmoid(vol_out['direction_pred']).cpu().numpy().flatten()

        # Direction prediction (if model available)
        if self.dir_model is not None:
            dir_out = self.dir_model(prices, features)
            dir_probs = torch.sigmoid(dir_out['direction_pred']).cpu().numpy().flatten()
        else:
            dir_probs = np.full_like(vol_probs, 0.5)

        results = []
        for i in range(len(vol_probs)):
            vol_prob = vol_probs[i]
            dir_prob = dir_probs[i]

            vol_regime = 'HIGH' if vol_prob > 0.5 else 'LOW'
            direction = 'UP' if dir_prob > 0.5 else 'DOWN'
            if self.dir_model is None:
                direction = 'UNKNOWN'

            results.append({
                'vol_regime': vol_regime,
                'direction': direction,
                'combined': f"{vol_regime}_{direction}",
                'vol_prob': float(vol_prob),
                'dir_prob': float(dir_prob),
                'vol_confidence': float(abs(vol_prob - 0.5) * 2),
                'dir_confidence': float(abs(dir_prob - 0.5) * 2) if self.dir_model else 0.0
            })

        return results

    def get_trading_recommendation(self, prediction: dict) -> dict:
        """
        Get trading recommendation based on combined prediction.

        Strategy decision matrix:
        - LOW + UP: LONG mean reversion
        - LOW + DOWN: SHORT mean reversion
        - HIGH + UP: NO TRADE (or cautious trend follow)
        - HIGH + DOWN: NO TRADE (or exit existing)

        Returns:
            dict with recommendation and reasoning
        """
        vol = prediction['vol_regime']
        direction = prediction['direction']
        vol_conf = prediction['vol_confidence']
        dir_conf = prediction['dir_confidence']

        if vol == 'HIGH':
            return {
                'action': 'NO_TRADE',
                'reasoning': 'High volatility regime - avoid trading',
                'confidence': vol_conf,
                'suggested_bias': None
            }

        # LOW volatility
        if direction == 'UP':
            return {
                'action': 'LONG_BIAS',
                'reasoning': 'Low volatility + upward direction - favor long mean reversion',
                'confidence': min(vol_conf, dir_conf),
                'suggested_bias': 'LONG'
            }
        elif direction == 'DOWN':
            return {
                'action': 'SHORT_BIAS',
                'reasoning': 'Low volatility + downward direction - favor short mean reversion',
                'confidence': min(vol_conf, dir_conf),
                'suggested_bias': 'SHORT'
            }
        else:
            return {
                'action': 'NEUTRAL',
                'reasoning': 'Low volatility but unknown direction',
                'confidence': vol_conf,
                'suggested_bias': None
            }


def demo():
    """Demo the combined predictor."""
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    print("Loading combined predictor...")

    try:
        predictor = CombinedPredictor()
        print("Combined predictor loaded successfully!")
        print(f"  Volatility model: Loaded")
        print(f"  Direction model: {'Loaded' if predictor.dir_model else 'Not available'}")
    except FileNotFoundError as e:
        print(f"Could not load models: {e}")
        print("Make sure to train models first with:")
        print("  python train_regime_extended.py")
        print("  python train_direction.py")
        return

    # Create dummy data for demo
    print("\nCreating dummy prediction...")
    window_size = 48
    n_price_features = 9
    n_eng_features = 30

    prices = torch.randn(1, window_size, n_price_features)
    features = torch.randn(1, window_size, n_eng_features)

    prediction = predictor.predict(prices, features)

    print("\nPrediction Results:")
    print(f"  Volatility Regime: {prediction['vol_regime']} (prob: {prediction['vol_prob']:.2%})")
    print(f"  Direction: {prediction['direction']} (prob: {prediction['dir_prob']:.2%})")
    print(f"  Combined: {prediction['combined']}")
    print(f"  Vol Confidence: {prediction['vol_confidence']:.2%}")
    print(f"  Dir Confidence: {prediction['dir_confidence']:.2%}")

    recommendation = predictor.get_trading_recommendation(prediction)
    print("\nTrading Recommendation:")
    print(f"  Action: {recommendation['action']}")
    print(f"  Reasoning: {recommendation['reasoning']}")
    print(f"  Confidence: {recommendation['confidence']:.2%}")


if __name__ == "__main__":
    demo()
