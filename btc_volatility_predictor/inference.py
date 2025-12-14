"""
Inference script for BTC volatility regime prediction.
Predicts whether next hour will be HIGH or LOW volatility.
"""

import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import pickle
import os

from config import Config
from models import SPHNet
from data.features import engineer_features


def fetch_recent_data(hours=72):
    """Fetch recent BTC/USDT 1h candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
               'taker_buy_quote', 'ignore']

    df = pd.DataFrame(data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)

    df['trades'] = df['trades'].astype(int)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'quote_volume']]

    return df


class VolatilityPredictor:
    """Inference wrapper for volatility regime prediction."""

    def __init__(self, checkpoint_path="checkpoints/best_regime_model.pt",
                 scalers_path="checkpoints/scalers.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Update config
        if 'config' in checkpoint:
            for k, v in checkpoint['config'].items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        self.vol_threshold = checkpoint.get('vol_threshold', 0.005)

        # Load model
        self.model = SPHNet(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load scalers
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
                self.price_scaler = scalers['price_scaler']
                self.feature_scaler = scalers['feature_scaler']
        else:
            print("Warning: Scalers not found. Run save_scalers() first.")
            self.price_scaler = None
            self.feature_scaler = None

        # Feature columns
        self.price_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return',
                          'hl_range', 'oc_range', 'log_return_abs']
        self.eng_cols = [
            'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
            'vol_park_1h', 'vol_park_24h', 'vol_rs_1h', 'vol_rs_24h',
            'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
            'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
            'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
            'atr_14', 'atr_24', 'macd_hist',
            'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]

    def predict(self, df=None):
        """
        Predict volatility regime for next hour.

        Args:
            df: DataFrame with OHLCV data. If None, fetches from Binance.

        Returns:
            dict with prediction, probability, and metadata
        """
        # Fetch data if not provided
        if df is None:
            df = fetch_recent_data(hours=self.config.window_size + 50)

        # Engineer features
        features_df = engineer_features(df)
        features_df = features_df.dropna()

        if len(features_df) < self.config.window_size:
            raise ValueError(f"Need at least {self.config.window_size} rows, got {len(features_df)}")

        # Get last window
        window_df = features_df.iloc[-self.config.window_size:]

        # Extract and scale features
        price_cols = [c for c in self.price_cols if c in window_df.columns]
        eng_cols = [c for c in self.eng_cols if c in window_df.columns]

        prices = window_df[price_cols].values.astype(np.float32)
        features = window_df[eng_cols].values.astype(np.float32)

        if self.price_scaler:
            prices = self.price_scaler.transform(prices)
        if self.feature_scaler:
            features = self.feature_scaler.transform(features)

        # To tensors
        prices_t = torch.tensor(prices, dtype=torch.float32).unsqueeze(0).to(self.device)
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(prices_t, features_t)
            prob = torch.sigmoid(outputs['direction_pred']).cpu().numpy()[0, 0]

        prediction = "HIGH" if prob > 0.5 else "LOW"
        confidence = prob if prob > 0.5 else 1 - prob

        return {
            'prediction': prediction,
            'probability': float(prob),
            'confidence': float(confidence),
            'threshold': self.vol_threshold,
            'timestamp': datetime.now().isoformat(),
            'next_hour': (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
        }


def save_scalers(train_data_path="data/processed/features.csv",
                 output_path="checkpoints/scalers.pkl"):
    """Save scalers from training data for inference."""
    from sklearn.preprocessing import RobustScaler

    df = pd.read_csv(train_data_path)

    # Use same split as training
    n_test = 10 * 24
    n_trainval = len(df) - n_test
    n_val = int(n_trainval * 0.15)
    n_train = n_trainval - n_val

    train_df = df.iloc[:n_train]

    price_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return',
                  'hl_range', 'oc_range', 'log_return_abs']
    eng_cols = [
        'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
        'vol_park_1h', 'vol_park_24h', 'vol_rs_1h', 'vol_rs_24h',
        'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
        'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
        'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
        'atr_14', 'atr_24', 'macd_hist',
        'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]

    price_cols = [c for c in price_cols if c in train_df.columns]
    eng_cols = [c for c in eng_cols if c in train_df.columns]

    price_scaler = RobustScaler()
    feature_scaler = RobustScaler()

    price_scaler.fit(train_df[price_cols].values)
    feature_scaler.fit(train_df[eng_cols].values)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'price_scaler': price_scaler,
            'feature_scaler': feature_scaler
        }, f)

    print(f"Scalers saved to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "save_scalers":
        save_scalers()
    else:
        # Save scalers first if not exists
        if not os.path.exists("checkpoints/scalers.pkl"):
            print("Saving scalers...")
            save_scalers()

        # Run prediction
        print("\nFetching recent BTC/USDT data...")
        predictor = VolatilityPredictor()
        result = predictor.predict()

        print("\n" + "="*50)
        print("VOLATILITY REGIME PREDICTION")
        print("="*50)
        print(f"Next Hour:    {result['next_hour']}")
        print(f"Prediction:   {result['prediction']} volatility")
        print(f"Confidence:   {result['confidence']:.1%}")
        print(f"Probability:  {result['probability']:.3f}")
        print("="*50)
