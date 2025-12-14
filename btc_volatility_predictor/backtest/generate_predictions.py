"""
Generate predictions for the entire test period.

This produces a DataFrame with:
- timestamp
- OHLCV data
- predicted regime (HIGH/LOW)
- actual regime (HIGH/LOW)
- prediction probability
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import SPHNet


# Feature columns (same as training)
PRICE_COLS = ['open', 'high', 'low', 'close', 'volume', 'log_return',
              'hl_range', 'oc_range', 'log_return_abs']

ENGINEERED_COLS = [
    'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h',
    'vol_rs_1h', 'vol_rs_24h',
    'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
    'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24', 'macd_hist',
    'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]


def load_model_and_threshold(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model and volatility threshold."""
    config = Config()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Update config from checkpoint
    if 'config' in checkpoint:
        for k, v in checkpoint['config'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    vol_threshold = checkpoint.get('vol_threshold', 0.005)

    # Create and load model
    model = SPHNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, vol_threshold


def prepare_scalers(train_df: pd.DataFrame) -> tuple:
    """Fit scalers on training data."""
    price_cols = [c for c in PRICE_COLS if c in train_df.columns]
    eng_cols = [c for c in ENGINEERED_COLS if c in train_df.columns]

    price_scaler = RobustScaler()
    feature_scaler = RobustScaler()

    price_scaler.fit(train_df[price_cols].values)
    feature_scaler.fit(train_df[eng_cols].values)

    return price_scaler, feature_scaler, price_cols, eng_cols


def generate_test_predictions(
    data_path: str = "data/processed/features.csv",
    checkpoint_path: str = "checkpoints/best_regime_model.pt",
    output_path: str = "backtest/results/test_predictions.csv",
    test_days: int = 10,
    val_ratio: float = 0.15
) -> pd.DataFrame:
    """
    Generate predictions for the entire test period.

    Args:
        data_path: Path to processed features CSV
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save predictions CSV
        test_days: Number of days in test set
        val_ratio: Validation ratio (to match training split)

    Returns:
        DataFrame with predictions for test period
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, config, vol_threshold = load_model_and_threshold(checkpoint_path, device)
    window_size = config.window_size

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Split data (same as training)
    n_total = len(df)
    n_test = test_days * 24
    n_trainval = n_total - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print(f"Train samples: {n_train}, Test samples: {len(test_df)}")
    print(f"Volatility threshold: {vol_threshold:.6f}")

    # Fit scalers on training data
    price_scaler, feature_scaler, price_cols, eng_cols = prepare_scalers(train_df)

    # Calculate actual regime based on target_volatility
    test_df = test_df.copy()
    test_df['actual_regime'] = (test_df['target_volatility'] > vol_threshold).astype(int)

    # Prepare storage
    predictions = []
    probabilities = []
    indices = []

    # Generate predictions for each sample in test set
    # We need window_size samples before each prediction
    test_start_idx = n_train + n_val

    print("Generating predictions...")
    with torch.no_grad():
        for i in tqdm(range(len(test_df))):
            global_idx = test_start_idx + i

            # Get window of data (from full dataframe to ensure we have history)
            window_start = global_idx - window_size
            if window_start < 0:
                # Not enough history, skip
                continue

            window_df = df.iloc[window_start:global_idx]

            if len(window_df) < window_size:
                continue

            # Extract and scale features
            prices = window_df[price_cols].values.astype(np.float32)
            features = window_df[eng_cols].values.astype(np.float32)

            prices_scaled = price_scaler.transform(prices)
            features_scaled = feature_scaler.transform(features)

            # To tensors
            prices_t = torch.tensor(prices_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            features_t = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict
            outputs = model(prices_t, features_t)
            prob = torch.sigmoid(outputs['direction_pred']).cpu().numpy()[0, 0]
            pred = 1 if prob > 0.5 else 0

            predictions.append(pred)
            probabilities.append(prob)
            indices.append(i)

    # Build results DataFrame
    results_df = test_df.iloc[indices].copy()
    results_df['predicted_regime'] = predictions
    results_df['prediction_prob'] = probabilities
    results_df['correct'] = (results_df['predicted_regime'] == results_df['actual_regime'])

    # Select output columns
    output_cols = ['open', 'high', 'low', 'close', 'volume',
                   'predicted_regime', 'actual_regime', 'prediction_prob', 'correct']

    # Add timestamp if available
    if 'timestamp' in results_df.columns:
        output_cols = ['timestamp'] + output_cols

    # Add useful indicators for strategies
    indicator_cols = ['rsi_14', 'rsi_6', 'bb_position', 'bb_bandwidth_20',
                      'atr_14', 'atr_24', 'macd_hist', 'volume_ma_ratio']
    for col in indicator_cols:
        if col in results_df.columns:
            output_cols.append(col)

    results_df = results_df[output_cols].reset_index(drop=True)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Print summary
    accuracy = results_df['correct'].mean()
    high_pct = results_df['predicted_regime'].mean()
    actual_high_pct = results_df['actual_regime'].mean()

    print(f"\n{'='*50}")
    print("PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(results_df)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Predicted HIGH: {high_pct:.1%}")
    print(f"Actual HIGH: {actual_high_pct:.1%}")
    print(f"Saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    generate_test_predictions()
