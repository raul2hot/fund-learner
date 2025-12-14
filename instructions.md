# SPH-Net Hybrid Transformer: Hello World Implementation Guide

## Overview

This document provides step-by-step instructions for implementing a minimal "hello world" version of the SPH-Net Hybrid Transformer architecture for financial time series prediction. The goal is to create a working baseline that demonstrates all core components.

---

## Project Structure

```
sph_net/
├── data/
│   └── synthetic.py          # Synthetic data generator for testing
├── models/
│   ├── __init__.py
│   ├── encoders.py           # Temporal & Feature encoders
│   ├── attention.py          # Co-attention fusion layer
│   ├── heads.py              # Prediction heads (regression, classification)
│   └── sph_net.py            # Main SPH-Net model assembly
├── utils/
│   ├── __init__.py
│   └── data_loader.py        # Walk-forward data loader
├── train.py                  # Training loop
├── evaluate.py               # Evaluation metrics
├── config.py                 # Hyperparameters & defaults
├── requirements.txt
└── README.md
```

---

## Step 1: Setup & Dependencies

### requirements.txt
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### Install
```bash
pip install -r requirements.txt
```

---

## Step 2: Configuration (config.py)

Create a config file with sensible defaults for the hello world version:

```python
"""SPH-Net Configuration - Hello World Defaults"""

from dataclasses import dataclass

@dataclass
class Config:
    # Data
    window_size: int = 64           # Sequence length T
    n_assets: int = 1               # Single asset for hello world
    price_features: int = 5         # OHLCV
    engineered_features: int = 10   # Technical indicators
    forecast_horizon: int = 1       # Predict 1 step ahead
    
    # Model architecture
    d_model: int = 64               # Embedding dimension
    n_heads: int = 4                # Attention heads
    n_encoder_layers: int = 2       # Transformer layers
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10              # Early stopping
    
    # Loss weights
    alpha_mse: float = 1.0          # Regression loss weight
    beta_ce: float = 0.5            # Classification loss weight
    
    # Device
    device: str = "cuda"            # or "cpu"
```

---

## Step 3: Synthetic Data Generator (data/synthetic.py)

For hello world, generate synthetic price data with known patterns:

```python
"""Synthetic data generator for testing SPH-Net"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_synthetic_prices(n_samples: int = 10000, seed: int = 42):
    """
    Generate synthetic OHLCV data with trend + noise.
    Returns: prices [n_samples, 5], features [n_samples, 10]
    """
    np.random.seed(seed)
    
    # Base price with trend and mean reversion
    t = np.arange(n_samples)
    trend = 0.0001 * t
    cycles = 0.02 * np.sin(2 * np.pi * t / 252)  # Yearly cycle
    noise = np.random.randn(n_samples) * 0.01
    
    log_returns = trend + cycles + noise
    close = 100 * np.exp(np.cumsum(log_returns))
    
    # Generate OHLCV from close
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_price = np.roll(close, 1) * (1 + np.random.randn(n_samples) * 0.002)
    open_price[0] = close[0]
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    prices = np.stack([open_price, high, low, close, volume], axis=1)
    
    # Generate synthetic engineered features (e.g., mock indicators)
    features = np.random.randn(n_samples, 10) * 0.1
    # Add some signal correlated with future returns
    features[:, 0] = np.roll(log_returns, -1) + np.random.randn(n_samples) * 0.005
    
    return prices.astype(np.float32), features.astype(np.float32), log_returns.astype(np.float32)


class TimeSeriesDataset(Dataset):
    """Walk-forward time series dataset"""
    
    def __init__(self, prices, features, returns, window_size=64, horizon=1):
        self.prices = prices
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.horizon = horizon
        
        # Valid indices (ensure we have enough data for window + horizon)
        self.valid_indices = len(prices) - window_size - horizon
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Input window
        price_window = self.prices[idx:idx + self.window_size]
        feat_window = self.features[idx:idx + self.window_size]
        
        # Target: next return(s) and direction
        target_idx = idx + self.window_size
        target_return = self.returns[target_idx:target_idx + self.horizon]
        target_direction = (target_return > 0).astype(np.float32)
        
        return {
            'prices': torch.tensor(price_window),
            'features': torch.tensor(feat_window),
            'target_return': torch.tensor(target_return),
            'target_direction': torch.tensor(target_direction)
        }


def create_dataloaders(config, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders with walk-forward split"""
    
    prices, features, returns = generate_synthetic_prices()
    n = len(prices)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_dataset = TimeSeriesDataset(
        prices[:train_end], features[:train_end], returns[:train_end],
        config.window_size, config.forecast_horizon
    )
    val_dataset = TimeSeriesDataset(
        prices[train_end:val_end], features[train_end:val_end], returns[train_end:val_end],
        config.window_size, config.forecast_horizon
    )
    test_dataset = TimeSeriesDataset(
        prices[val_end:], features[val_end:], returns[val_end:],
        config.window_size, config.forecast_horizon
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
```

---

## Step 4: Model Components (models/)

### 4.1 Encoders (models/encoders.py)

```python
"""Temporal and Feature Encoders for SPH-Net"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Transformer-based encoder for price/return sequences.
    Input: [batch, T, P_price]
    Output: [batch, T, d_model]
    """
    
    def __init__(self, input_dim: int, d_model: int, n_heads: int, 
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        
        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [batch, T, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x  # [batch, T, d_model]


class FeatureEncoder(nn.Module):
    """
    MLP-based encoder for engineered features.
    Input: [batch, T, P_feat]
    Output: [batch, T, d_model]
    """
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x):
        # x: [batch, T, input_dim]
        return self.encoder(x)  # [batch, T, d_model]
```

### 4.2 Co-Attention (models/attention.py)

```python
"""Co-Attention Fusion Layer for SPH-Net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoAttentionFusion(nn.Module):
    """
    Bi-directional co-attention between temporal and feature tokens.
    
    Temporal tokens attend to feature tokens (and vice versa).
    Returns fused representation combining both streams.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Temporal -> Feature attention
        self.q_temporal = nn.Linear(d_model, d_model)
        self.k_feature = nn.Linear(d_model, d_model)
        self.v_feature = nn.Linear(d_model, d_model)
        
        # Feature -> Temporal attention
        self.q_feature = nn.Linear(d_model, d_model)
        self.k_temporal = nn.Linear(d_model, d_model)
        self.v_temporal = nn.Linear(d_model, d_model)
        
        # Output projections
        self.out_temporal = nn.Linear(d_model, d_model)
        self.out_feature = nn.Linear(d_model, d_model)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
    
    def _attention(self, Q, K, V):
        """Scaled dot-product attention"""
        # Q, K, V: [batch, n_heads, seq_len, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)
    
    def _reshape_for_attention(self, x, proj):
        """Project and reshape for multi-head attention"""
        batch, seq_len, _ = x.shape
        x = proj(x)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
    
    def forward(self, temporal_tokens, feature_tokens):
        """
        Args:
            temporal_tokens: [batch, T, d_model]
            feature_tokens: [batch, T, d_model]
        Returns:
            fused: [batch, T, d_model]
        """
        batch = temporal_tokens.shape[0]
        
        # Temporal attends to features
        Q_t = self._reshape_for_attention(temporal_tokens, self.q_temporal)
        K_f = self._reshape_for_attention(feature_tokens, self.k_feature)
        V_f = self._reshape_for_attention(feature_tokens, self.v_feature)
        
        attn_t = self._attention(Q_t, K_f, V_f)
        attn_t = attn_t.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        attn_t = self.out_temporal(attn_t)
        fused_temporal = self.norm_t(temporal_tokens + attn_t)
        
        # Features attend to temporal
        Q_f = self._reshape_for_attention(feature_tokens, self.q_feature)
        K_t = self._reshape_for_attention(temporal_tokens, self.k_temporal)
        V_t = self._reshape_for_attention(temporal_tokens, self.v_temporal)
        
        attn_f = self._attention(Q_f, K_t, V_t)
        attn_f = attn_f.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        attn_f = self.out_feature(attn_f)
        fused_features = self.norm_f(feature_tokens + attn_f)
        
        # Combine both streams
        combined = torch.cat([fused_temporal, fused_features], dim=-1)
        fused = self.fusion(combined)
        
        return fused
```

### 4.3 Prediction Heads (models/heads.py)

```python
"""Prediction heads for SPH-Net"""

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """Predicts continuous return values"""
    
    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )
    
    def forward(self, x):
        # x: [batch, d_model] (last token or pooled)
        return self.head(x)


class ClassificationHead(nn.Module):
    """Predicts direction (up/down)"""
    
    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)  # Binary classification per horizon
        )
    
    def forward(self, x):
        # x: [batch, d_model]
        return self.head(x)  # Logits, apply sigmoid for probabilities


class UncertaintyHead(nn.Module):
    """Predicts aleatoric uncertainty (variance)"""
    
    def __init__(self, d_model: int, horizon: int = 1, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x):
        return self.head(x)
```

### 4.4 Main Model (models/sph_net.py)

```python
"""SPH-Net: Hybrid Transformer for Financial Time Series"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead


class SPHNet(nn.Module):
    """
    SPH-Net Hybrid Transformer
    
    Architecture:
    1. Temporal Encoder (Transformer) for price/returns
    2. Feature Encoder (MLP) for engineered features
    3. Co-Attention Fusion
    4. Prediction Heads (regression, classification, uncertainty)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )
        
        self.feature_encoder = FeatureEncoder(
            input_dim=config.engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Optional: Additional transformer block after fusion
        self.decoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Prediction heads
        self.regression_head = RegressionHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.classification_head = ClassificationHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.uncertainty_head = UncertaintyHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
    
    def forward(self, prices, features):
        """
        Args:
            prices: [batch, T, P_price] - OHLCV or returns
            features: [batch, T, P_feat] - Engineered features
        
        Returns:
            dict with 'return_pred', 'direction_pred', 'uncertainty'
        """
        # Encode
        temporal_tokens = self.temporal_encoder(prices)      # [batch, T, d_model]
        feature_tokens = self.feature_encoder(features)      # [batch, T, d_model]
        
        # Fuse with co-attention
        fused = self.co_attention(temporal_tokens, feature_tokens)  # [batch, T, d_model]
        
        # Decode
        decoded = self.decoder(fused)  # [batch, T, d_model]
        
        # Use last token for prediction
        last_token = decoded[:, -1, :]  # [batch, d_model]
        
        # Predictions
        return_pred = self.regression_head(last_token)
        direction_pred = self.classification_head(last_token)
        uncertainty = self.uncertainty_head(last_token)
        
        return {
            'return_pred': return_pred,
            'direction_pred': direction_pred,
            'uncertainty': uncertainty
        }
```

### 4.5 Models __init__.py

```python
"""SPH-Net Models"""

from .sph_net import SPHNet
from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead

__all__ = [
    'SPHNet',
    'TemporalEncoder',
    'FeatureEncoder', 
    'CoAttentionFusion',
    'RegressionHead',
    'ClassificationHead',
    'UncertaintyHead'
]
```

---

## Step 5: Training Loop (train.py)

```python
"""Training script for SPH-Net"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from config import Config
from models import SPHNet
from data.synthetic import create_dataloaders


def train_epoch(model, dataloader, optimizer, criterion_mse, criterion_bce, config, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_return = batch['target_return'].to(device)
        target_direction = batch['target_direction'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(prices, features)
        
        # Compute losses
        loss_mse = criterion_mse(outputs['return_pred'], target_return)
        loss_bce = criterion_bce(outputs['direction_pred'], target_direction)
        
        # Combined loss
        loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, criterion_mse, criterion_bce, config, device):
    """Validation pass"""
    model.eval()
    total_loss = 0
    total_mse = 0
    correct_direction = 0
    total_samples = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            prices = batch['prices'].to(device)
            features = batch['features'].to(device)
            target_return = batch['target_return'].to(device)
            target_direction = batch['target_direction'].to(device)
            
            outputs = model(prices, features)
            
            loss_mse = criterion_mse(outputs['return_pred'], target_return)
            loss_bce = criterion_bce(outputs['direction_pred'], target_direction)
            loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            
            # Direction accuracy
            pred_dir = (torch.sigmoid(outputs['direction_pred']) > 0.5).float()
            correct_direction += (pred_dir == target_direction).sum().item()
            total_samples += target_direction.numel()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'direction_acc': correct_direction / total_samples
    }


def main():
    # Configuration
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    model = SPHNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion_mse = nn.HuberLoss()  # More robust than MSE
    criterion_bce = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_bce, config, device)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion_mse, criterion_bce, config, device)
        print(f"Val Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}, Dir Acc: {val_metrics['direction_acc']:.4f}")
        
        scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = validate(model, test_loader, criterion_mse, criterion_bce, config, device)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test Direction Accuracy: {test_metrics['direction_acc']:.4f}")


if __name__ == "__main__":
    main()
```

---

## Step 6: Run Hello World

```bash
# Create project structure
mkdir -p sph_net/data sph_net/models sph_net/utils

# Copy files to appropriate locations (as shown in project structure)

# Run training
cd sph_net
python train.py
```

---

## Success Criteria

The hello world implementation is successful when:

1. **Model compiles**: No errors when instantiating `SPHNet(config)`
2. **Forward pass works**: `model(prices, features)` returns valid outputs
3. **Training runs**: Loss decreases over epochs
4. **Validation works**: Direction accuracy > 50% (better than random)
5. **No data leakage**: Walk-forward split is properly implemented

---

## Expected Output

```
Using device: cuda
Train batches: 109, Val batches: 23, Test batches: 23
Model parameters: 298,561

Epoch 1/50
Training: 100%|██████████| 109/109 [00:02<00:00, 45.32it/s]
Train Loss: 0.012345
Val Loss: 0.010234, MSE: 0.008765, Dir Acc: 0.5234
✓ Saved best model

...

Final Test Evaluation
==================================================
Test Loss: 0.009876
Test MSE: 0.007654
Test Direction Accuracy: 0.5512
```

---

## Next Steps After Hello World

1. **Real Data**: Replace synthetic data with actual market data
2. **More Features**: Add technical indicators (RSI, MACD, Bollinger Bands)
3. **Cross-Asset**: Implement GNN encoder for multi-asset relationships
4. **GA Tuning**: Add genetic algorithm for hyperparameter optimization
5. **Backtesting**: Integrate with a backtesting framework (backtrader, zipline)
6. **Production**: Export to ONNX, add monitoring

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size or d_model |
| Loss not decreasing | Lower learning rate, check data normalization |
| Direction accuracy ~50% | Model not learning; add more signal to features |
| NaN losses | Add gradient clipping, check for data issues |

---

## Reference

This implementation follows the SPH-Net hybrid transformer architecture:
- Temporal Transformer encoder for sequential price data
- Feature MLP encoder for engineered indicators  
- Bi-directional co-attention fusion
- Multi-task prediction heads (regression + classification + uncertainty)