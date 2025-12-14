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
