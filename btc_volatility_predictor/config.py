"""SPH-Net Configuration for BTC Volatility Prediction"""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    window_size: int = 72           # 3 days of hourly data (72 hours)
    n_assets: int = 1               # Single asset (BTC/USDT)
    price_features: int = 9         # OHLCV + derived (will be set from data)
    engineered_features: int = 32   # Technical indicators (will be set from data)
    forecast_horizon: int = 1       # Predict 1 hour ahead

    # Model architecture
    d_model: int = 128              # Embedding dimension (increased for complexity)
    n_heads: int = 8                # Attention heads
    n_encoder_layers: int = 4       # Transformer layers (deeper)
    dropout: float = 0.15           # Slightly higher for regularization

    # Prediction heads
    use_uncertainty: bool = True    # Predict uncertainty for volatility

    # Training
    batch_size: int = 64
    learning_rate: float = 5e-4     # Lower LR for stability
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15              # Early stopping patience

    # Loss weights
    alpha_mse: float = 1.0          # Regression loss weight
    beta_ce: float = 0.3            # Classification loss weight (auxiliary)
    gamma_uncertainty: float = 0.1  # Uncertainty loss weight

    # Learning rate scheduler
    lr_scheduler: str = "cosine"    # "cosine" or "plateau"
    warmup_epochs: int = 5

    # Device
    device: str = "cuda"            # or "cpu"

    # Paths
    data_path: str = "data/processed/features.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def update_from_metadata(self, metadata: dict):
        """Update config from dataset metadata."""
        self.price_features = metadata.get('n_price_features', self.price_features)
        self.engineered_features = metadata.get('n_engineered_features', self.engineered_features)
        self.window_size = metadata.get('window_size', self.window_size)
