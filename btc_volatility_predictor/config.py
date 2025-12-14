"""SPH-Net Configuration for BTC Volatility Prediction"""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    window_size: int = 48           # 2 days - shorter is often better
    n_assets: int = 1               # Single asset (BTC/USDT)
    price_features: int = 9         # OHLCV + derived (will be set from data)
    engineered_features: int = 32   # Technical indicators (will be set from data)
    forecast_horizon: int = 1       # Predict 1 hour ahead

    # Model architecture - SIMPLER to avoid overfitting
    d_model: int = 64               # Smaller embedding
    n_heads: int = 4                # Fewer heads
    n_encoder_layers: int = 2       # Shallower
    dropout: float = 0.3            # More dropout

    # Prediction heads
    use_uncertainty: bool = False   # Disable for simpler model

    # Training
    batch_size: int = 32            # Smaller batches, more gradient noise
    learning_rate: float = 1e-3     # Higher LR with smaller model
    weight_decay: float = 1e-3      # Stronger regularization
    epochs: int = 150
    patience: int = 20              # More patience

    # Loss weights
    alpha_mse: float = 1.0          # Focus on regression
    beta_ce: float = 0.1            # Reduce classification weight
    gamma_uncertainty: float = 0.0  # Disabled

    # Learning rate scheduler
    lr_scheduler: str = "cosine"    # "cosine" or "plateau"
    warmup_epochs: int = 10         # Longer warmup

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
