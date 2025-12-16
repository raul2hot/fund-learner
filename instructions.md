# Dynamic Trade Threshold Implementation Guide
**Project:** BTC/USDT Two-Stage Trading Classifier  
**Task:** Replace static threshold (0.55) with adaptive threshold based on market regime  
**Priority:** High - This is the key improvement needed to pass walk-forward validation

---

## Executive Summary

The model currently uses a fixed `trade_threshold = 0.55` which works well in trending markets but **over-trades in choppy/uncertain conditions** like May 2021. We need to implement an adaptive threshold that tightens (increases) during uncertain market regimes.

**Target Outcome:**
- May 2021 period: Reduce loss from -6.10% to breakeven or profitable
- Other periods: Maintain profitability (no significant degradation)
- Overall Sharpe: Maintain >1.5

---

## Phase 1: Analysis (Do This First)

### Step 1.1: Load and Compare Predictions

Create a new script `scripts/analyze_dynamic_threshold.py`:

```python
#!/usr/bin/env python
"""
Analyze predictions across walk-forward periods to understand
why May 2021 failed and what threshold would have helped.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json

# Load all period predictions
PERIODS = {
    'covid': 'experiments/walk_forward/period_0_covid/predictions.csv',
    'may2021': 'experiments/walk_forward/period_1_may2021/predictions.csv',
    'luna': 'experiments/walk_forward/period_2_luna/predictions.csv',
    'ftx': 'experiments/walk_forward/period_3_ftx/predictions.csv',
    'etf': 'experiments/walk_forward/period_4_etf/predictions.csv',
}

def load_predictions():
    """Load all prediction files."""
    data = {}
    for name, path in PERIODS.items():
        if Path(path).exists():
            data[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(data[name])} samples")
        else:
            print(f"Missing: {path}")
    return data

def analyze_trade_prob_distribution(data):
    """Compare trade_prob distributions across periods."""
    print("\n" + "="*70)
    print("TRADE PROBABILITY DISTRIBUTION BY PERIOD")
    print("="*70)
    
    for name, df in data.items():
        trades = df[df['should_trade'] == True]
        print(f"\n{name.upper()}:")
        print(f"  Total samples: {len(df)}")
        print(f"  Trades taken: {len(trades)} ({len(trades)/len(df)*100:.1f}%)")
        if len(trades) > 0:
            print(f"  trade_prob stats (for taken trades):")
            print(f"    mean: {trades['trade_prob'].mean():.3f}")
            print(f"    std:  {trades['trade_prob'].std():.3f}")
            print(f"    min:  {trades['trade_prob'].min():.3f}")
            print(f"    25%:  {trades['trade_prob'].quantile(0.25):.3f}")
            print(f"    50%:  {trades['trade_prob'].quantile(0.50):.3f}")
            print(f"    75%:  {trades['trade_prob'].quantile(0.75):.3f}")
            print(f"    max:  {trades['trade_prob'].max():.3f}")

def threshold_sweep(df, period_name):
    """Sweep through thresholds and find optimal for a period."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP: {period_name.upper()}")
    print(f"{'='*70}")
    
    results = []
    for thresh in np.arange(0.50, 0.85, 0.05):
        trades = df[df['trade_prob'] >= thresh].copy()
        n_trades = len(trades)
        
        if n_trades == 0:
            results.append({
                'threshold': thresh,
                'n_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_return': 0,
            })
            continue
        
        # Calculate returns
        total_return = trades['trade_return'].sum() * 100
        win_rate = (trades['trade_return'] > 0).mean() * 100
        avg_return = trades['trade_return'].mean() * 100
        
        # Sharpe approximation
        if trades['trade_return'].std() > 0:
            sharpe = (trades['trade_return'].mean() / trades['trade_return'].std()) * np.sqrt(len(trades) * 4)
        else:
            sharpe = 0
        
        results.append({
            'threshold': thresh,
            'n_trades': n_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{'Thresh':>7} {'Trades':>8} {'TotRet%':>10} {'WinRate%':>10} {'AvgRet%':>10} {'Sharpe':>8}")
    print("-"*60)
    for _, row in results_df.iterrows():
        print(f"{row['threshold']:>7.2f} {row['n_trades']:>8} {row['total_return']:>10.2f} "
              f"{row['win_rate']:>10.1f} {row['avg_return']:>10.4f} {row.get('sharpe', 0):>8.2f}")
    
    # Find break-even threshold
    profitable = results_df[results_df['total_return'] > 0]
    if len(profitable) > 0:
        min_profitable_thresh = profitable['threshold'].min()
        print(f"\nMinimum threshold for profitability: {min_profitable_thresh:.2f}")
    else:
        print(f"\nNo threshold makes this period profitable")
    
    return results_df

def analyze_low_confidence_trades(df, period_name, threshold=0.55):
    """Analyze trades that just barely passed the threshold."""
    print(f"\n{'='*70}")
    print(f"LOW CONFIDENCE TRADE ANALYSIS: {period_name.upper()}")
    print(f"{'='*70}")
    
    trades = df[df['should_trade'] == True].copy()
    
    # Split into confidence buckets
    buckets = [
        (0.55, 0.60, 'Marginal (0.55-0.60)'),
        (0.60, 0.65, 'Medium (0.60-0.65)'),
        (0.65, 0.70, 'High (0.65-0.70)'),
        (0.70, 1.00, 'Very High (0.70+)'),
    ]
    
    print(f"\n{'Bucket':<25} {'Count':>8} {'TotRet%':>10} {'WinRate%':>10} {'AvgRet%':>10}")
    print("-"*70)
    
    for low, high, name in buckets:
        subset = trades[(trades['trade_prob'] >= low) & (trades['trade_prob'] < high)]
        if len(subset) > 0:
            total_ret = subset['trade_return'].sum() * 100
            win_rate = (subset['trade_return'] > 0).mean() * 100
            avg_ret = subset['trade_return'].mean() * 100
            print(f"{name:<25} {len(subset):>8} {total_ret:>10.2f} {win_rate:>10.1f} {avg_ret:>10.4f}")
        else:
            print(f"{name:<25} {0:>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

def main():
    data = load_predictions()
    
    if not data:
        print("No prediction files found. Run walk_forward_validation.py first.")
        return
    
    # 1. Compare distributions
    analyze_trade_prob_distribution(data)
    
    # 2. Threshold sweep for each period
    all_sweeps = {}
    for name, df in data.items():
        all_sweeps[name] = threshold_sweep(df, name)
    
    # 3. Analyze low confidence trades (especially for May 2021)
    if 'may2021' in data:
        analyze_low_confidence_trades(data['may2021'], 'may2021')
    
    # 4. Find optimal global threshold
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLD ACROSS ALL PERIODS")
    print("="*70)
    
    for thresh in np.arange(0.55, 0.75, 0.05):
        total_return = 0
        n_periods_profitable = 0
        
        for name, df in data.items():
            if name == 'covid':  # Skip non-primary
                continue
            trades = df[df['trade_prob'] >= thresh]
            if len(trades) > 0:
                period_return = trades['trade_return'].sum() * 100
                total_return += period_return
                if period_return > 0:
                    n_periods_profitable += 1
        
        print(f"Threshold {thresh:.2f}: {n_periods_profitable}/4 profitable, Total: {total_return:+.2f}%")
    
    # Save analysis results
    output_dir = Path('experiments/walk_forward/analysis')
    output_dir.mkdir(exist_ok=True)
    
    for name, sweep_df in all_sweeps.items():
        sweep_df.to_csv(output_dir / f'{name}_threshold_sweep.csv', index=False)
    
    print(f"\nSaved sweep results to {output_dir}")

if __name__ == "__main__":
    main()
```

### Step 1.2: Run the Analysis

```bash
cd /path/to/project
python scripts/analyze_dynamic_threshold.py
```

**Expected outputs:**
1. Which threshold makes May 2021 profitable (or minimizes loss)
2. How trade_prob distribution differs between May 2021 and successful periods
3. Whether low-confidence (0.55-0.60) trades are consistently losers in May 2021

---

## Phase 2: Identify Regime Indicators

### Step 2.1: Analyze Features That Predict "Choppiness"

The codebase already computes regime features in `features/technical_indicators.py`:
- `vol_percentile` - Where current volatility ranks historically
- `vol_ratio` - Short-term vs long-term volatility ratio  
- `trend_efficiency` - Net movement / total path (1.0 = perfect trend, 0 = choppy)
- `bb_width_percentile` - Bollinger Band width percentile

Create `scripts/analyze_regime_features.py`:

```python
#!/usr/bin/env python
"""
Analyze which regime features could predict May 2021's choppiness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Load full dataset with features
DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

def load_featured_data():
    """Load data and compute regime features."""
    from features.feature_pipeline import FeaturePipeline
    
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    pipeline = FeaturePipeline()
    df = pipeline.compute_all_features(df)
    
    return df

def analyze_regime_by_period(df):
    """Compare regime features across different periods."""
    
    periods = {
        'may2021': ('2021-05-01', '2021-07-31'),
        'luna': ('2022-05-01', '2022-07-31'),
        'ftx': ('2022-11-01', '2023-01-31'),
        'etf': ('2024-01-01', '2024-03-31'),
    }
    
    regime_cols = [
        'vol_percentile', 'vol_ratio', 'trend_efficiency', 
        'trend_efficiency_72h', 'bb_width_percentile', 'atr_percentile'
    ]
    
    print("\n" + "="*80)
    print("REGIME FEATURES BY PERIOD")
    print("="*80)
    
    results = {}
    for period_name, (start, end) in periods.items():
        mask = (df['timestamp'] >= start) & (df['timestamp'] < end)
        period_df = df[mask]
        
        if len(period_df) == 0:
            print(f"\n{period_name}: No data")
            continue
        
        print(f"\n{period_name.upper()} ({start} to {end}):")
        print(f"  Samples: {len(period_df)}")
        
        results[period_name] = {}
        for col in regime_cols:
            if col in period_df.columns:
                mean_val = period_df[col].mean()
                std_val = period_df[col].std()
                results[period_name][col] = mean_val
                print(f"  {col:25s}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Compute difference from May 2021 to others
    print("\n" + "="*80)
    print("MAY 2021 vs OTHER PERIODS (difference in means)")
    print("="*80)
    
    if 'may2021' in results:
        may21 = results['may2021']
        for period_name, metrics in results.items():
            if period_name == 'may2021':
                continue
            print(f"\n{period_name}:")
            for col in regime_cols:
                if col in may21 and col in metrics:
                    diff = may21[col] - metrics[col]
                    print(f"  {col:25s}: {diff:+.3f}")
    
    return results

def find_discriminating_features(df):
    """Find features that separate profitable from unprofitable periods."""
    
    # Define periods and their outcomes
    periods = {
        'may2021': {'start': '2021-05-01', 'end': '2021-07-31', 'profitable': False},
        'luna': {'start': '2022-05-01', 'end': '2022-07-31', 'profitable': True},
        'ftx': {'start': '2022-11-01', 'end': '2023-01-31', 'profitable': True},
        'etf': {'start': '2024-01-01', 'end': '2024-03-31', 'profitable': True},
    }
    
    regime_cols = [
        'vol_percentile', 'vol_ratio', 'trend_efficiency', 
        'trend_efficiency_72h', 'bb_width_percentile'
    ]
    
    profitable_data = []
    unprofitable_data = []
    
    for period_name, info in periods.items():
        mask = (df['timestamp'] >= info['start']) & (df['timestamp'] < info['end'])
        period_df = df[mask]
        
        for col in regime_cols:
            if col in period_df.columns:
                if info['profitable']:
                    profitable_data.extend(period_df[col].dropna().tolist())
                else:
                    unprofitable_data.extend(period_df[col].dropna().tolist())
    
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION: PROFITABLE vs UNPROFITABLE PERIODS")
    print("="*80)
    
    # This is simplified - in practice you'd want per-feature analysis
    # The key insight: lower trend_efficiency likely indicates choppy market

def suggest_threshold_adjustment(df):
    """
    Suggest how to adjust threshold based on regime features.
    
    Key hypothesis: When trend_efficiency is low, the market is choppy
    and we should require higher confidence to trade.
    """
    
    print("\n" + "="*80)
    print("THRESHOLD ADJUSTMENT RULES")
    print("="*80)
    
    print("""
    Based on analysis, consider these rules:
    
    1. TREND EFFICIENCY BASED:
       - trend_efficiency > 0.5: Normal threshold (0.55)
       - trend_efficiency 0.3-0.5: Elevated threshold (0.60)
       - trend_efficiency < 0.3: High threshold (0.65-0.70)
    
    2. VOLATILITY RATIO BASED:
       - vol_ratio < 1.2: Normal threshold (0.55)
       - vol_ratio 1.2-1.5: Elevated threshold (0.60)
       - vol_ratio > 1.5: High threshold (0.65)
    
    3. ROLLING TRADE FREQUENCY BASED:
       - If >20 trades in last 24h: Increase threshold by 0.05
       - This is self-adaptive to model confidence patterns
    
    4. COMBINED (Recommended):
       base_threshold = 0.55
       if trend_efficiency_24h < 0.3:
           threshold += 0.05
       if vol_ratio > 1.3:
           threshold += 0.05
       threshold = min(threshold, 0.70)  # Cap at 0.70
    """)

def main():
    print("Loading and processing data...")
    df = load_featured_data()
    
    analyze_regime_by_period(df)
    suggest_threshold_adjustment(df)
    
    print("\n" + "="*80)
    print("NEXT STEP: Implement adaptive threshold in CalibratedTwoStageModel")
    print("="*80)

if __name__ == "__main__":
    main()
```

---

## Phase 3: Implementation

### Step 3.1: Modify CalibratedTwoStageModel

Edit `sph_net/models/two_stage.py`. Add adaptive threshold logic to `CalibratedTwoStageModel`:

```python
class CalibratedTwoStageModel(nn.Module):
    """
    Wrapper for TwoStageModel with ADAPTIVE thresholds and risk management.
    
    NEW: Dynamic threshold based on regime features.
    """

    # Regime detection thresholds
    TREND_EFFICIENCY_LOW = 0.3      # Below this = choppy market
    TREND_EFFICIENCY_MED = 0.5      # Below this = somewhat choppy
    VOL_RATIO_HIGH = 1.3            # Above this = elevated volatility
    
    # Threshold adjustments
    BASE_THRESHOLD = 0.55
    CHOPPY_ADJUSTMENT = 0.05        # Add this when choppy
    HIGH_VOL_ADJUSTMENT = 0.05      # Add this when high vol ratio
    MAX_THRESHOLD = 0.70            # Never go above this
    
    def __init__(
        self,
        model: 'TwoStageModel',
        trade_threshold: float = 0.55,  # Base threshold (may be adjusted dynamically)
        direction_threshold: float = 0.5,
        max_position: float = 1.0,
        use_position_sizing: bool = False,
        filter_high_volatility: bool = True,
        vol_threshold: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        # NEW: Adaptive threshold settings
        use_adaptive_threshold: bool = False,
        trend_efficiency_col_idx: int = None,  # Index of trend_efficiency in features
        vol_ratio_col_idx: int = None,          # Index of vol_ratio in features
    ):
        super().__init__()
        self.model = model
        self.base_threshold = trade_threshold
        self.trade_threshold = trade_threshold
        self.direction_threshold = direction_threshold
        self.max_position = max_position
        self.use_position_sizing = use_position_sizing
        self.filter_high_volatility = filter_high_volatility
        self.vol_threshold = vol_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Adaptive threshold config
        self.use_adaptive_threshold = use_adaptive_threshold
        self.trend_efficiency_col_idx = trend_efficiency_col_idx
        self.vol_ratio_col_idx = vol_ratio_col_idx
    
    def compute_adaptive_threshold(
        self,
        features: torch.Tensor,
        trend_efficiency: torch.Tensor = None,
        vol_ratio: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute per-sample adaptive threshold based on regime features.
        
        Args:
            features: [batch, window_size, n_features] - full feature tensor
            trend_efficiency: [batch] - optional pre-computed values
            vol_ratio: [batch] - optional pre-computed values
        
        Returns:
            threshold: [batch] - adaptive threshold per sample
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Start with base threshold
        threshold = torch.full((batch_size,), self.base_threshold, device=device)
        
        if not self.use_adaptive_threshold:
            return threshold
        
        # Extract regime features from last timestep
        last_features = features[:, -1, :]  # [batch, n_features]
        
        # Get trend efficiency
        if trend_efficiency is None and self.trend_efficiency_col_idx is not None:
            trend_efficiency = last_features[:, self.trend_efficiency_col_idx]
        
        # Get vol ratio  
        if vol_ratio is None and self.vol_ratio_col_idx is not None:
            vol_ratio = last_features[:, self.vol_ratio_col_idx]
        
        # Apply adjustments
        if trend_efficiency is not None:
            # Low trend efficiency = choppy market = higher threshold
            choppy_mask = trend_efficiency < self.TREND_EFFICIENCY_LOW
            somewhat_choppy_mask = (trend_efficiency >= self.TREND_EFFICIENCY_LOW) & \
                                   (trend_efficiency < self.TREND_EFFICIENCY_MED)
            
            threshold = torch.where(choppy_mask, threshold + 0.10, threshold)  # Very choppy
            threshold = torch.where(somewhat_choppy_mask, threshold + 0.05, threshold)  # Somewhat choppy
        
        if vol_ratio is not None:
            # High vol ratio = elevated uncertainty = higher threshold
            high_vol_mask = vol_ratio > self.VOL_RATIO_HIGH
            threshold = torch.where(high_vol_mask, threshold + self.HIGH_VOL_ADJUSTMENT, threshold)
        
        # Cap at maximum
        threshold = torch.clamp(threshold, max=self.MAX_THRESHOLD)
        
        return threshold
    
    @torch.no_grad()
    def predict_with_sizing(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None,
        trend_efficiency: torch.Tensor = None,
        vol_ratio: torch.Tensor = None,
    ) -> dict:
        """
        Get predictions with ADAPTIVE thresholds and regime filtering.
        
        MODIFIED: Now uses per-sample adaptive threshold.
        """
        outputs = self.model(prices, features)
        
        # Get probabilities
        tradeable_probs = F.softmax(outputs['tradeable_logits'], dim=-1)
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)
        
        trade_prob = tradeable_probs[:, 1]
        long_prob = direction_probs[:, 0]
        short_prob = direction_probs[:, 1]
        
        # ADAPTIVE THRESHOLD
        threshold = self.compute_adaptive_threshold(
            features, trend_efficiency, vol_ratio
        )
        
        # Apply threshold (now per-sample)
        should_trade = trade_prob >= threshold
        
        # Volatility regime filtering (unchanged)
        regime_filtered = torch.zeros_like(should_trade)
        if self.filter_high_volatility and volatility is not None:
            if self.vol_threshold is None:
                vol_thresh = torch.quantile(volatility, self.VOL_HIGH_THRESHOLD)
            else:
                vol_thresh = self.vol_threshold
            
            is_high_vol = volatility > vol_thresh
            regime_filtered = is_high_vol & should_trade
            should_trade = should_trade & ~is_high_vol
        
        # Direction and position sizing (unchanged)
        is_long = long_prob > short_prob
        direction_confidence = torch.abs(long_prob - 0.5) * 2
        
        if self.use_position_sizing:
            scaled_prob = (trade_prob - threshold) / (1.0 - threshold)
            scaled_prob = scaled_prob.clamp(0, 1)
            position_size = scaled_prob * (0.7 + 0.3 * direction_confidence)
            position_size = position_size.clamp(0, self.max_position)
            position_size = torch.where(should_trade, position_size, torch.zeros_like(position_size))
        else:
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
            'adaptive_threshold': threshold,  # NEW: return the threshold used
        }
```

### Step 3.2: Update load_calibrated_model

Edit `scripts/run_calibrated.py`:

```python
def load_calibrated_model(
    model_path: Path,
    trade_threshold: float = 0.55,
    filter_high_volatility: bool = True,
    stop_loss_pct: float = -0.02,
    take_profit_pct: float = None,
    # NEW: Adaptive threshold params
    use_adaptive_threshold: bool = False,
    feature_info_path: Path = None,  # To get feature column indices
) -> CalibratedTwoStageModel:
    """Load trained model and wrap with calibration and risk management."""

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    model = TwoStageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Find regime feature indices if adaptive threshold enabled
    trend_efficiency_idx = None
    vol_ratio_idx = None
    
    if use_adaptive_threshold and feature_info_path and feature_info_path.exists():
        with open(feature_info_path) as f:
            feature_info = json.load(f)
        eng_cols = feature_info.get('engineered_columns', [])
        
        if 'trend_efficiency' in eng_cols:
            trend_efficiency_idx = eng_cols.index('trend_efficiency')
        if 'vol_ratio' in eng_cols:
            vol_ratio_idx = eng_cols.index('vol_ratio')
        
        print(f"Adaptive threshold enabled:")
        print(f"  trend_efficiency index: {trend_efficiency_idx}")
        print(f"  vol_ratio index: {vol_ratio_idx}")

    calibrated = CalibratedTwoStageModel(
        model,
        trade_threshold=trade_threshold,
        filter_high_volatility=filter_high_volatility,
        use_position_sizing=False,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_adaptive_threshold=use_adaptive_threshold,
        trend_efficiency_col_idx=trend_efficiency_idx,
        vol_ratio_col_idx=vol_ratio_idx,
    )

    return calibrated, config
```

### Step 3.3: Update Walk-Forward Validation

Edit `scripts/walk_forward_validation.py` to use adaptive threshold:

```python
# In INFERENCE_CONFIG, add:
INFERENCE_CONFIG = {
    'trade_threshold': 0.55,          # Base threshold
    'filter_high_volatility': True,
    'stop_loss_pct': -0.02,
    'use_adaptive_threshold': True,   # NEW: Enable adaptive threshold
}

# In evaluate_period(), update the CalibratedTwoStageModel instantiation:
def evaluate_period(
    model: TwoStageModel,
    test_loader: DataLoader,
    config: SPHNetConfig,
    trade_threshold: float = 0.55,
    filter_high_vol: bool = True,
    use_adaptive_threshold: bool = True,  # NEW
    feature_info: dict = None,             # NEW
) -> Dict:
    """Evaluate model on test period with adaptive threshold."""
    
    # Get feature indices
    trend_efficiency_idx = None
    vol_ratio_idx = None
    if feature_info:
        eng_cols = feature_info.get('engineered_columns', [])
        if 'trend_efficiency' in eng_cols:
            trend_efficiency_idx = eng_cols.index('trend_efficiency')
        if 'vol_ratio' in eng_cols:
            vol_ratio_idx = eng_cols.index('vol_ratio')
    
    calibrated_model = CalibratedTwoStageModel(
        model,
        trade_threshold=trade_threshold,
        filter_high_volatility=filter_high_vol,
        use_position_sizing=False,
        use_adaptive_threshold=use_adaptive_threshold,
        trend_efficiency_col_idx=trend_efficiency_idx,
        vol_ratio_col_idx=vol_ratio_idx,
    )
    # ... rest of function
```

---

## Phase 4: Validation

### Step 4.1: Create Comparison Script

Create `scripts/compare_threshold_strategies.py`:

```python
#!/usr/bin/env python
"""
Compare static vs adaptive threshold strategies across all walk-forward periods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from sph_net.models.two_stage import TwoStageModel, CalibratedTwoStageModel
from data.dataset import TradingDataset
# ... imports

def run_comparison():
    """Run both strategies on all periods and compare."""
    
    strategies = {
        'static_055': {'use_adaptive': False, 'base_threshold': 0.55},
        'static_060': {'use_adaptive': False, 'base_threshold': 0.60},
        'static_065': {'use_adaptive': False, 'base_threshold': 0.65},
        'adaptive': {'use_adaptive': True, 'base_threshold': 0.55},
    }
    
    periods = ['period_1_may2021', 'period_2_luna', 'period_3_ftx', 'period_4_etf']
    
    results = []
    
    for period_id in periods:
        for strategy_name, strategy_config in strategies.items():
            # Load model and run inference with this strategy
            metrics = evaluate_with_strategy(period_id, strategy_config)
            
            results.append({
                'period': period_id,
                'strategy': strategy_name,
                **metrics
            })
    
    results_df = pd.DataFrame(results)
    
    # Pivot for easy comparison
    pivot = results_df.pivot(index='period', columns='strategy', values='total_return')
    print("\nTOTAL RETURN BY PERIOD AND STRATEGY:")
    print(pivot.round(2))
    
    # Summary
    print("\n\nSUMMARY:")
    for strategy in strategies.keys():
        strategy_results = results_df[results_df['strategy'] == strategy]
        avg_return = strategy_results['total_return'].mean()
        n_profitable = (strategy_results['total_return'] > 0).sum()
        print(f"{strategy}: Avg Return={avg_return:+.2f}%, Profitable={n_profitable}/4")
    
    return results_df

if __name__ == "__main__":
    run_comparison()
```

### Step 4.2: Run Full Walk-Forward with Adaptive Threshold

```bash
# First, run analysis to understand the data
python scripts/analyze_dynamic_threshold.py

# Then implement changes and re-run walk-forward
python scripts/walk_forward_validation.py

# Compare strategies
python scripts/compare_threshold_strategies.py
```

---

## Phase 5: Tuning

### Key Parameters to Tune

1. **Base threshold** (currently 0.55)
   - Range: 0.50 - 0.60
   - Lower = more trades in trending markets

2. **Choppy adjustment** (currently +0.05 per condition)
   - Range: 0.03 - 0.10
   - Higher = more aggressive filtering in choppy markets

3. **Trend efficiency thresholds** (currently LOW=0.3, MED=0.5)
   - These define what counts as "choppy"
   - May need adjustment based on analysis

4. **Vol ratio threshold** (currently 1.3)
   - Above this = elevated volatility = increase threshold

### Tuning Strategy

```python
# Grid search over key parameters
param_grid = {
    'base_threshold': [0.50, 0.55, 0.60],
    'choppy_adjustment': [0.05, 0.10, 0.15],
    'trend_efficiency_low': [0.25, 0.30, 0.35],
    'vol_ratio_high': [1.2, 1.3, 1.5],
}

# For each combination:
# 1. Run walk-forward on May 2021 only (fast iteration)
# 2. Check if it improves (reduces loss or becomes profitable)
# 3. Then validate on other periods (ensure no degradation)
```

---

## Success Criteria

| Metric | Minimum Target | Ideal Target |
|--------|----------------|--------------|
| May 2021 Return | > -3% | > 0% |
| Other 3 Periods | All > 0% | All > 0% |
| Average Return | > +10% | > +14% |
| Average Sharpe | > 1.5 | > 2.0 |

---

## Files to Modify

1. **`sph_net/models/two_stage.py`**
   - Add `compute_adaptive_threshold()` method
   - Modify `predict_with_sizing()` to use adaptive threshold
   - Add new config parameters to `__init__`

2. **`scripts/run_calibrated.py`**
   - Update `load_calibrated_model()` to support adaptive threshold
   - Add feature index lookup

3. **`scripts/walk_forward_validation.py`**
   - Update `INFERENCE_CONFIG` with adaptive threshold flag
   - Update `evaluate_period()` to pass feature info

4. **New files to create:**
   - `scripts/analyze_dynamic_threshold.py` - Analysis script
   - `scripts/analyze_regime_features.py` - Regime analysis
   - `scripts/compare_threshold_strategies.py` - Comparison script

---

## Quick Start Commands

```bash
# 1. Analyze current predictions
python scripts/analyze_dynamic_threshold.py

# 2. Implement changes to two_stage.py (see Phase 3.1)

# 3. Test on May 2021 only (fast)
python -c "
from scripts.walk_forward_validation import *
# Quick test on just May 2021
"

# 4. Full validation
python scripts/walk_forward_validation.py

# 5. Review results
cat experiments/walk_forward/summary_report.json
```

---

## Debugging Tips

1. **If adaptive threshold makes things worse:**
   - Check that trend_efficiency_col_idx is correct
   - Print actual threshold values being used
   - May need to invert the logic (high trend_efficiency = choppy?)

2. **If May 2021 still fails:**
   - Try more aggressive threshold (0.65-0.70 fixed)
   - Check if the problem is direction accuracy, not trade frequency
   - Consider filtering by rolling win rate

3. **If other periods degrade:**
   - The adaptive threshold is too aggressive
   - Reduce choppy_adjustment or raise trend_efficiency thresholds
   - Consider using adaptive only in certain volatility regimes

---

## Summary

**Problem:** Static 0.55 threshold over-trades in choppy markets (May 2021)

**Solution:** Adaptive threshold that increases when:
- `trend_efficiency` is low (choppy market)
- `vol_ratio` is high (elevated uncertainty)

**Implementation:** Modify `CalibratedTwoStageModel.predict_with_sizing()` to compute per-sample threshold based on regime features.

**Validation:** Re-run walk-forward, ensure May 2021 improves without degrading other periods.