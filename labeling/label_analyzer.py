"""
Label Distribution Analysis and Visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LabelAnalyzer:
    """Analyzes label quality and distribution."""

    def __init__(self, df: pd.DataFrame):
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        self.df = df

    def compute_statistics(self) -> Dict:
        """Comprehensive statistics about labels."""
        valid = self.df[self.df['label'].notna()].copy()

        stats = {
            'total_rows': len(self.df),
            'valid_labels': len(valid),
            'nan_labels': self.df['label'].isna().sum(),
        }

        # Distribution
        for label in range(5):
            count = (valid['label'] == label).sum()
            stats[f'class_{label}_count'] = count
            stats[f'class_{label}_pct'] = round(count / len(valid) * 100, 2) if len(valid) > 0 else 0

        # Return statistics per class
        if 'next_return' in valid.columns:
            for label in range(5):
                class_data = valid[valid['label'] == label]
                if len(class_data) > 0:
                    stats[f'class_{label}_mean_return'] = round(
                        class_data['next_return'].mean() * 100, 4
                    )
                    stats[f'class_{label}_std_return'] = round(
                        class_data['next_return'].std() * 100, 4
                    )

        # MAE statistics for tradeable classes
        if 'next_mae_long' in valid.columns and 'next_mae_short' in valid.columns:
            for label, col in [(0, 'next_mae_long'), (4, 'next_mae_short')]:
                class_data = valid[valid['label'] == label]
                if len(class_data) > 0:
                    stats[f'class_{label}_mean_mae'] = round(
                        class_data[col].mean() * 100, 4
                    )

        return stats

    def check_class_balance(self) -> Tuple[bool, str]:
        """
        Check if class distribution is reasonable.

        Expected distribution (roughly):
        - HIGH_BULL: 5-10%
        - BULL: 20-25%
        - RANGE_BOUND: 35-45%
        - BEAR: 20-25%
        - LOW_BEAR: 5-10%
        """
        valid = self.df[self.df['label'].notna()]
        total = len(valid)

        if total == 0:
            return False, "No valid labels found"

        warnings = []

        for label in range(5):
            pct = (valid['label'] == label).sum() / total * 100

            if label in [0, 4]:  # HIGH_BULL, LOW_BEAR
                if pct < 2:
                    warnings.append(f"Class {label} severely underrepresented ({pct:.1f}%)")
                elif pct > 15:
                    warnings.append(f"Class {label} overrepresented ({pct:.1f}%)")
            elif label == 2:  # RANGE_BOUND
                if pct < 25:
                    warnings.append(f"RANGE_BOUND too low ({pct:.1f}%)")
                elif pct > 60:
                    warnings.append(f"RANGE_BOUND too high ({pct:.1f}%)")

        if warnings:
            return False, "; ".join(warnings)
        return True, "Distribution looks reasonable"

    def validate_label_correctness(self) -> Dict:
        """
        Validate that labels are correctly assigned.
        Spot-check that thresholds are applied correctly.
        """
        valid = self.df[self.df['label'].notna()].copy()

        if 'next_return' not in valid.columns or 'next_mae_long' not in valid.columns:
            return {'all_valid': True, 'note': 'Could not validate - missing columns'}

        # HIGH_BULL should have: return >= 1.5% AND mae_long < 0.5%
        high_bull = valid[valid['label'] == 0]
        if len(high_bull) > 0:
            correct_return = (high_bull['next_return'] >= 0.015).all()
            correct_mae = (high_bull['next_mae_long'] < 0.005).all()
        else:
            correct_return, correct_mae = True, True

        # LOW_BEAR should have: return <= -1.5% AND mae_short < 0.5%
        low_bear = valid[valid['label'] == 4]
        if len(low_bear) > 0:
            correct_return_bear = (low_bear['next_return'] <= -0.015).all()
            correct_mae_bear = (low_bear['next_mae_short'] < 0.005).all()
        else:
            correct_return_bear, correct_mae_bear = True, True

        return {
            'high_bull_return_check': correct_return,
            'high_bull_mae_check': correct_mae,
            'low_bear_return_check': correct_return_bear,
            'low_bear_mae_check': correct_mae_bear,
            'all_valid': all([
                correct_return, correct_mae,
                correct_return_bear, correct_mae_bear
            ])
        }
