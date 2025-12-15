"""
Alignment validation utilities.

CRITICAL: These validations must pass before using data for training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


def validate_no_lookahead(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that no look-ahead bias exists in the aligned dataset.

    Args:
        df: Aligned DataFrame to validate

    Returns:
        Dict with validation results
    """
    results = {
        'passed': True,
        'checks': [],
        'issues': [],
    }

    # Check 1: All age columns should be >= 0
    age_cols = [c for c in df.columns if '_age_hours' in c]
    for col in age_cols:
        min_age = df[col].min()
        if pd.notna(min_age) and min_age < 0:
            results['passed'] = False
            results['issues'].append({
                'check': 'negative_age',
                'column': col,
                'min_age': min_age,
                'description': f"Column {col} has negative age ({min_age:.2f}h) indicating future data"
            })
        else:
            results['checks'].append({
                'check': 'age_positive',
                'column': col,
                'status': 'PASS',
                'min_age': min_age if pd.notna(min_age) else None
            })

    # Check 2: Timestamps should be monotonically increasing
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        if not ts.is_monotonic_increasing:
            results['passed'] = False
            results['issues'].append({
                'check': 'monotonic_timestamps',
                'description': 'Timestamps are not monotonically increasing'
            })
        else:
            results['checks'].append({
                'check': 'monotonic_timestamps',
                'status': 'PASS'
            })

    return results


class AlignmentValidator:
    """
    Comprehensive validator for aligned datasets.
    """

    def __init__(self, aligned_df: pd.DataFrame):
        """
        Initialize validator.

        Args:
            aligned_df: Aligned DataFrame to validate
        """
        self.df = aligned_df

    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            Dict with all validation results
        """
        results = {
            'lookahead': self.check_lookahead(),
            'timestamps': self.check_timestamps(),
            'completeness': self.check_completeness(),
            'data_quality': self.check_data_quality(),
        }

        results['all_passed'] = all(
            r.get('passed', True) for r in results.values()
        )

        return results

    def check_lookahead(self) -> Dict[str, Any]:
        """Check for look-ahead bias."""
        return validate_no_lookahead(self.df)

    def check_timestamps(self) -> Dict[str, Any]:
        """Validate timestamp quality."""
        result = {
            'passed': True,
            'checks': [],
        }

        if 'timestamp' not in self.df.columns:
            result['passed'] = False
            result['checks'].append({
                'check': 'timestamp_exists',
                'status': 'FAIL',
                'description': 'No timestamp column found'
            })
            return result

        ts = self.df['timestamp']

        # Check for NaT
        nat_count = ts.isna().sum()
        if nat_count > 0:
            result['passed'] = False
            result['checks'].append({
                'check': 'no_nat',
                'status': 'FAIL',
                'nat_count': nat_count
            })
        else:
            result['checks'].append({
                'check': 'no_nat',
                'status': 'PASS'
            })

        # Check monotonicity
        if not ts.is_monotonic_increasing:
            result['checks'].append({
                'check': 'monotonic',
                'status': 'WARN',
                'description': 'Timestamps not strictly increasing'
            })
        else:
            result['checks'].append({
                'check': 'monotonic',
                'status': 'PASS'
            })

        # Check for gaps
        if len(ts) > 1:
            diffs = ts.diff().dropna()
            expected_diff = pd.Timedelta('1h')  # Assuming 1h frequency
            gaps = diffs[diffs > expected_diff * 2]
            if len(gaps) > 0:
                result['checks'].append({
                    'check': 'no_gaps',
                    'status': 'WARN',
                    'gap_count': len(gaps),
                    'max_gap_hours': gaps.max().total_seconds() / 3600
                })
            else:
                result['checks'].append({
                    'check': 'no_gaps',
                    'status': 'PASS'
                })

        return result

    def check_completeness(self) -> Dict[str, Any]:
        """Check data completeness."""
        result = {
            'passed': True,
            'columns': {},
        }

        for col in self.df.columns:
            if col == 'timestamp':
                continue

            missing_pct = self.df[col].isna().mean() * 100
            result['columns'][col] = {
                'missing_pct': round(missing_pct, 2),
                'status': 'OK' if missing_pct < 5 else 'WARN' if missing_pct < 20 else 'HIGH_MISSING'
            }

        return result

    def check_data_quality(self) -> Dict[str, Any]:
        """Check for data quality issues."""
        result = {
            'passed': True,
            'checks': [],
        }

        # Check for infinite values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                result['checks'].append({
                    'check': 'no_infinites',
                    'column': col,
                    'status': 'WARN',
                    'inf_count': inf_count
                })

        # Check for extreme values (potential data errors)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in self.df.columns:
                # Check for sudden large moves (>50% in one hour)
                pct_change = self.df[col].pct_change().abs()
                extreme = pct_change[pct_change > 0.5]
                if len(extreme) > 0:
                    result['checks'].append({
                        'check': 'no_extreme_moves',
                        'column': col,
                        'status': 'WARN',
                        'extreme_count': len(extreme)
                    })

        return result

    def print_report(self):
        """Print a human-readable validation report."""
        results = self.validate_all()

        print("\n" + "=" * 60)
        print("ALIGNMENT VALIDATION REPORT")
        print("=" * 60)

        for check_name, check_result in results.items():
            if check_name == 'all_passed':
                continue

            status = "PASS" if check_result.get('passed', True) else "FAIL"
            print(f"\n{check_name.upper()}: {status}")

            if 'checks' in check_result:
                for check in check_result['checks']:
                    check_status = check.get('status', 'UNKNOWN')
                    print(f"  - {check.get('check', 'unknown')}: {check_status}")
                    if check_status == 'FAIL':
                        print(f"    {check.get('description', '')}")

            if 'issues' in check_result and check_result['issues']:
                for issue in check_result['issues']:
                    print(f"  ISSUE: {issue.get('description', 'Unknown issue')}")

        print("\n" + "=" * 60)
        final_status = "ALL CHECKS PASSED" if results['all_passed'] else "SOME CHECKS FAILED"
        print(f"FINAL STATUS: {final_status}")
        print("=" * 60 + "\n")

        return results['all_passed']
