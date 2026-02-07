"""
Data Validation Script for Baseball Prediction Models

Validates downloaded Statcast data for:
- Required columns exist
- Missing value percentages
- Date ranges
- Reasonable value ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent.parent


# Required columns for each model (from data_dictionary.md)
CONTACT_MODEL_COLUMNS = [
    'plate_x', 'plate_z', 'zone',  # Location
    'pfx_x', 'pfx_z', 'release_speed',  # Movement
    'balls', 'strikes',  # Count
    'stand', 'p_throws',  # Handedness
    'description', 'pitch_type',  # Outcome and type
]

BATTED_BALL_COLUMNS = [
    'launch_speed', 'launch_angle',  # Primary features
    'hc_x', 'hc_y',  # Hit coordinates
    'events', 'bb_type',  # Outcome
    'stand', 'home_team',  # Context
]

# Expected value ranges (from data_dictionary.md)
VALUE_RANGES = {
    'plate_x': (-2.5, 2.5),
    'plate_z': (0.0, 5.0),
    'pfx_x': (-25, 25),
    'pfx_z': (-25, 25),
    'release_speed': (50, 105),
    'launch_speed': (30, 120),
    'launch_angle': (-90, 90),
    'balls': (0, 3),
    'strikes': (0, 2),
}


def check_missing_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Check missing data percentages for all columns.

    Args:
        df: DataFrame to check
        name: Name of dataset for reporting

    Returns:
        DataFrame with missing value statistics
    """
    print(f"\n{'='*50}")
    print(f"Missing Data Analysis: {name}")
    print(f"{'='*50}")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    stats = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct.round(2)
    }).sort_values('missing_pct', ascending=False)

    # Show columns with >0% missing
    missing_cols = stats[stats['missing_pct'] > 0]
    if len(missing_cols) > 0:
        print("\nColumns with missing values:")
        print(missing_cols.head(20).to_string())
    else:
        print("\nNo missing values found!")

    return stats


def check_required_columns(df: pd.DataFrame, required: list, name: str) -> bool:
    """
    Verify required columns exist in DataFrame.

    Args:
        df: DataFrame to check
        required: List of required column names
        name: Name of dataset for reporting

    Returns:
        True if all required columns present
    """
    print(f"\n{'='*50}")
    print(f"Required Columns Check: {name}")
    print(f"{'='*50}")

    missing_cols = [col for col in required if col not in df.columns]

    if missing_cols:
        print(f"MISSING COLUMNS: {missing_cols}")
        return False
    else:
        print(f"All {len(required)} required columns present")
        return True


def check_value_ranges(df: pd.DataFrame, name: str) -> dict:
    """
    Validate values are within expected ranges.

    Args:
        df: DataFrame to check
        name: Name of dataset for reporting

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*50}")
    print(f"Value Range Validation: {name}")
    print(f"{'='*50}")

    results = {}

    for col, (min_val, max_val) in VALUE_RANGES.items():
        if col in df.columns:
            col_data = df[col].dropna()
            actual_min = col_data.min()
            actual_max = col_data.max()
            out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()
            pct_out = (out_of_range / len(col_data)) * 100

            results[col] = {
                'expected_range': (min_val, max_val),
                'actual_range': (actual_min, actual_max),
                'out_of_range_count': out_of_range,
                'out_of_range_pct': pct_out
            }

            status = "OK" if pct_out < 1 else "WARNING"
            print(f"  {col}: [{actual_min:.2f}, {actual_max:.2f}] "
                  f"(expected [{min_val}, {max_val}]) - {status}")

    return results


def check_date_ranges(df: pd.DataFrame, name: str) -> tuple:
    """
    Validate date ranges in the dataset.

    Args:
        df: DataFrame to check
        name: Name of dataset for reporting

    Returns:
        Tuple of (min_date, max_date)
    """
    print(f"\n{'='*50}")
    print(f"Date Range Check: {name}")
    print(f"{'='*50}")

    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        min_date = df['game_date'].min()
        max_date = df['game_date'].max()
        print(f"  Date range: {min_date.date()} to {max_date.date()}")
        return min_date, max_date

    print("  game_date column not found")
    return None, None


def check_class_distribution(df: pd.DataFrame, name: str):
    """
    Check class distribution for target variables.

    Args:
        df: DataFrame to check
        name: Name of dataset for reporting
    """
    print(f"\n{'='*50}")
    print(f"Class Distribution: {name}")
    print(f"{'='*50}")

    # For contact model - description
    if 'description' in df.columns:
        print("\nPitch outcomes (description):")
        desc_counts = df['description'].value_counts()
        desc_pct = (desc_counts / len(df) * 100).round(2)
        for desc, count in desc_counts.head(10).items():
            print(f"  {desc}: {count:,} ({desc_pct[desc]:.2f}%)")

    # For hit outcome model - events
    if 'events' in df.columns:
        events_df = df[df['events'].notna()]
        if len(events_df) > 0:
            print("\nHit outcomes (events):")
            event_counts = events_df['events'].value_counts()
            event_pct = (event_counts / len(events_df) * 100).round(2)
            for event, count in event_counts.head(15).items():
                print(f"  {event}: {count:,} ({event_pct[event]:.2f}%)")


def validate_pitch_data(filepath: Path) -> dict:
    """
    Validate pitch data file.

    Args:
        filepath: Path to pitch data CSV

    Returns:
        Validation results dictionary
    """
    print(f"\n{'#'*60}")
    print(f"Validating: {filepath.name}")
    print(f"{'#'*60}")

    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    results = {
        'file': filepath.name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
    }

    # Run all checks
    results['required_cols_ok'] = check_required_columns(
        df, CONTACT_MODEL_COLUMNS, "Pitch Data"
    )
    results['missing_stats'] = check_missing_data(df, "Pitch Data")
    results['value_ranges'] = check_value_ranges(df, "Pitch Data")
    results['date_range'] = check_date_ranges(df, "Pitch Data")
    check_class_distribution(df, "Pitch Data")

    return results


def validate_batted_ball_data(filepath: Path) -> dict:
    """
    Validate batted ball data file.

    Args:
        filepath: Path to batted ball data CSV

    Returns:
        Validation results dictionary
    """
    print(f"\n{'#'*60}")
    print(f"Validating: {filepath.name}")
    print(f"{'#'*60}")

    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    results = {
        'file': filepath.name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
    }

    # Run all checks
    results['required_cols_ok'] = check_required_columns(
        df, BATTED_BALL_COLUMNS, "Batted Ball Data"
    )
    results['missing_stats'] = check_missing_data(df, "Batted Ball Data")
    results['value_ranges'] = check_value_ranges(df, "Batted Ball Data")
    results['date_range'] = check_date_ranges(df, "Batted Ball Data")
    check_class_distribution(df, "Batted Ball Data")

    return results


def generate_validation_report(results: list) -> str:
    """
    Generate summary validation report.

    Args:
        results: List of validation result dictionaries

    Returns:
        Markdown formatted report
    """
    report = ["# Data Validation Report", ""]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    for r in results:
        report.append(f"## {r['file']}")
        report.append(f"- Total rows: {r['total_rows']:,}")
        report.append(f"- Total columns: {r['total_columns']}")
        report.append(f"- Required columns OK: {r['required_cols_ok']}")
        if r['date_range'][0]:
            report.append(f"- Date range: {r['date_range'][0].date()} to {r['date_range'][1].date()}")
        report.append("")

    return "\n".join(report)


def main():
    """Main validation function."""
    print("="*60)
    print("Baseball Prediction Models - Data Validation")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pitch_dir = PROJECT_ROOT / "data" / "raw" / "pitch_data"
    batted_dir = PROJECT_ROOT / "data" / "raw" / "batted_ball_data"

    all_results = []

    # Validate pitch data
    pitch_files = list(pitch_dir.glob("*_full_pitches.csv"))
    for f in sorted(pitch_files):
        try:
            results = validate_pitch_data(f)
            all_results.append(results)
        except Exception as e:
            print(f"Error validating {f}: {e}")

    # Validate batted ball data
    batted_files = list(batted_dir.glob("*_batted_balls.csv"))
    for f in sorted(batted_files):
        try:
            results = validate_batted_ball_data(f)
            all_results.append(results)
        except Exception as e:
            print(f"Error validating {f}: {e}")

    # Generate and save report
    if all_results:
        report = generate_validation_report(all_results)
        report_path = PROJECT_ROOT / "data" / "validation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nValidation report saved to: {report_path}")

    print(f"\n{'='*60}")
    print("Validation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
