"""
Data Preparation Script

Creates train/validation/test splits for both models using temporal split strategy.
Reference: prompt.md - Phase 2.4 Train/Validation/Test Split
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.contact_features import engineer_contact_features, get_contact_model_features
from src.features.outcome_features import engineer_outcome_features, get_outcome_model_features


def temporal_split(df: pd.DataFrame, date_col: str = 'game_date') -> tuple:
    """
    Split data temporally for proper evaluation.

    Strategy from prompt:
    - Training: 2023 season + first half of 2024 (60%)
    - Validation: Second half of 2024 through August (20%)
    - Test: September-October 2024 (20%)

    Args:
        df: DataFrame with date column
        date_col: Name of date column

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Define split dates
    train_end = '2024-06-30'
    val_end = '2024-08-31'

    # Split
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]

    print(f"\nTemporal Split:")
    print(f"  Train: {train[date_col].min().date()} to {train[date_col].max().date()} ({len(train):,} samples)")
    print(f"  Val:   {val[date_col].min().date()} to {val[date_col].max().date()} ({len(val):,} samples)")
    print(f"  Test:  {test[date_col].min().date()} to {test[date_col].max().date()} ({len(test):,} samples)")

    return train, val, test


def prepare_contact_data():
    """Prepare data for contact prediction model."""
    print("=" * 60)
    print("Preparing Contact Prediction Data")
    print("=" * 60)

    # Load pitch data
    pitch_dir = PROJECT_ROOT / "data" / "raw" / "pitch_data"
    pitches_2023 = pd.read_csv(pitch_dir / "2023_full_pitches.csv")
    pitches_2024 = pd.read_csv(pitch_dir / "2024_full_pitches.csv")
    pitches = pd.concat([pitches_2023, pitches_2024], ignore_index=True)

    print(f"Total pitches loaded: {len(pitches):,}")

    # Engineer features
    pitches = engineer_contact_features(pitches)

    # Get feature columns
    feature_cols = get_contact_model_features()

    # Add pitch type dummies (only the new dummy columns, not original string columns)
    if 'pitch_type' in pitches.columns:
        pitch_dummies = pd.get_dummies(pitches['pitch_type'], prefix='pitchtype', dummy_na=False)
        pitches = pd.concat([pitches, pitch_dummies], axis=1)
        pitch_type_cols = list(pitch_dummies.columns)  # Only the actual dummy columns
        feature_cols = feature_cols + pitch_type_cols

    # Filter to available columns and exclude string columns
    available_features = [c for c in feature_cols if c in pitches.columns]
    # Only keep numeric features
    numeric_df = pitches[available_features].select_dtypes(include=[np.number])
    available_features = list(numeric_df.columns)

    # Create target
    target_col = 'contact'

    # Remove rows with missing values in key columns
    key_cols = ['plate_x', 'plate_z', 'release_speed', 'game_date', target_col]
    pitches = pitches.dropna(subset=[c for c in key_cols if c in pitches.columns])

    print(f"After cleaning: {len(pitches):,}")

    # Temporal split
    train, val, test = temporal_split(pitches)

    # Extract features and target
    X_train = train[available_features].copy()
    y_train = train[target_col]
    X_val = val[available_features].copy()
    y_val = val[target_col]
    X_test = test[available_features].copy()
    y_test = test[target_col]

    # Fill missing values only for numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    train_medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(train_medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)

    # Save
    output_dir = PROJECT_ROOT / "data" / "processed" / "contact_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_df.to_csv(output_dir / "train.csv", index=False)

    val_df = X_val.copy()
    val_df['target'] = y_val.values
    val_df.to_csv(output_dir / "validation.csv", index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save feature list
    with open(output_dir / "features.txt", 'w') as f:
        f.write('\n'.join(available_features))

    print(f"\nSaved to: {output_dir}")
    print(f"  train.csv: {len(train_df):,} samples")
    print(f"  validation.csv: {len(val_df):,} samples")
    print(f"  test.csv: {len(test_df):,} samples")
    print(f"  features.txt: {len(available_features)} features")

    # Class distribution
    print(f"\nClass distribution:")
    print(f"  Train contact rate: {y_train.mean():.2%}")
    print(f"  Val contact rate: {y_val.mean():.2%}")
    print(f"  Test contact rate: {y_test.mean():.2%}")


def prepare_outcome_data():
    """Prepare data for hit outcome model."""
    print("\n" + "=" * 60)
    print("Preparing Hit Outcome Data")
    print("=" * 60)

    # Load batted ball data
    batted_dir = PROJECT_ROOT / "data" / "raw" / "batted_ball_data"
    batted_2023 = pd.read_csv(batted_dir / "2023_batted_balls.csv")
    batted_2024 = pd.read_csv(batted_dir / "2024_batted_balls.csv")
    batted = pd.concat([batted_2023, batted_2024], ignore_index=True)

    print(f"Total batted balls loaded: {len(batted):,}")

    # Engineer features
    batted = engineer_outcome_features(batted)

    # Get feature columns
    feature_cols = get_outcome_model_features()
    available_features = [c for c in feature_cols if c in batted.columns]

    # Target column
    target_col = 'outcome'

    # Remove rows with missing outcome
    batted = batted.dropna(subset=[target_col])
    print(f"After cleaning: {len(batted):,}")

    # Temporal split
    train, val, test = temporal_split(batted)

    # Extract features and target
    X_train = train[available_features].copy()
    y_train = train[target_col]
    X_val = val[available_features].copy()
    y_val = val[target_col]
    X_test = test[available_features].copy()
    y_test = test[target_col]

    # Fill missing values only for numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    train_medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(train_medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)

    # Save
    output_dir = PROJECT_ROOT / "data" / "processed" / "hit_outcome"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_df.to_csv(output_dir / "train.csv", index=False)

    val_df = X_val.copy()
    val_df['target'] = y_val.values
    val_df.to_csv(output_dir / "validation.csv", index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save feature list
    with open(output_dir / "features.txt", 'w') as f:
        f.write('\n'.join(available_features))

    print(f"\nSaved to: {output_dir}")
    print(f"  train.csv: {len(train_df):,} samples")
    print(f"  validation.csv: {len(val_df):,} samples")
    print(f"  test.csv: {len(test_df):,} samples")
    print(f"  features.txt: {len(available_features)} features")

    # Class distribution
    print(f"\nClass distribution (Train):")
    for cls, count in y_train.value_counts().items():
        print(f"  {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")


def main():
    """Main function."""
    print("=" * 60)
    print("Data Preparation Script")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    prepare_contact_data()
    prepare_outcome_data()

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
