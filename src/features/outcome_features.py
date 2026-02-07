"""
Hit Outcome Feature Engineering

Creates features for predicting the type of hit AFTER contact,
using batted ball characteristics measured immediately after impact.

Reference: research.md - "Model 2: Hit Outcome Prediction (Post-Contact)"
Reference: data_dictionary.md - Feature definitions
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


# Expected class distribution (from research.md)
CLASS_DISTRIBUTION = {
    'out': 0.70,
    'single': 0.15,
    'double': 0.08,
    'triple': 0.015,
    'home_run': 0.055,
}

# Class weights for imbalanced learning (from research.md)
CLASS_WEIGHTS = {
    'out': 1.0,
    'single': 1.0,
    'double': 2.0,
    'triple': 10.0,  # Very rare, needs heavy weighting
    'home_run': 1.5,
}


def create_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physical/kinematic features from batted ball data.

    Features:
        - exit_velocity_centered: Normalized exit velocity
        - launch_angle_squared: Quadratic effect of launch angle
        - barrel_indicator: Optimal EV/LA combination

    Args:
        df: DataFrame with launch_speed and launch_angle

    Returns:
        DataFrame with new physical features
    """
    df = df.copy()

    # Exit velocity centered (normalized)
    ev_mean = df['launch_speed'].mean()
    df['exit_velocity_centered'] = df['launch_speed'] - ev_mean

    # Launch angle squared (captures quadratic effect)
    df['launch_angle_squared'] = df['launch_angle'] ** 2

    # Barrel indicator (optimal EV/LA combination)
    # Per research.md: >98 mph EV and 26-30° LA
    df['barrel_indicator'] = (
        (df['launch_speed'] >= 98) &
        (df['launch_angle'] >= 26) &
        (df['launch_angle'] <= 30)
    ).astype(int)

    # Sweet spot indicator (wider optimal range)
    df['sweet_spot'] = (
        (df['launch_angle'] >= 8) &
        (df['launch_angle'] <= 32)
    ).astype(int)

    # Hard hit indicator (>95 mph)
    df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)

    return df


def create_spray_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spray angle features from hit coordinates.

    Features:
        - spray_angle: Horizontal direction of hit
        - distance_from_foul_line: How close to foul territory
        - depth_of_hit: Approximate depth of batted ball

    Args:
        df: DataFrame with hc_x and hc_y columns

    Returns:
        DataFrame with new spray features
    """
    df = df.copy()

    # Hit coordinates reference:
    # hc_x: 125.42 is approximately center field
    # hc_y: Distance down the field

    # Spray angle (in degrees)
    # Negative = pull side, Positive = opposite field
    center_x = 125.42
    if 'hc_x' in df.columns and 'hc_y' in df.columns:
        df['spray_angle'] = np.arctan2(
            df['hc_x'] - center_x,
            df['hc_y']
        ) * (180 / np.pi)

        # Distance from foul line (approximation)
        df['distance_from_foul_line'] = np.abs(df['spray_angle'])

        # Depth of hit
        df['depth_of_hit'] = df['hc_y']

    return df


def create_batted_ball_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on batted ball type classification.

    Features:
        - is_ground_ball, is_fly_ball, is_line_drive, is_popup

    Args:
        df: DataFrame with bb_type column

    Returns:
        DataFrame with batted ball type indicators
    """
    df = df.copy()

    if 'bb_type' in df.columns:
        df['is_ground_ball'] = (df['bb_type'] == 'ground_ball').astype(int)
        df['is_fly_ball'] = (df['bb_type'] == 'fly_ball').astype(int)
        df['is_line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
        df['is_popup'] = (df['bb_type'] == 'popup').astype(int)

    return df


def create_handedness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create batter handedness features.

    Args:
        df: DataFrame with stand column

    Returns:
        DataFrame with handedness indicators
    """
    df = df.copy()

    if 'stand' in df.columns:
        df['batter_left'] = (df['stand'] == 'L').astype(int)
        df['batter_right'] = (df['stand'] == 'R').astype(int)

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical target variable for hit outcome.

    Maps events to simplified outcome categories:
        - out: field_out, force_out, grounded_into_double_play, etc.
        - single: single
        - double: double
        - triple: triple
        - home_run: home_run

    Args:
        df: DataFrame with events column

    Returns:
        DataFrame with outcome target variable
    """
    df = df.copy()

    # Mapping from raw events to simplified categories
    event_mapping = {
        # Singles
        'single': 'single',

        # Doubles
        'double': 'double',

        # Triples
        'triple': 'triple',

        # Home runs
        'home_run': 'home_run',

        # Outs (various types)
        'field_out': 'out',
        'force_out': 'out',
        'grounded_into_double_play': 'out',
        'double_play': 'out',
        'fielders_choice': 'out',
        'fielders_choice_out': 'out',
        'sac_fly': 'out',
        'sac_bunt': 'out',
        'sac_fly_double_play': 'out',
        'triple_play': 'out',
        'field_error': 'out',  # Still an out for batter
    }

    df['outcome'] = df['events'].map(event_mapping)

    # Remove unmapped events
    df = df[df['outcome'].notna()].copy()

    return df


def engineer_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering for hit outcome model.

    Args:
        df: Raw batted ball data DataFrame

    Returns:
        DataFrame with all engineered features
    """
    print("Engineering hit outcome features...")

    # Apply all feature transformations
    df = create_physical_features(df)
    df = create_spray_features(df)
    df = create_batted_ball_type_features(df)
    df = create_handedness_features(df)
    df = create_target_variable(df)

    print(f"  Created {df.shape[1]} total columns")

    return df


def get_outcome_model_features() -> list:
    """
    Return list of features for hit outcome model.

    Returns:
        List of feature column names
    """
    return [
        # Primary features (80% importance combined)
        'launch_speed',        # 45% importance
        'launch_angle',        # 35% importance

        # Physical features
        'exit_velocity_centered',
        'launch_angle_squared',
        'barrel_indicator',
        'sweet_spot',
        'hard_hit',

        # Spray features (10% importance)
        'spray_angle',
        'distance_from_foul_line',
        'depth_of_hit',

        # Batted ball type
        'is_ground_ball',
        'is_fly_ball',
        'is_line_drive',
        'is_popup',

        # Handedness
        'batter_left',
        'batter_right',
    ]


def prepare_outcome_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for hit outcome model training.

    Args:
        df: Raw batted ball data

    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    # Engineer features
    df = engineer_outcome_features(df)

    # Get feature columns
    feature_cols = get_outcome_model_features()

    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df['outcome'].copy()

    # Handle missing values
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X):,}")
    print(f"  Missing values filled: {missing_before:,}")

    # Show class distribution
    print("\n  Class distribution:")
    for cls, count in y.value_counts().items():
        pct = count / len(y) * 100
        print(f"    {cls}: {count:,} ({pct:.1f}%)")

    return X, y


def get_class_weights() -> Dict[str, float]:
    """
    Return class weights for handling imbalanced data.

    Returns:
        Dictionary mapping class names to weights
    """
    return CLASS_WEIGHTS.copy()


def encode_target_for_training(y: pd.Series) -> Tuple[pd.Series, dict]:
    """
    Encode target variable as integers for model training.

    Args:
        y: Series with outcome categories

    Returns:
        Tuple of (encoded Series, label mapping)
    """
    # Define consistent ordering
    label_order = ['out', 'single', 'double', 'triple', 'home_run']
    label_map = {label: i for i, label in enumerate(label_order)}

    y_encoded = y.map(label_map)

    return y_encoded, label_map
