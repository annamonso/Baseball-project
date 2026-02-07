"""
Contact Prediction Feature Engineering

Creates features for predicting whether a batter will make contact
with the ball BEFORE impact, using pitch characteristics.

Reference: research.md - "Model 1: Contact Prediction (Pre-Impact)"
Reference: data_dictionary.md - Feature definitions
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pitch location-based features.

    Features:
        - distance_from_center: Distance from center of strike zone
        - in_zone: Binary indicator if pitch in strike zone

    Args:
        df: DataFrame with plate_x and plate_z columns

    Returns:
        DataFrame with new location features
    """
    df = df.copy()

    # Strike zone center is approximately (0, 2.5) in feet
    # Reference: data_dictionary.md
    strike_zone_center_x = 0.0
    strike_zone_center_z = 2.5

    # Distance from center of strike zone
    df['distance_from_center'] = np.sqrt(
        (df['plate_x'] - strike_zone_center_x) ** 2 +
        (df['plate_z'] - strike_zone_center_z) ** 2
    )

    # In-zone indicator (simplified strike zone)
    # Width: -0.83 to 0.83 feet (half of 17-inch plate)
    # Height: 1.5 to 3.5 feet (typical strike zone)
    df['in_zone'] = (
        (np.abs(df['plate_x']) < 0.83) &
        (df['plate_z'] > 1.5) &
        (df['plate_z'] < 3.5)
    ).astype(int)

    return df


def create_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pitch movement-based features.

    Features:
        - total_movement: Magnitude of pitch break (inches)
        - effective_velocity: Perceived velocity accounting for rise

    Args:
        df: DataFrame with pfx_x, pfx_z, release_speed columns

    Returns:
        DataFrame with new movement features
    """
    df = df.copy()

    # Total movement magnitude (inches)
    df['total_movement'] = np.sqrt(df['pfx_x'] ** 2 + df['pfx_z'] ** 2)

    # Effective velocity (perceived speed)
    # Pitches with more "rise" (positive pfx_z) appear faster
    # Division by 12 is empirical scaling factor
    df['effective_velocity'] = df['release_speed'] - (df['pfx_z'] / 12)

    return df


def create_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create count situation features.

    Features:
        - count_advantage: Pitcher's count advantage (strikes - balls)
        - two_strikes: Binary indicator for two-strike count
        - hitters_count: Binary for advantageous batter count

    Args:
        df: DataFrame with balls and strikes columns

    Returns:
        DataFrame with new count features
    """
    df = df.copy()

    # Count advantage for pitcher
    df['count_advantage'] = df['strikes'] - df['balls']

    # Two-strike indicator
    df['two_strikes'] = (df['strikes'] == 2).astype(int)

    # Hitter's count indicator (2-0, 2-1, 3-0, 3-1)
    df['hitters_count'] = (
        (df['balls'] >= 2) & (df['strikes'] <= 1)
    ).astype(int)

    return df


def create_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pitcher/batter matchup features.

    Features:
        - same_side: Same-handed batter and pitcher
        - platoon_advantage: Opposite-handed matchup (batter advantage)

    Args:
        df: DataFrame with stand and p_throws columns

    Returns:
        DataFrame with new matchup features
    """
    df = df.copy()

    # Same-side matchup (harder for batters)
    df['same_side'] = (df['stand'] == df['p_throws']).astype(int)

    # Platoon advantage (opposite hands favors batter)
    df['platoon_advantage'] = (df['stand'] != df['p_throws']).astype(int)

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable for contact prediction.

    Target:
        - contact: 1 if ball was hit into play, 0 otherwise

    Args:
        df: DataFrame with description column

    Returns:
        DataFrame with contact target variable
    """
    df = df.copy()

    # Contact = ball was hit into play
    df['contact'] = (df['description'] == 'hit_into_play').astype(int)

    return df


def encode_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode pitch type.

    Args:
        df: DataFrame with pitch_type column

    Returns:
        DataFrame with pitch type dummy variables
    """
    df = df.copy()

    if 'pitch_type' in df.columns:
        # Get dummies for common pitch types
        pitch_dummies = pd.get_dummies(
            df['pitch_type'],
            prefix='pitch',
            dummy_na=False
        )
        df = pd.concat([df, pitch_dummies], axis=1)

    return df


def engineer_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering for contact prediction model.

    Args:
        df: Raw pitch data DataFrame

    Returns:
        DataFrame with all engineered features
    """
    print("Engineering contact prediction features...")

    # Apply all feature transformations
    df = create_location_features(df)
    df = create_movement_features(df)
    df = create_count_features(df)
    df = create_matchup_features(df)
    df = create_target_variable(df)
    df = encode_pitch_type(df)

    print(f"  Created {df.shape[1]} total columns")

    return df


def get_contact_model_features() -> list:
    """
    Return list of features for contact prediction model.

    Returns:
        List of feature column names
    """
    return [
        # Primary location features (60% importance)
        'plate_x',
        'plate_z',
        'distance_from_center',
        'in_zone',

        # Movement features (12% importance)
        'pfx_x',
        'pfx_z',
        'total_movement',

        # Velocity features (15% importance)
        'release_speed',
        'effective_velocity',

        # Count features (8% importance)
        'balls',
        'strikes',
        'count_advantage',
        'two_strikes',
        'hitters_count',

        # Matchup features (5% importance)
        'same_side',
        'platoon_advantage',
    ]


def prepare_contact_data(
    df: pd.DataFrame,
    include_pitch_type: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for contact prediction model training.

    Args:
        df: Raw pitch data
        include_pitch_type: Whether to include pitch type dummies

    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    # Engineer features
    df = engineer_contact_features(df)

    # Get feature columns
    feature_cols = get_contact_model_features()

    # Add pitch type dummies if requested
    if include_pitch_type:
        pitch_cols = [c for c in df.columns if c.startswith('pitch_')]
        feature_cols = feature_cols + pitch_cols

    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df['contact'].copy()

    # Handle missing values
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X):,}")
    print(f"  Missing values filled: {missing_before:,}")
    print(f"  Contact rate: {y.mean():.2%}")

    return X, y
