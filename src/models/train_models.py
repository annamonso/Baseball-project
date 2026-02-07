"""
Model Training Script

Trains both contact prediction and hit outcome models.
Saves trained models and metadata to models/ directory.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.contact_prediction.lightgbm_model import ContactLightGBMModel
from src.models.hit_outcome.xgboost_model import OutcomeXGBoostModel


def load_contact_data():
    """Load prepared contact prediction data."""
    data_dir = PROJECT_ROOT / "data" / "processed" / "contact_prediction"

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Load feature list
    with open(data_dir / "features.txt", 'r') as f:
        features = [line.strip() for line in f.readlines()]

    # Split features and target
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    X_test = test[features]
    y_test = test['target']

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def load_outcome_data():
    """Load prepared hit outcome data."""
    data_dir = PROJECT_ROOT / "data" / "processed" / "hit_outcome"

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "validation.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Load feature list
    with open(data_dir / "features.txt", 'r') as f:
        features = [line.strip() for line in f.readlines()]

    # Split features and target
    X_train = train[features]
    y_train = train['target']
    X_val = val[features]
    y_val = val['target']
    X_test = test[features]
    y_test = test['target']

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_contact_model():
    """Train contact prediction model."""
    print("=" * 60)
    print("Training Contact Prediction Model (LightGBM)")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, features = load_contact_data()

    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(features)}")

    # Create and train model
    model = ContactLightGBMModel()

    # Train with validation
    model.train(X_train, y_train, X_val, y_val, n_estimators=500)

    # Evaluate on validation set
    print("\nValidation Metrics:")
    val_metrics = model.evaluate(X_val, y_val)
    for name, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Evaluate on test set
    print("\nTest Metrics:")
    test_metrics = model.evaluate(X_test, y_test)
    for name, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    output_dir = PROJECT_ROOT / "models" / "contact_prediction"
    model.save(output_dir / "lightgbm_model")

    return model, test_metrics


def train_outcome_model():
    """Train hit outcome model."""
    print("\n" + "=" * 60)
    print("Training Hit Outcome Model (XGBoost)")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, features = load_outcome_data()

    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(features)}")

    # Create and train model
    model = OutcomeXGBoostModel()

    # Train with SMOTE for class imbalance
    model.train(X_train, y_train, X_val, y_val, apply_smote=True)

    # Evaluate on validation set
    print("\nValidation Metrics:")
    val_metrics = model.evaluate(X_val, y_val)
    for name, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Evaluate on test set
    print("\nTest Metrics:")
    test_metrics = model.evaluate(X_test, y_test)
    for name, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix (Test):")
    cm = model.get_confusion_matrix(X_test, y_test)
    print(cm.to_string())

    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    output_dir = PROJECT_ROOT / "models" / "hit_outcome"
    model.save(output_dir / "xgboost_model")

    return model, test_metrics


def main():
    """Main training function."""
    print("=" * 60)
    print("Baseball Prediction Models - Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if data exists
    contact_data = PROJECT_ROOT / "data" / "processed" / "contact_prediction" / "train.csv"
    outcome_data = PROJECT_ROOT / "data" / "processed" / "hit_outcome" / "train.csv"

    if not contact_data.exists() or not outcome_data.exists():
        print("\nPrepared data not found. Running data preparation...")
        from data.scripts.prepare_data import main as prepare_main
        prepare_main()

    # Train models
    contact_model, contact_metrics = train_contact_model()
    outcome_model, outcome_metrics = train_outcome_model()

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"\nContact Model (Test):")
    print(f"  Accuracy: {contact_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {contact_metrics['roc_auc']:.4f}")
    print(f"  F1: {contact_metrics['f1']:.4f}")

    print(f"\nOutcome Model (Test):")
    print(f"  Accuracy: {outcome_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {outcome_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {outcome_metrics['f1_weighted']:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
