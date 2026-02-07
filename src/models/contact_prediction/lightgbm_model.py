"""
LightGBM Contact Prediction Model

Primary model for predicting whether a batter will make contact.
Reference: research.md - "Model 1: Primary Model: LightGBM"
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# Default hyperparameters from research.md
DEFAULT_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'max_depth': 12,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# Hyperparameter tuning grid from prompt
TUNING_GRID = {
    'num_leaves': [31, 50, 100],
    'max_depth': [8, 12, 15, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'min_child_samples': [20, 50, 100],
}


class ContactLightGBMModel:
    """LightGBM model for contact prediction."""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the model.

        Args:
            params: Optional dictionary of LightGBM parameters
        """
        self.params = params or DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.training_metadata = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        n_estimators: int = 500,
        early_stopping_rounds: int = 50
    ) -> 'ContactLightGBMModel':
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            n_estimators: Number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Self for chaining
        """
        self.feature_names = list(X_train.columns)

        print(f"Training LightGBM Contact Model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {len(self.feature_names)}")

        # Create model
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **self.params
        )

        # Prepare callbacks
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            print(f"  Validation samples: {len(X_val):,}")

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

        # Store metadata
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'params': self.params,
            'n_estimators': self.model.n_estimators_,
        }

        print(f"  Training complete!")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict contact probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of probabilities [P(no contact), P(contact)]
        """
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Features DataFrame
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        y_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'threshold': threshold,
        }

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores
        """
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Normalize to percentages
        importance['importance_pct'] = (
            importance['importance'] / importance['importance'].sum() * 100
        )

        return importance

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict:
        """
        Perform cross-validation.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds

        Returns:
            Dictionary with CV results
        """
        print(f"Running {cv}-fold cross-validation...")

        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        results = {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
        }

        print(f"  CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        return results

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """
        Perform hyperparameter tuning with GridSearchCV.

        Args:
            X: Features DataFrame
            y: Target Series
            param_grid: Parameter grid (uses default if None)
            cv: Number of CV folds

        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            # Use smaller grid for faster tuning
            param_grid = {
                'num_leaves': [31, 50],
                'max_depth': [8, 12],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 500],
            }

        print(f"Hyperparameter tuning with {cv}-fold CV...")
        print(f"  Grid: {param_grid}")

        base_model = lgb.LGBMClassifier(**self.params)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            }
        }

        print(f"  Best params: {results['best_params']}")
        print(f"  Best score: {results['best_score']:.4f}")

        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_

        return results

    def save(self, filepath: Path) -> None:
        """
        Save model and metadata.

        Args:
            filepath: Path to save model (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = filepath.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        print(f"Metadata saved to: {meta_path}")

    @classmethod
    def load(cls, filepath: Path) -> 'ContactLightGBMModel':
        """
        Load model and metadata.

        Args:
            filepath: Path to model file (without extension)

        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)

        instance = cls()

        # Load model
        model_path = filepath.with_suffix('.pkl')
        instance.model = joblib.load(model_path)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                instance.training_metadata = json.load(f)
            instance.feature_names = instance.training_metadata.get('feature_names', [])
            instance.params = instance.training_metadata.get('params', DEFAULT_PARAMS)

        return instance


def train_contact_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Path,
    tune: bool = False
) -> ContactLightGBMModel:
    """
    Train and save contact prediction model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        output_dir: Directory to save model
        tune: Whether to perform hyperparameter tuning

    Returns:
        Trained model
    """
    model = ContactLightGBMModel()

    if tune:
        # Hyperparameter tuning on training data
        model.tune_hyperparameters(X_train, y_train)

    # Train model
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\nValidation Metrics:")
    val_metrics = model.evaluate(X_val, y_val)
    for name, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    model_path = output_dir / "lightgbm_model"
    model.save(model_path)

    return model
