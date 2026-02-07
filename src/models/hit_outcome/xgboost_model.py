"""
XGBoost Hit Outcome Prediction Model

Primary model for predicting hit outcome type after contact.
Reference: research.md - "Model 2: Primary Model: XGBoost"
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE


# Label mapping
LABEL_ORDER = ['out', 'single', 'double', 'triple', 'home_run']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_ORDER)}
REVERSE_LABEL_MAP = {i: label for label, i in LABEL_MAP.items()}

# Default hyperparameters from research.md
DEFAULT_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 5,
    'max_depth': 12,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}

# Class weights from research.md
CLASS_WEIGHTS = {
    'out': 1.0,
    'single': 1.0,
    'double': 2.0,
    'triple': 10.0,
    'home_run': 1.5,
}

# wOBA weights for calculating expected wOBA
WOBA_WEIGHTS = {
    'out': 0.0,
    'single': 0.88,
    'double': 1.24,
    'triple': 1.56,
    'home_run': 2.00,
}


class OutcomeXGBoostModel:
    """XGBoost model for hit outcome prediction."""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the model.

        Args:
            params: Optional dictionary of XGBoost parameters
        """
        self.params = params or DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.training_metadata = {}

    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        """Encode string labels to integers."""
        return y.map(LABEL_MAP).values

    def _decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Decode integer labels to strings."""
        return np.array([REVERSE_LABEL_MAP[i] for i in y])

    def _get_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Calculate sample weights based on class."""
        return y.map(CLASS_WEIGHTS).values

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        apply_smote: bool = True,
        early_stopping_rounds: int = 50
    ) -> 'OutcomeXGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets (string labels)
            X_val: Optional validation features
            y_val: Optional validation targets
            apply_smote: Whether to apply SMOTE for class imbalance
            early_stopping_rounds: Early stopping patience

        Returns:
            Self for chaining
        """
        self.feature_names = list(X_train.columns)

        print(f"Training XGBoost Hit Outcome Model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {len(self.feature_names)}")

        # Apply SMOTE if requested
        if apply_smote:
            print("  Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train, y_train
            )
            print(f"  Resampled training samples: {len(X_train_resampled):,}")
        else:
            X_train_resampled = X_train
            y_train_resampled = y_train

        # Encode labels
        y_train_encoded = self._encode_labels(y_train_resampled)

        # Get sample weights
        sample_weights = self._get_sample_weights(y_train_resampled)

        # Create model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self._encode_labels(y_val)
            eval_set = [(X_val, y_val_encoded)]
            print(f"  Validation samples: {len(X_val):,}")

        # Train
        self.model.fit(
            X_train_resampled,
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )

        # Store metadata
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_samples_resampled': len(X_train_resampled),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'params': {k: v for k, v in self.params.items() if k != 'n_jobs'},
            'label_order': LABEL_ORDER,
            'applied_smote': apply_smote,
        }

        print(f"  Training complete!")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make class predictions.

        Args:
            X: Features DataFrame

        Returns:
            Array of predicted class labels (strings)
        """
        y_pred_encoded = self.model.predict(X)
        return self._decode_labels(y_pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Dictionary mapping class names to probability arrays
        """
        proba = self.model.predict_proba(X)
        return {
            label: proba[:, i]
            for i, label in enumerate(LABEL_ORDER)
        }

    def predict_xwoba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate expected wOBA from predicted probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of expected wOBA values
        """
        proba = self.predict_proba(X)
        xwoba = np.zeros(len(X))

        for label, weight in WOBA_WEIGHTS.items():
            xwoba += proba[label] * weight

        return xwoba

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Features DataFrame
            y: True labels (strings)

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)

        # Overall metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
        }

        # Per-class metrics
        for label in LABEL_ORDER:
            mask = y == label
            if mask.sum() > 0:
                label_pred = y_pred[mask]
                metrics[f'{label}_precision'] = (label_pred == label).mean()

        # Per-class F1 scores
        f1_scores = f1_score(y, y_pred, average=None, labels=LABEL_ORDER)
        for i, label in enumerate(LABEL_ORDER):
            metrics[f'{label}_f1'] = f1_scores[i]

        return metrics

    def get_confusion_matrix(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get confusion matrix as DataFrame.

        Args:
            X: Features DataFrame
            y: True labels

        Returns:
            Confusion matrix DataFrame
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred, labels=LABEL_ORDER)
        return pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)

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

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ) -> Dict:
        """
        Perform hyperparameter tuning with GridSearchCV.

        Args:
            X: Features DataFrame
            y: Target Series (string labels)
            param_grid: Parameter grid (uses default if None)
            cv: Number of CV folds

        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            # Use smaller grid for faster tuning
            param_grid = {
                'max_depth': [8, 12],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [500, 1000],
            }

        print(f"Hyperparameter tuning with {cv}-fold CV...")

        # Encode labels
        y_encoded = self._encode_labels(y)

        # Create base model
        base_params = {k: v for k, v in self.params.items()
                       if k not in param_grid}
        base_model = xgb.XGBClassifier(**base_params)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y_encoded)

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        }

        print(f"  Best params: {results['best_params']}")
        print(f"  Best score: {results['best_score']:.4f}")

        # Update model
        self.params.update(grid_search.best_params_)

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
    def load(cls, filepath: Path) -> 'OutcomeXGBoostModel':
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


def train_outcome_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Path,
    tune: bool = False
) -> OutcomeXGBoostModel:
    """
    Train and save hit outcome model.

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
    model = OutcomeXGBoostModel()

    if tune:
        model.tune_hyperparameters(X_train, y_train)

    # Train model
    model.train(X_train, y_train, X_val, y_val, apply_smote=True)

    # Evaluate
    print("\nValidation Metrics:")
    val_metrics = model.evaluate(X_val, y_val)
    for name, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = model.get_confusion_matrix(X_val, y_val)
    print(cm.to_string())

    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    model_path = output_dir / "xgboost_model"
    model.save(model_path)

    return model
