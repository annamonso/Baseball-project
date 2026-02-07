"""
Baseball Prediction API

FastAPI backend for contact and hit outcome predictions.
Loads trained models and provides prediction endpoints.
"""

import sys
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ContactPredictionRequest(BaseModel):
    """Request model for contact prediction."""
    pitch_type: str = Field(..., description="Type of pitch (FF, SL, CH, CU, etc.)")
    release_speed: float = Field(..., ge=50, le=105, description="Pitch velocity in mph")
    pfx_x: float = Field(..., ge=-25, le=25, description="Horizontal movement in inches")
    pfx_z: float = Field(..., ge=-25, le=25, description="Vertical movement in inches")
    plate_x: float = Field(..., ge=-2.5, le=2.5, description="Horizontal location in feet")
    plate_z: float = Field(..., ge=0, le=5, description="Vertical location in feet")
    balls: int = Field(..., ge=0, le=3, description="Number of balls")
    strikes: int = Field(..., ge=0, le=2, description="Number of strikes")
    stand: str = Field(..., pattern="^[LR]$", description="Batter stance (L or R)")
    p_throws: str = Field(..., pattern="^[LR]$", description="Pitcher throws (L or R)")


class ContactPredictionResponse(BaseModel):
    """Response model for contact prediction."""
    prob_contact: float = Field(..., description="Probability of contact (0-1)")
    threshold: float = Field(..., description="Classification threshold")
    predicted: int = Field(..., description="Binary prediction (0 or 1)")
    top_features: Optional[List[Dict[str, float]]] = Field(
        None, description="Top feature contributions"
    )
    request_id: str = Field(..., description="Unique request ID")
    latency_ms: float = Field(..., description="Prediction latency in ms")


class OutcomePredictionRequest(BaseModel):
    """Request model for hit outcome prediction."""
    launch_speed: float = Field(..., ge=30, le=120, description="Exit velocity in mph")
    launch_angle: float = Field(..., ge=-90, le=90, description="Launch angle in degrees")
    spray_angle: Optional[float] = Field(None, ge=-45, le=45, description="Spray angle in degrees")
    stand: Optional[str] = Field(None, pattern="^[LR]$", description="Batter stance")
    sprint_speed: Optional[float] = Field(None, ge=24, le=31, description="Sprint speed ft/s")


class OutcomePredictionResponse(BaseModel):
    """Response model for hit outcome prediction."""
    probs: Dict[str, float] = Field(..., description="Class probabilities")
    predicted: str = Field(..., description="Predicted outcome class")
    xwoba: Optional[float] = Field(None, description="Expected wOBA")
    request_id: str = Field(..., description="Unique request ID")
    latency_ms: float = Field(..., description="Prediction latency in ms")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    version: str


# ============================================================================
# Feature Engineering Functions
# ============================================================================

def engineer_contact_features(data: ContactPredictionRequest) -> pd.DataFrame:
    """Engineer features for contact prediction from request data."""

    # Base features
    features = {
        'plate_x': data.plate_x,
        'plate_z': data.plate_z,
        'pfx_x': data.pfx_x,
        'pfx_z': data.pfx_z,
        'release_speed': data.release_speed,
        'balls': data.balls,
        'strikes': data.strikes,
    }

    # Engineered features
    # Distance from center of strike zone
    features['distance_from_center'] = np.sqrt(
        data.plate_x ** 2 + (data.plate_z - 2.5) ** 2
    )

    # In-zone indicator
    features['in_zone'] = int(
        abs(data.plate_x) < 0.83 and
        data.plate_z > 1.5 and
        data.plate_z < 3.5
    )

    # Total movement
    features['total_movement'] = np.sqrt(data.pfx_x ** 2 + data.pfx_z ** 2)

    # Effective velocity
    features['effective_velocity'] = data.release_speed - (data.pfx_z / 12)

    # Count features
    features['count_advantage'] = data.strikes - data.balls
    features['two_strikes'] = int(data.strikes == 2)
    features['hitters_count'] = int(data.balls >= 2 and data.strikes <= 1)

    # Matchup features
    features['same_side'] = int(data.stand == data.p_throws)
    features['platoon_advantage'] = int(data.stand != data.p_throws)

    return pd.DataFrame([features])


def engineer_outcome_features(data: OutcomePredictionRequest) -> pd.DataFrame:
    """Engineer features for outcome prediction from request data."""

    # Base features
    features = {
        'launch_speed': data.launch_speed,
        'launch_angle': data.launch_angle,
    }

    # Engineered features
    # Exit velocity centered (use typical mean of ~87 mph)
    features['exit_velocity_centered'] = data.launch_speed - 87.0

    # Launch angle squared
    features['launch_angle_squared'] = data.launch_angle ** 2

    # Barrel indicator
    features['barrel_indicator'] = int(
        data.launch_speed >= 98 and
        26 <= data.launch_angle <= 30
    )

    # Sweet spot
    features['sweet_spot'] = int(8 <= data.launch_angle <= 32)

    # Hard hit
    features['hard_hit'] = int(data.launch_speed >= 95)

    # Spray angle (use 0 if not provided)
    spray = data.spray_angle if data.spray_angle is not None else 0
    features['spray_angle'] = spray
    features['distance_from_foul_line'] = abs(spray)
    features['depth_of_hit'] = 200  # Default approximation

    # Batted ball type estimates based on launch angle
    features['is_ground_ball'] = int(data.launch_angle < 10)
    features['is_fly_ball'] = int(data.launch_angle > 25)
    features['is_line_drive'] = int(10 <= data.launch_angle <= 25)
    features['is_popup'] = int(data.launch_angle > 50)

    # Handedness
    features['batter_left'] = int(data.stand == 'L') if data.stand else 0
    features['batter_right'] = int(data.stand == 'R') if data.stand else 1

    return pd.DataFrame([features])


# ============================================================================
# Mock Models (used when real models aren't available)
# ============================================================================

class MockContactModel:
    """Mock contact model for development/testing."""

    def predict_proba(self, X):
        # Simple heuristic: in-zone pitches more likely to be contacted
        in_zone = X.get('in_zone', X.iloc[:, X.columns.get_loc('in_zone') if 'in_zone' in X.columns else 0])
        prob = 0.7 if in_zone.values[0] else 0.5
        # Adjust by count
        if 'two_strikes' in X.columns and X['two_strikes'].values[0]:
            prob *= 0.9
        return np.array([[1 - prob, prob]])


class MockOutcomeModel:
    """Mock outcome model for development/testing."""

    def predict_proba(self, X):
        ev = X['launch_speed'].values[0]
        la = X['launch_angle'].values[0]

        # Simple heuristics
        if ev >= 98 and 26 <= la <= 30:  # Barrel
            return np.array([[0.1, 0.2, 0.2, 0.1, 0.4]])
        elif ev >= 95 and 10 <= la <= 25:  # Hard hit line drive
            return np.array([[0.3, 0.4, 0.2, 0.05, 0.05]])
        elif la > 50:  # Popup
            return np.array([[0.95, 0.05, 0.0, 0.0, 0.0]])
        elif la < 0:  # Ground ball
            return np.array([[0.75, 0.2, 0.05, 0.0, 0.0]])
        else:  # Default
            return np.array([[0.7, 0.15, 0.08, 0.02, 0.05]])

    def predict(self, X):
        proba = self.predict_proba(X)
        labels = ['out', 'single', 'double', 'triple', 'home_run']
        return np.array([labels[np.argmax(proba[0])]])


# ============================================================================
# Application Setup
# ============================================================================

# Global model storage
models = {
    'contact': None,
    'outcome': None,
}


def load_models():
    """Load trained models or use mocks."""
    contact_model_path = PROJECT_ROOT / "models" / "contact_prediction" / "lightgbm_model.pkl"
    outcome_model_path = PROJECT_ROOT / "models" / "hit_outcome" / "xgboost_model.pkl"

    # Try to load contact model
    if contact_model_path.exists():
        try:
            import joblib
            models['contact'] = joblib.load(contact_model_path)
            logger.info(f"Loaded contact model from {contact_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load contact model: {e}")
            models['contact'] = MockContactModel()
    else:
        logger.warning("Contact model not found, using mock")
        models['contact'] = MockContactModel()

    # Try to load outcome model
    if outcome_model_path.exists():
        try:
            import joblib
            models['outcome'] = joblib.load(outcome_model_path)
            logger.info(f"Loaded outcome model from {outcome_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load outcome model: {e}")
            models['outcome'] = MockOutcomeModel()
    else:
        logger.warning("Outcome model not found, using mock")
        models['outcome'] = MockOutcomeModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Baseball Prediction API...")
    load_models()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Baseball Prediction API",
    description="API for contact and hit outcome predictions using MLB Statcast data",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "contact": models['contact'] is not None,
            "outcome": models['outcome'] is not None,
        },
        version="1.0.0"
    )


@app.post("/api/predict/contact", response_model=ContactPredictionResponse)
async def predict_contact(request: ContactPredictionRequest):
    """
    Predict probability of batter making contact.

    Uses pitch characteristics available before the ball reaches the plate.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        # Engineer features
        features = engineer_contact_features(request)

        # Make prediction
        proba = models['contact'].predict_proba(features)
        prob_contact = float(proba[0, 1])

        # Apply threshold
        threshold = 0.5
        predicted = int(prob_contact >= threshold)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"[{request_id}] Contact prediction: {prob_contact:.3f} ({latency_ms:.1f}ms)")

        return ContactPredictionResponse(
            prob_contact=round(prob_contact, 4),
            threshold=threshold,
            predicted=predicted,
            top_features=None,  # Could add SHAP values here
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/outcome", response_model=OutcomePredictionResponse)
async def predict_outcome(request: OutcomePredictionRequest):
    """
    Predict hit outcome type after contact.

    Uses batted ball characteristics measured immediately after impact.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        # Engineer features
        features = engineer_outcome_features(request)

        # Make prediction
        proba = models['outcome'].predict_proba(features)

        # Map to class names
        labels = ['out', 'single', 'double', 'triple', 'home_run']
        probs = {label: float(proba[0, i]) for i, label in enumerate(labels)}

        # Predicted class
        predicted = labels[np.argmax(proba[0])]

        # Calculate xwOBA
        woba_weights = {'out': 0.0, 'single': 0.88, 'double': 1.24,
                        'triple': 1.56, 'home_run': 2.00}
        xwoba = sum(probs[label] * woba_weights[label] for label in labels)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"[{request_id}] Outcome prediction: {predicted} ({latency_ms:.1f}ms)")

        return OutcomePredictionResponse(
            probs={k: round(v, 4) for k, v in probs.items()},
            predicted=predicted,
            xwoba=round(xwoba, 3),
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Evaluation & Report Endpoints
# ============================================================================

@app.get("/api/evaluation")
async def get_evaluation():
    """
    Get model evaluation metrics.

    Returns evaluation results for both contact and outcome models.
    """
    results_path = PROJECT_ROOT / "results" / "evaluation_results.json"

    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run evaluation script first."
        )

    try:
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results['models']
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/download")
async def download_report():
    """
    Download the comprehensive PDF project report.
    """
    report_path = PROJECT_ROOT / "results" / "Baseball_ML_Project_Report.pdf"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found. Generate the report first."
        )

    return FileResponse(
        path=str(report_path),
        filename="Baseball_ML_Project_Report.pdf",
        media_type="application/pdf"
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
