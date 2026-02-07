# Local Development Guide

This guide explains how to run the Baseball Prediction application locally.

## Project Structure

```
Baseball project/
├── backend/                 # FastAPI backend
│   └── main.py             # API server
├── web/                     # React frontend
│   ├── src/
│   │   ├── api/            # API client and types
│   │   ├── components/     # React components
│   │   │   └── charts/     # Chart components
│   │   └── pages/          # Page components
│   └── package.json
├── data/
│   ├── raw/                # Downloaded Statcast data
│   │   ├── pitch_data/     # Pitch-by-pitch data
│   │   └── batted_ball_data/
│   ├── processed/          # Feature-engineered data
│   └── scripts/            # Data processing scripts
├── src/
│   ├── features/           # Feature engineering modules
│   └── models/             # Model training code
├── models/                  # Trained model artifacts
│   ├── contact_prediction/
│   └── hit_outcome/
├── notebooks/              # Jupyter notebooks
├── DESIGN.md              # UI design system
├── requirements.txt       # Python dependencies
└── LOCAL_DEV.md           # This file
```

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- npm 9 or higher

## Backend Setup

### 1. Create and activate virtual environment

Navigate to the project root and create a Python virtual environment. Activate it before running any Python commands.

### 2. Install Python dependencies

Install the packages listed in `requirements.txt`. The key dependencies include:
- FastAPI and uvicorn for the API server
- pandas, numpy for data processing
- scikit-learn, xgboost, lightgbm for ML models
- pybaseball for data access

### 3. Start the backend server

Run the FastAPI server from the `backend/` directory. By default, it starts on port 8000.

The server will:
- Load trained models from `models/` directory (or use mock models if not available)
- Expose prediction endpoints at `/api/predict/contact` and `/api/predict/outcome`
- Provide a health check at `/health`

### 4. Verify backend is running

Access the health endpoint at `http://localhost:8000/health`. You should see a JSON response with status "healthy" and information about loaded models.

The API documentation is available at `http://localhost:8000/docs`.

## Frontend Setup

### 1. Install Node dependencies

Navigate to the `web/` directory and install packages using npm.

### 2. Configure API URL (optional)

By default, the frontend connects to `http://localhost:8000`. To change this, create a `.env.local` file in the `web/` directory with:

```
VITE_API_URL=http://your-api-url:port
```

### 3. Start the development server

Run the Vite development server. It typically starts on port 5173.

### 4. Access the application

Open your browser to the URL shown in the terminal (usually `http://localhost:5173`).

## Using the Application

### Contact Prediction Page (`/contact`)

This page predicts whether a batter will make contact with a pitch.

**Inputs:**
- Pitch type (Four-Seam, Slider, etc.)
- Velocity (50-105 mph)
- Horizontal and vertical movement (inches)
- Pitch location (feet from center of plate)
- Count (balls and strikes)
- Batter and pitcher handedness

**Outputs:**
- Contact probability (0-100%)
- Strike zone visualization with pitch location
- Adjustable threshold for contact/no-contact classification

### Hit Outcome Page (`/outcome`)

This page predicts the outcome type after a ball is hit.

**Inputs:**
- Exit velocity (30-120 mph)
- Launch angle (-90° to 90°)
- Spray angle (optional)
- Batter handedness and sprint speed (optional)

**Outputs:**
- Predicted outcome (Out, Single, Double, Triple, Home Run)
- Probability distribution for all outcomes
- Expected wOBA (weighted on-base average)

## Training Your Own Models

### 1. Download data

Run the data download script to fetch Statcast data. This downloads pitch-by-pitch data for 2023-2024 seasons.

### 2. Validate data

Run the validation script to verify data quality and check for missing values.

### 3. Feature engineering

The feature engineering modules in `src/features/` create the features needed for training:
- `contact_features.py` - Features for contact prediction
- `outcome_features.py` - Features for hit outcome prediction

### 4. Train models

The model training code is in `src/models/`. Training creates model artifacts in the `models/` directory with corresponding metadata JSON files.

### 5. Update the backend

The backend automatically loads models from the `models/` directory on startup. Restart the server after training new models.

## Troubleshooting

### Port already in use

If ports 8000 or 5173 are in use, either stop the conflicting process or configure different ports.

### CORS errors

The backend includes CORS middleware that allows all origins in development. If you encounter CORS errors, verify the backend is running and accessible.

### Missing models

If models aren't found, the backend uses mock models that provide reasonable default predictions. Train your own models for accurate predictions.

### Data download failures

The pybaseball library may timeout on large queries. The download script handles this by downloading month by month with retries.

### Missing dependencies

Ensure both Python and Node dependencies are installed. Check that you're using the correct virtual environment for Python commands.

## API Reference

### POST /api/predict/contact

Predict contact probability.

**Request:**
```json
{
  "pitch_type": "FF",
  "release_speed": 92.0,
  "pfx_x": 0.0,
  "pfx_z": 10.0,
  "plate_x": 0.0,
  "plate_z": 2.5,
  "balls": 0,
  "strikes": 0,
  "stand": "R",
  "p_throws": "R"
}
```

**Response:**
```json
{
  "prob_contact": 0.72,
  "threshold": 0.5,
  "predicted": 1,
  "request_id": "abc123",
  "latency_ms": 5.2
}
```

### POST /api/predict/outcome

Predict hit outcome.

**Request:**
```json
{
  "launch_speed": 95.0,
  "launch_angle": 15.0,
  "spray_angle": 0.0,
  "stand": "R",
  "sprint_speed": 27.0
}
```

**Response:**
```json
{
  "probs": {
    "out": 0.35,
    "single": 0.40,
    "double": 0.15,
    "triple": 0.02,
    "home_run": 0.08
  },
  "predicted": "single",
  "xwoba": 0.432,
  "request_id": "def456",
  "latency_ms": 8.1
}
```

### GET /health

Check server health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "contact": true,
    "outcome": true
  },
  "version": "1.0.0"
}
```
