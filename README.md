# Baseball Pitch & Hit Outcome Prediction

A full-stack machine learning application that predicts pitch contact probability and hit outcomes using MLB Statcast data. Features a FastAPI backend serving trained gradient boosting models and an interactive React frontend with real-time visualizations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Backend API](#backend-api)
- [Frontend Application](#frontend-application)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## Project Overview

This project tackles two predictive tasks in baseball analytics:

1. **Contact Prediction** -- Given pre-impact pitch characteristics (location, velocity, movement, count), predict whether the batter will make contact with the pitch. Binary classification using LightGBM.

2. **Hit Outcome Prediction** -- Given post-impact batted ball metrics (exit velocity, launch angle, spray angle), predict the result: Out, Single, Double, Triple, or Home Run. Multi-class classification using XGBoost.

Both models are served through a REST API and consumed by an interactive web application with strike zone plots, baseball field visualizations, probability distributions, and a model performance dashboard.

---

## Key Results

| Model | Accuracy | ROC-AUC | Weighted F1 | Test Samples |
|-------|----------|---------|-------------|--------------|
| Contact Prediction (LightGBM) | **83.0%** | **0.805** | -- | 125,679 |
| Hit Outcome (XGBoost) | **85.4%** | -- | **0.871** | 21,486 |

**Hit Outcome Per-Class Performance:**

| Class | Proportion | Precision | Recall | F1 |
|-------|-----------|-----------|--------|-----|
| Out | 68% | 0.857 | 0.906 | 0.925 |
| Single | 21% | 0.864 | 0.761 | 0.805 |
| Double | 5.9% | 0.515 | 0.621 | 0.564 |
| Triple | 0.5% | 0.040 | 0.317 | 0.064 |
| Home Run | 4.3% | 0.834 | 0.879 | 0.855 |

---

## Architecture

```
┌─────────────────────┐       HTTP/JSON        ┌─────────────────────┐
│   React Frontend    │ ◄───────────────────►  │   FastAPI Backend   │
│   (Vite + TS)       │                         │   (Python)          │
│                     │                         │                     │
│  - Strike Zone Plot │   POST /predict/contact │  - Feature Eng.     │
│  - Field Plot       │   POST /predict/outcome │  - LightGBM Model   │
│  - Probability Bars │   GET  /evaluation      │  - XGBoost Model    │
│  - Metrics Dashboard│   GET  /health          │  - Pydantic Schemas │
└─────────────────────┘                         └─────────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────────┐
                                                │   Trained Models    │
                                                │   (joblib .pkl)     │
                                                └─────────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────────┐
                                                │   MLB Statcast Data │
                                                │   (pybaseball)      │
                                                │   2023-2024 seasons │
                                                └─────────────────────┘
```

---

## Tech Stack

### Data & Machine Learning

| Technology | Purpose |
|-----------|---------|
| **Python 3.13** | Primary language for data processing and ML |
| **pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | ML utilities, preprocessing, cross-validation, baseline models |
| **XGBoost** | Gradient boosting for hit outcome multi-class classification |
| **LightGBM** | Gradient boosting for contact binary classification |
| **imbalanced-learn (SMOTE)** | Synthetic oversampling for rare class handling (Triples) |
| **pybaseball** | MLB Statcast data access via Baseball Savant |
| **matplotlib / seaborn** | Exploratory data analysis and static visualizations |
| **Jupyter** | Interactive notebook-based analysis |
| **joblib** | Model serialization and persistence |

### Backend

| Technology | Purpose |
|-----------|---------|
| **FastAPI** | Async REST API framework with automatic OpenAPI docs |
| **Uvicorn** | ASGI server |
| **Pydantic v2** | Request/response validation and type-safe schemas |

### Frontend

| Technology | Purpose |
|-----------|---------|
| **React 19** | UI component framework |
| **TypeScript 5.9** | Type-safe JavaScript |
| **Vite 7** | Build tool and dev server |
| **Tailwind CSS 4** | Utility-first styling with custom design tokens |
| **Recharts 3** | Responsive data visualization (bar charts, distributions) |
| **React Hook Form** | Performant form state management |
| **Zod 4** | Schema-based client-side validation |
| **React Router 7** | Client-side routing |

---

## Data Pipeline

### Source

**MLB Statcast** via Baseball Savant -- pitch-level tracking data covering ~700,000 pitches per season with 120+ metrics per record (pitch velocity, movement, location, batted ball characteristics, game context).

### Processing Workflow

```
Raw Statcast CSVs (2023-2024)
        │
        ▼
  Data Validation
  (completeness, ranges, missing values)
        │
        ▼
  Feature Engineering
  (16+ derived features per model)
        │
        ▼
  Temporal Train/Val/Test Split
  ├── Train: 2023 + H1 2024
  ├── Val:   H2 2024 (Jul-Aug)
  └── Test:  Sep-Oct 2024
        │
        ▼
  Class Imbalance Handling
  (SMOTE for rare classes)
```

---

## Feature Engineering

### Contact Model (16 features)

**Raw inputs:** pitch location (`plate_x`, `plate_z`), movement (`pfx_x`, `pfx_z`), velocity (`release_speed`), count (`balls`, `strikes`), handedness (`stand`, `p_throws`)

**Engineered features:**
- `distance_from_center` -- Euclidean distance from strike zone center
- `in_zone` -- binary strike zone indicator
- `total_movement` -- magnitude of pitch break
- `effective_velocity` -- velocity adjusted for vertical movement
- `count_advantage` -- strikes minus balls
- `two_strikes`, `hitters_count` -- situational count flags
- `same_side`, `platoon_advantage` -- handedness matchup indicators

**Top features by importance:** `plate_x` (12.7%), `pfx_x` (12.5%), `plate_z` (12.1%), `pfx_z` (11.8%), `distance_from_center` (11.0%)

### Hit Outcome Model (16+ features)

**Raw inputs:** `launch_speed`, `launch_angle`, `spray_angle`, `stand`, `sprint_speed`

**Engineered features:**
- `barrel_indicator` -- exit velocity >= 98 mph AND launch angle 26-30 degrees
- `sweet_spot` -- launch angle between 8-32 degrees
- `hard_hit` -- exit velocity >= 95 mph
- Batted ball type flags: `is_ground_ball`, `is_fly_ball`, `is_line_drive`, `is_popup`
- `distance_from_foul_line`, `depth_of_hit`
- `exit_velocity_centered`, `launch_angle_squared`

**Top features by importance:** `is_fly_ball` (13.4%), `sweet_spot` (10.3%), `is_ground_ball` (9.8%), `is_line_drive` (8.2%), `distance_from_foul_line` (8.2%)

---

## Machine Learning Models

### Contact Prediction -- LightGBM

- **Task:** Binary classification (contact vs. no contact)
- **Algorithm:** Gradient Boosted Decision Trees (GBDT)
- **Key hyperparameters:** `num_leaves=31`, `max_depth=12`, `learning_rate=0.05`, `feature_fraction=0.9`
- **Training:** 5-fold cross-validation
- **Result:** 83.0% accuracy, 0.805 ROC-AUC

### Hit Outcome -- XGBoost

- **Task:** 5-class classification (Out, Single, Double, Triple, Home Run)
- **Algorithm:** Multi-class gradient boosting (`multi:softprob`)
- **Key hyperparameters:** `max_depth=12`, `learning_rate=0.01`, `n_estimators=1000`, `subsample=0.8`
- **Class weights:** `{out: 1.0, single: 1.0, double: 2.0, triple: 10.0, home_run: 1.5}` to address imbalance
- **Imbalance handling:** SMOTE oversampling for rare Triple class (0.5% of data)
- **Result:** 85.4% accuracy, 0.871 weighted F1
- **xwOBA calculation:** Weighted probability output using wOBA weights (single: 0.88, double: 1.24, triple: 1.56, HR: 2.00)

---

## Backend API

FastAPI server with automatic Swagger documentation at `/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict/contact` | POST | Predict contact probability from pitch characteristics |
| `/api/predict/outcome` | POST | Predict hit outcome probabilities from batted ball metrics |
| `/api/evaluation` | GET | Retrieve model evaluation metrics, confusion matrices, feature importance |
| `/api/report/download` | GET | Download the full PDF project report |
| `/health` | GET | Server status and model availability check |

**Runtime feature engineering** is performed on each request -- raw pitch/batted ball inputs are transformed into the full feature vector before model inference. Typical response latency is 5-8ms.

Models are loaded from serialized `.pkl` files at startup, with automatic fallback to heuristic-based mock models for development without trained artifacts.

---

## Frontend Application

Five-page interactive application with a dark theme and teal accent color scheme.

### Pages

**Home** -- Project overview with navigation cards to each section.

**Contact Prediction** -- Input form for pitch characteristics with preset buttons (Fastball Middle, Breaking Ball Corner, Two-Strike). Displays an interactive strike zone plot showing pitch location color-coded by contact probability, a probability bar with adjustable classification threshold, and a contact/no-contact verdict badge.

**Hit Outcome Prediction** -- Input form for batted ball metrics with preset buttons (Barrel, Line Drive, Fly Ball, Ground Ball). Displays a baseball field spray chart, a horizontal probability distribution bar chart color-coded by outcome, the predicted outcome, and an expected wOBA value with interpretation (Below Avg / Average / Above Avg / Elite).

**Model Performance** -- Dashboard loading evaluation metrics from the backend. Displays metric cards (accuracy, precision, recall, F1, ROC-AUC), a confusion matrix heatmap, and a feature importance bar chart. Togglable between the contact and outcome models.

**Documentation** -- Project overview, pipeline explanation, API reference, and a button to download the full PDF report.

---

## Project Structure

```
.
├── backend/
│   └── main.py                  # FastAPI server, endpoints, feature engineering
├── web/
│   ├── src/
│   │   ├── api/                 # API client and TypeScript type definitions
│   │   ├── components/charts/   # StrikeZonePlot, BaseballFieldPlot, OutcomeProbs, ProbBar
│   │   └── pages/               # HomePage, ContactPage, OutcomePage, PerformancePage, DocsPage
│   ├── App.tsx                  # Router and navigation layout
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── data/
│   ├── raw/                     # Downloaded Statcast CSVs (2023-2024)
│   ├── processed/               # Train/val/test splits for each model
│   └── scripts/                 # Data download and processing utilities
├── models/
│   ├── contact_prediction/      # LightGBM model artifact + metadata
│   └── hit_outcome/             # XGBoost model artifact + metadata
├── src/
│   ├── features/                # contact_features.py, outcome_features.py
│   └── models/                  # Model training scripts
├── notebooks/                   # Jupyter notebooks for EDA and analysis
├── results/
│   ├── evaluation_results.json  # Comprehensive metrics for both models
│   └── Baseball_ML_Project_Report.pdf
├── References/
│   └── Model/                   # research.md, data_dictionary.md, academic papers
├── DESIGN.md                    # UI design system (colors, typography, components)
├── LOCAL_DEV.md                 # Development setup instructions
└── requirements.txt             # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 18+
- npm

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd web
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.

### Verify

```bash
curl http://localhost:8000/health
```

---

## Skills Demonstrated

- End-to-end ML pipeline: data acquisition, cleaning, feature engineering, model training, evaluation, and deployment
- Gradient boosting methods (XGBoost, LightGBM) with hyperparameter tuning and cross-validation
- Handling class imbalance with SMOTE and weighted loss functions
- REST API design with FastAPI, Pydantic validation, and runtime feature engineering
- Modern frontend development with React, TypeScript, and Vite
- Interactive data visualization (strike zone plots, spray charts, probability distributions)
- Form validation with React Hook Form and Zod schemas
- Responsive UI with Tailwind CSS and a custom design system
- Working with real-world sports analytics data (MLB Statcast)
