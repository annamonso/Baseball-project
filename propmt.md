# Baseball Prediction Models - Prompt

## General Objective

Build two machine learning models for baseball prediction using MLB Statcast data:

1. **Contact Prediction Model**: Predict whether a batter will make contact with the ball BEFORE impact, using pitch characteristics available before the ball reaches the plate.

2. **Hit Outcome Model**: Predict the type of hit ((1B, 2B, 3B, HR, BB, IBB, HBP, GDP, SO, SF, SH)) AFTER contact, using batted ball characteristics measured immediately after impact.

The final deliverable is a full-stack web application that allows users to input pitch/batted ball parameters and receive real-time predictions with interactive visualizations.

---

# Pipeline

## Phase 1 - Data Ingestion

### Objective
Download and store MLB Statcast data from Baseball Savant using the pybaseball library.

### Key References
- @SOURCES.md - Complete guide to Statcast data access methods
- @SOURCES.md - Section "Complete Data Acquisition Strategy"
- @SOURCES.md - Section "Data Quality Considerations"

### Tasks

#### 1.1 Environment Setup
- Create Python virtual environment
- Install dependencies: pybaseball, pandas, numpy, jupyter, scikit-learn, xgboost, lightgbm
- Enable pybaseball caching to avoid redundant downloads

#### 1.2 Data Download Strategy

**For Contact Prediction Model:**
- Download full pitch-by-pitch data from Statcast (2023-2024 seasons)
- Use monthly downloads to avoid timeouts (reference @SOURCES.md)
- Include all pitch types and outcomes
- Target: ~700,000 pitches per season

**For Hit Outcome Model:**
- Filter pitch data for batted balls only (description == 'hit_into_play')
- Ensure launch_speed and launch_angle are present
- Remove rows with missing critical features
- Target: ~150,000 batted balls per season

#### 1.3 Data Storage
- Save raw data as CSV files partitioned by year and month
- Create full season files combining all months
- Expected storage: ~300MB per season (compressed)
- Organize in data/raw/pitch_data/ and data/raw/batted_ball_data/

#### 1.4 Data Validation
- Create validation script (data/scripts/validate_data.py)
- Verify required columns exist (see @data_dictionary.md)
- Check missing value percentages
- Validate date ranges
- Confirm reasonable value ranges (e.g., exit velocity 50-120 mph)
- Identify Statcast coverage gaps (2015-2016 have more missing data per @SOURCES.md)

### Success Criteria
-  Successfully download 2023-2024 seasons
-  <10% missing values in critical fields
-  Validation script confirms data integrity
-  All files properly organized in data/raw/ directory

---

## Phase 2 - Data Analysis & Feature Engineering

### Objective
Explore datasets, create engineered features based on @research.md recommendations, and prepare data for model training.

### Key References
- @research.md - Section  "Model 1: Contact Prediction (Pre-Impact)"
- @research.md - Section "Hit Outcome Prediction (Post-Contact)"
- @research.md - Section "Data Considerations"
- @data_dictionary.md - Feature definitions

### Tasks

#### 2.1 Exploratory Data Analysis
Create comprehensive analysis in Jupyter notebook (notebooks/01_data_exploration.ipynb)

**For Contact Prediction:**
- Visualize pitch location heatmaps (plate_x, plate_z)
- Analyze contact rate by zone (zones 1-14)
- Plot velocity distributions by pitch type
- Examine count situation effects (balls, strikes)
- Check for class imbalance (~70% contact rate expected)

**For Hit Outcome Prediction:**
- Create exit velocity vs launch angle scatter plots (colored by outcome)
- Visualize outcome distribution (check severe imbalance for triples <2%)
- Analyze park factor effects
- Plot spray angle distributions
- Examine batted ball type (ground_ball, fly_ball, line_drive) distributions

**Key Insights to Document:**
- Feature correlations (identify multicollinearity)
- Missing data patterns
- Outliers and anomalies
- Temporal trends (game evolution over seasons)

#### 2.2 Feature Engineering - Contact Model
Reference @research.md -  "Model 1: Contact Prediction (Pre-Impact)"

Create src/features/contact_features.py

**Primary Features:**
- plate_x, plate_z (location at home plate)
- distance_from_center (engineered metric)
- in_zone indicator (simplified strike zone)

**Pitch Movement Features:**
- pfx_x, pfx_z (horizontal and vertical movement)
- total_movement (magnitude of break)
- effective_velocity (perceived speed accounting for movement)

**Count Situation Features:**
- balls, strikes (raw count)
- count_advantage (strikes - balls)
- two_strikes indicator
- hitters_count indicator (3-0, 3-1, 2-0, 2-1)

**Matchup Features:**
- same_side indicator (batter and pitcher handedness)
- platoon advantage metrics

**Target Variable:**
- contact: 1 if hit_into_play, 0 otherwise

#### 2.3 Feature Engineering - Hit Outcome Prediction (Post-Contact)
Reference @research.md - "Model 2: Feature Engineering Strategy"

Create src/features/outcome_features.py

**Primary Features:**
- launch_speed (exit velocity) - 45% importance
- launch_angle - 35% importance
- spray_angle (horizontal direction) - 10% importance

**Physical Features:**
- exit_velocity_centered (normalized)
- launch_angle_squared (quadratic effect per @research.md)
- barrel_indicator (optimal EV/LA combination)

**Park-Specific Features:**
- park_factor_1B, park_factor_2B, park_factor_3B, park_factor_HR
- Adjusted by batter handedness
- Source from FanGraphs data

**Player Features :**
- sprint_speed (for triple prediction)
- batter_handedness
- career GB/FB/LD rates

**Spatial Features:**
- hc_x, hc_y (hit coordinates)
- distance_from_foul_line
- depth_of_hit

**Target Variable:**
- events: categorical (Single, Double, Triple, Home run, Out on a batted ball,Force out,Double play,Fielder's choice,Sacrifice fly,Sacrifice bunt)


#### 2.4 Train/Validation/Test Split

**Strategy:** Temporal split (not random) to avoid data leakage

**Recommended Split:**
- Training: 2023 season + first half of 2024 (60%)
- Validation: Second half of 2024 through August (20%)
- Test: September-October 2024 (20%)

**Save processed datasets:**
- data/processed/contact_prediction/train.csv
- data/processed/contact_prediction/validation.csv
- data/processed/contact_prediction/test.csv
- data/processed/hit_outcome/train.csv
- data/processed/hit_outcome/validation.csv
- data/processed/hit_outcome/test.csv

#### 2.6 Feature Documentation
Update @data_dictionary.md with:
- Feature name and description
- Data type and range
- Derivation formula (for engineered features)
- Importance ranking (from @research.md)
- Missing value strategy

### Success Criteria
-  All features engineered per @research.md specifications
-  Missing values handled appropriately (<5% in critical features)
-  Class imbalance addressed for outcome model
-  Train/val/test splits created with temporal separation
-  Complete feature documentation in @data_dictionary.md

---

## Phase 3 - Model Development

### Objective
Train, tune, and validate machine learning models using best practices from @research.md.

### Key References
- @research.md - Section "Model 1: Selected Models"
- @research.md - Section "Model 2: Selected Models"
- @research.md - Hyperparameters for both models

### Tasks

#### 3.1 Baseline Models
Establish performance benchmarks before tuning.

Create notebooks/02_baseline_models.ipynb

**Contact Prediction Baseline:**
- Naive baseline: Always predict "contact" (majority class ~70%)
- Logistic Regression (simple linear model)
- Random Forest with default parameters
- Target to beat: >75% accuracy

**Hit Outcome Baseline:**
- Naive baseline: Always predict "out" (majority class ~70%)
- Logistic Regression (multi-class)
- Random Forest with default parameters
- Target to beat: >65% accuracy

#### 3.2 Primary Model - Contact Prediction (LightGBM)
Reference @research.md - "Model 1: Primary Model: LightGBM"

Create src/models/contact_prediction/lightgbm_model.py

**Implementation Steps:**
1. Load training data with engineered features
2. Initialize LGBMClassifier with recommended hyperparameters from @research.md
3. Train on full training set
4. Evaluate on validation set
5. Perform hyperparameter tuning with GridSearchCV 
6. Retrain with best parameters
7. Generate predictions on test set
8. Save trained model as models/contact_prediction/lightgbm_model.pkl

**Hyperparameter Tuning Grid (from @research.md):**
- num_leaves: [31, 50, 100]
- max_depth: [8, 12, 15, -1]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [100, 500, 1000]
- min_child_samples: [20, 50, 100]

**Validation Strategy:**
- 5-fold cross-validation on training set
- Monitor both training and validation metrics
- Check for overfitting (gap between train/val accuracy)

#### 3.3 Alternative Models - Contact Prediction

Create notebooks/03_contact_model_comparison.ipynb

**XGBoost (for comparison):**
- Expected accuracy: 88-91% per @research.md
- Better handling of missing values
- More robust regularization
- Compare performance to LightGBM

**Random Forest:**
- Good baseline
- Feature importance analysis
- Less prone to overfitting

**Neural Network (optional stretch goal):**
- Architecture from @research.md: 79 inputs → [80, 80] hidden → 1 output
- Only if time permits and baseline models are solid

#### 3.4 Primary Model - Hit Outcome (XGBoost)
Reference @research.md - "Model 2: Primary Model: XGBoost"

Create src/models/hit_outcome/xgboost_model.py


**Implementation Steps:**
1. Load batted ball data with engineered features
2. Apply SMOTE for class imbalance (especially triples)
3. Initialize XGBClassifier with multi:softprob objective
4. Set class weights inversely proportional to frequency
5. Train and validate
6. Hyperparameter tuning
7. Evaluate on test set
8. Save model as models/hit_outcome/xgboost_model.pkl

**Hyperparameter Tuning Grid (from @research.md):**
- max_depth: [8, 12, 15]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [500, 1000, 1500]
- subsample: [0.8, 0.9, 1.0]
- colsample_bytree: [0.8, 0.9, 1.0]

**Class Weight Strategy (from @research.md):**
- out: 1.0 (baseline)
- single: 1.0
- double: 2.0
- triple: 10.0 (very rare, needs heavy weighting)
- home_run: 1.5

#### 3.5 Alternative Models - Hit Outcome

Create notebooks/04_outcome_model_comparison.ipynb

**Random Forest:**
- Expected ~70% accuracy for outcome type
- Best for feature importance visualization
- Compare to XGBoost

**Gradient Boosted Trees (scikit-learn):**
- Faster training than XGBoost
- Good baseline

#### 3.6 Model Persistence
Save all trained models with metadata:
- models/contact_prediction/lightgbm_model.pkl
- models/contact_prediction/xgboost_model.pkl
- models/contact_prediction/random_forest_model.pkl
- models/hit_outcome/xgboost_model.pkl
- models/hit_outcome/random_forest_model.pkl

Include metadata files (JSON):
- Training date
- Feature list (exact order)
- Hyperparameters used
- Performance metrics
- Library versions

#### 3.7 Feature Importance Analysis
Extract and save feature importance from tree-based models

Create notebooks/05_feature_importance.ipynb

Compare to expected importance from @research.md:
- **Contact model**: plate_x/plate_z should dominate (60%)
- **Outcome model**: launch_speed/launch_angle should dominate (80%)

Document any surprises or deviations in docs/feature_importance_analysis.md

### Success Criteria
- ✓ Contact model achieves >80% accuracy (target from @research.md)
- ✓ Hit outcome model achieves >77% accuracy (target from @research.md)
- ✓ All models outperform naive baselines significantly
- ✓ Feature importance aligns with @research.md expectations
- ✓ Models properly saved and versioned
- ✓ No severe overfitting (train/test gap <5%)

---

## Phase 4 - Model Performance Evaluation

### Objective
Comprehensive evaluation of model performance using metrics from @research.md and error analysis.

### Key References
- @research.md - Section "Evaluation Metrics"

### Tasks

#### 4.1 Classification Metrics

Create notebooks/06_model_evaluation.ipynb

**For Contact Prediction Model:**

Primary metrics (from @research.md):
- Overall accuracy
- ROC-AUC score
- Precision-Recall curve
- F1 score

Secondary metrics:
- Calibration plot (predicted probability vs actual contact rate)
- Brier score (probability accuracy)
- Log loss

Generate classification report showing:
- Precision, Recall, F1 for both classes (contact/no-contact)
- Support (number of samples per class)

**For Hit Outcome Model:**

Primary metrics (from @research.md):
- Overall accuracy
- Per-class F1 scores (out, single, double, triple, HR)
- Confusion matrix (5x5)

Secondary metrics:
- Macro-averaged F1 (treats all classes equally)
- Weighted F1 (accounts for class imbalance)
- Mean Absolute Error for ordinal outcomes
- Correlation of xwOBA to actual wOBA

Business metrics:
- Improvement over naive baseline (most common class)
- Per-class precision and recall

#### 4.2 Confusion Matrix Analysis

**Contact Model:**
Create 2x2 confusion matrix showing:
- True Positives: Correctly predicted contact
- True Negatives: Correctly predicted no contact
- False Positives: Predicted contact but missed
- False Negatives: Predicted no contact but ball was hit

Analyze patterns:
- Which pitch types cause most false positives?
- What zones lead to false negatives?
- Count situations with highest error rates

**Hit Outcome Model:**
Create 5x5 confusion matrix for all outcome types

Expected challenges (from @research.md):
- Triples extremely difficult to predict (will have low recall)
- Confusion between doubles and singles likely
- Home runs vs fly ball outs challenging (similar launch conditions)

Visualize as heatmap with annotations
Save as results/confusion_matrices.png

#### 4.4 Error Analysis

Create notebooks/07_error_analysis.ipynb

**Systematic Misclassification Patterns:**

For Contact Model:
- Identify pitches with high confidence but wrong prediction
- Analyze by pitch type, zone, count
- Check if errors cluster in certain game situations
- Examine borderline pitches (close to strike zone edge)

For Hit Outcome Model:
- Find batted balls with highest prediction error
- Examine triples that were missed (expected per @research.md)
- Check if specific parks cause more errors
- Analyze near-barrel scenarios (EV ~95-100, LA ~25-30)

**Confidence Calibration:**
- Plot predicted probabilities vs actual outcomes
- Check if model is over/under-confident
- Ideal: 70% predicted probability → 70% actual rate

Save analysis as results/error_analysis.csv

#### 4.5 Model Comparison

**Contact Model:**
Compare LightGBM vs XGBoost vs Random Forest:
- Accuracy
- Training time
- Inference speed
- Model size
- Feature importance stability

**Outcome Model:**
Compare XGBoost vs Random Forest vs baseline:
- Overall accuracy
- Per-class F1 scores
- Handling of rare classes (triples)
- Computational efficiency

Create comparison table and recommend final model for production
Save as docs/model_comparison.md

#### 4.6 Probability Analysis

**Contact Model:**
- Distribution of predicted contact probabilities
- Separate by actual outcome (did contact occur?)
- Identify optimal threshold (default 0.5 may not be best)

**Outcome Model:**
- Examine probability distributions across all 5 classes
- Calculate expected wOBA using probabilities
- Compare xwOBA (expected) to actual wOBA
- Target: correlation >0.85 per @research.md


---

## Phase 5 - Visualization (Stitch MCP Server)

### Objective
Build an interactive visualization layer for model inputs and predictions, aligned with the Stitch UI design system, using Stitch MCP for design extraction and UI scaffolding where helpful.

### Key References
- @research.md, sections on feature importance and evaluation metrics
- @data_dictionary.md, feature definitions and ranges
- ~/.agents/skills/design-md/SKILL.md (semantic design system generation)
- Stitch MCP tools (project, screens, HTML export)

---

### Strategy, Stitch first, local UI folder second

You have two valid sources of truth for UI:

1) Stitch MCP (preferred)  
Use Stitch as the canonical design source. Extract theme, tokens, and screen HTML directly via MCP. This avoids manual syncing.

2) Downloaded Stitch UI folder (fallback or offline)  
Use only if MCP cannot fetch HTML, or if you need static assets, or want to diff generated code against a known export.

Rule: if MCP can fetch the screen HTML successfully, the downloaded folder is not required for design extraction.

---
@ 
### Tasks

#### 5.1 Generate design system from Stitch

Goal: create DESIGN.md as the source of truth for UI tokens.

Steps (via Claude Code + Stitch MCP):
- Use mcp__stitch__list_projects to find the Stitch project called Baseball Analytics Landing Page
- Use mcp__stitch__list_screens to identify the primary screen (named Baseball Analytics Landing Page)
- Use mcp__stitch__get_screen to pull screen details and obtain HTML export URLs if present
- Generate DESIGN.md in repo root following the design-md runbook:
  - Color palette and semantic tokens (primary, surface, text, danger, etc)
  - Typography scale and font families
  - Spacing, radii, shadows
  - Component patterns (buttons, inputs, cards, chart containers)

Deliverables:
- DESIGN.md at repo root

Success Criteria:
- DESIGN.md contains hex codes and semantic tokens
- Sections include at least: Colors, Typography, Spacing, Radius, Shadows, Components

---

#### 5.2 Define visualization requirements

Create docs/ui/visualization_spec.md with screens and interactions.

Screen A, Contact Prediction:
- Input panel:
  - pitch_type, release_speed, pfx_x, pfx_z
  - plate_x, plate_z
  - balls, strikes
  - batter and pitcher handedness
- Output panel:
  - predicted contact probability (0 to 1)
  - threshold toggle (default 0.5)
  - explanation: top feature contributions (global or local)

Visuals:
- Strike zone plot with (plate_x, plate_z) overlay
- Probability gauge or horizontal bar
- Feature contribution bar chart (top 10)

Screen B, Hit Outcome:
- Input panel:
  - launch_speed, launch_angle, spray_angle
  - optional: park, batter handedness, sprint_speed
- Output panel:
  - class probabilities for (out, 1B, 2B, 3B, HR)
  - predicted outcome (argmax)
  - derived xwOBA (optional)

Visuals:
- EV vs LA scatter point on launch angle chart
- Probability distribution bar chart
- Park factor note (if enabled)

Shared:
- Example input presets
- Validation and range hints from @data_dictionary.md
- Latency target: <200 ms for prediction endpoint, excluding UI render

---

#### 5.3 Build frontend visualization components

Recommended stack:
- Vite + React + TypeScript
- Recharts for charts
- React Hook Form + Zod for validation
- Tailwind tokens aligned with DESIGN.md

Create components:
- web/src/components/charts/StrikeZonePlot.tsx
- web/src/components/charts/ProbBar.tsx
- web/src/components/charts/OutcomeProbs.tsx
- web/src/components/charts/EVLAPoint.tsx
- web/src/components/forms/ContactForm.tsx
- web/src/components/forms/OutcomeForm.tsx

Input validation rules:
- enforce min and max ranges from @data_dictionary.md
- inline error messages
- consistent units (mph, degrees)

Success Criteria:
- Forms work with mocked API responses
- Charts update reactively when predictions change
- Visual style matches DESIGN.md tokens

---

#### 5.4 Connect UI to prediction endpoints

Define API contracts (implementation in Phase 6):

POST /api/predict/contact
- request: contact model feature payload
- response:
  {
    prob_contact: number,
    threshold: number,
    predicted: 0 | 1,
    top_features?: Array<{ name: string, contribution: number }>
  }

POST /api/predict/outcome
- request: outcome model feature payload
- response:
  {
    probs: { out: number, single: number, double: number, triple: number, home_run: number },
    predicted: string,
    xwoba?: number
  }

Frontend integration:
- web/src/api/client.ts (fetch wrapper)
- web/src/api/predict.ts (typed API calls)

Success Criteria:
- Full request and response loop works locally
- Errors handled gracefully (400 validation, 500 server)


---
### Success Criteria

- UI has two functional pages, Contact and Outcome
- Each page has inputs, prediction output, and at least two interactive charts
- Styling is consistent with DESIGN.md

---


# Phase 6: Full-Stack Local Deployment (React + FastAPI)

---

## Objective

Stand up the entire application locally (frontend + backend) so I can open a local URL, enter pitch or batted-ball inputs, and get real-time predictions plus interactive visualizations. Use React for the web app, and follow the Stitch design system already defined in `DESIGN.md`. Use the Stitch Skills repo, specifically the `react-components` skill, to generate consistent UI components aligned with Stitch.

---

## Scope

Deliver a working local development setup with:

- A **React + TypeScript** frontend with two pages: Contact Prediction and Hit Outcome
- A **Python backend API** that loads trained models and returns predictions
- **End-to-end wiring** between UI forms, charts, and API responses
- **Local run instructions** and a single "start everything locally" workflow (without providing literal commands)

---

## Required Inputs and Existing Assets

Assume the repo already contains (or will contain) these from earlier phases:

- `DESIGN.md` (design tokens and component patterns)
- `data_dictionary.md` (feature definitions and valid ranges)
- **Trained model artifacts and metadata:**
  - `models/contact_prediction/lightgbm_model.pkl`
  - `models/hit_outcome/xgboost_model.pkl`
  - Metadata JSON files listing exact feature order and training versions
- **Feature engineering modules:**
  - `src/features/contact_features.py`
  - `src/features/outcome_features.py`
- **API contracts already agreed:**
  - `POST /api/predict/contact`
  - `POST /api/predict/outcome`

If any of these are missing, create placeholders and document clearly what needs to be added later, but keep the local app runnable with mocked outputs.

---

## Architecture Requirements

### Frontend

- **React + TypeScript**
- Form handling with **React Hook Form** and validation with **Zod**
- Charting with **Recharts**
- Styling with **Tailwind**, using tokens derived from `DESIGN.md`
- **Two routes/pages:**
  - `/contact` for Contact Prediction
  - `/outcome` for Hit Outcome
- Shared layout and navigation (simple top nav is fine)
- **Robust client-side validation** using ranges from `data_dictionary.md`
- **Friendly error states:**
  - invalid inputs
  - backend unavailable
  - backend 400 validation errors
  - backend 500 errors
- **Performance target:** UI interactions should feel instant, and API calls should be fast on localhost

### Backend

- **Python inference server** (FastAPI recommended)
- Loads model files on startup
- Validates request payloads with strict schemas
- Reconstructs feature vectors in the exact order expected by the models (use metadata JSON)
- Returns predictions exactly matching the contract
- Includes optional "explainability" fields if available:
  - contact endpoint: `top_features` contributions if you can compute them, otherwise omit
- Includes a basic health endpoint for local debugging

### Local Networking

- Frontend should call backend locally via a single base URL
- Use a development proxy or CORS configuration so requests work seamlessly in local dev
- Provide clear environment variable names for API base URL

---

## Use Stitch Skills (react-components)

Use the Stitch Skills repository, specifically `react-components`, to generate:

- Buttons, inputs, selects, toggles
- Cards and sections
- Page layout primitives
- Error banners and helper text
- Chart container styling wrappers

### Rules:

- The Stitch-generated components must match `DESIGN.md` tokens (colors, spacing, typography, radii, shadows).
- Prefer composing the app from Stitch components rather than custom styling.
- Keep component APIs consistent across forms (same prop naming, same validation display patterns).
- Do not embed business logic inside UI primitives, keep logic in page components or hooks.

### Deliverable expectation:

A small, reusable component library folder in the frontend, generated or scaffolded via Stitch skills, then used across both pages.

---

## Implementation Tasks

### 6.1 Repo Structure and Local App Layout

Create or finalize a clean monorepo-like layout:

- `backend/` for the API server
- `web/` for the React app
- shared docs and model artifacts in top-level folders

Add documentation that explains:

- what runs where
- where models are stored
- where feature order is defined
- how to change the API base URL

### 6.2 Backend API Implementation

Implement:

- **`POST /api/predict/contact`**
  - Inputs: pre-impact features from the spec
  - Output: `prob_contact`, `threshold`, `predicted`, and optional `top_features`
- **`POST /api/predict/outcome`**
  - Inputs: post-contact features from the spec
  - Output: `probs` dict, `predicted`, and optional `xwoba`

**Key requirements:**

- Validate feature ranges when possible (use `data_dictionary.md`)
- Handle missing optional fields gracefully
- Ensure probability outputs sum to 1 for outcome endpoint
- Provide deterministic behavior for the same input
- Log a compact request id and timing per request for local debugging

### 6.3 Frontend Pages and Interactions

Implement two pages with the UI and visuals described in Phase 5:

#### Contact page

- Form inputs for pitch, location, movement, count, handedness
- Strike zone plot overlay of `plate_x`, `plate_z`
- Probability bar or gauge
- Optional top feature contributions chart
- Threshold toggle that affects predicted label

#### Outcome page

- Form inputs for `launch_speed`, `launch_angle`, `spray_angle`, plus optional context fields
- EV vs LA visualization, single point overlay
- Outcome probabilities bar chart
- Predicted outcome display, optional xwOBA display

### 6.4 API Client Layer

Add a typed API layer in the frontend:

- A fetch wrapper with:
  - timeouts
  - JSON parsing safeguards
  - consistent error normalization
- Typed request/response models matching backend schemas
- A single place to configure API base URL

### 6.5 Local Run Experience

Provide a short `LOCAL_DEV.md` that explains how to:

- start the backend locally
- start the frontend locally
- verify the system (health endpoint and a sample prediction through the UI)
- troubleshoot common issues (ports in use, missing models, CORS/proxy issues)

Avoid listing literal shell commands, describe the steps and expected outcomes instead.

### 6.6 Quality Gates

Before declaring done, ensure:

- Both pages render and are navigable
- Submitting each form calls the correct endpoint and updates charts
- Validation blocks invalid ranges
- Backend returns correct JSON shapes
- Errors render clearly and do not break the app
- Linting and type-checking pass (where applicable)

---
