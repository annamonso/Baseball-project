# Baseball Prediction Models - Research Documentation

## Project Overview
**Objective**: Develop two machine learning models using MLB Statcast data:
1. **Contact Prediction Model**: Predict whether batter will make contact before ball impact
2. **Hit Outcome Model**: Predict type of hit (1B, 2B, 3B, HR, BB, IBB, HBP, GDP, SO, SF, SH) after contact

**Data Source**: MLB Statcast database (2015-present)
**Access Method**: pybaseball Python library

---

## Model 1: Contact Prediction (Pre-Impact)

### Problem Definition
- **Input**: Pitch characteristics before reaching home plate
- **Output**: Binary classification (contact/no contact) or probability of contact
- **Challenge**: Real-time prediction with only pre-pitch features

### Selected Models & Rationale

#### Primary Model: LightGBM
- **Accuracy**: 80.5% (proven in swing decision tasks)
- **Speed**: Fastest training time among gradient boosting methods
- **Memory**: Most efficient for large Statcast datasets
- **Reference**: "Modeling Swing Probability" (Towards Data Science, 2025)

**Key Features** (in order of importance):
1. Pitch location (plate_x, plate_z, zone) - 60% importance
2. Pitch velocity (release_speed) - 15% importance
3. Pitch movement (pfx_x, pfx_z) - 12% importance
4. Count situation (balls, strikes) - 8% importance
5. Pitcher/batter handedness matchup - 5% importance

**Hyperparameters**:
```python
{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'max_depth': 12
}
```

#### Alternative Models for Comparison:

**XGBoost**
- Accuracy: 88-91% on similar tasks
- Better handling of missing values
- More robust regularization
- Reference: Northwestern University pitch prediction study (2017)

**Random Forest**
- Accuracy: 77-88%
- Best for feature importance analysis
- Less prone to overfitting
- Good baseline model

**Neural Network (MLP)**
- Architecture: 79 inputs → [80, 80] hidden → 1 output
- Captures complex non-linear relationships
- Reference: Singlearity-PA model (Baseball Prospectus, 2020)

**CatBoost**
- Best for categorical features (pitch type, zone)
- Built-in overfitting detection
- Works well with default hyperparameters

### Feature Engineering Strategy

**Temporal Features**:
- Previous pitch type and outcome
- Pitch sequence patterns (last 3 pitches)
- Time since last pitch

**Contextual Features**:
- Runners on base
- Game situation (inning, score differential)
- Pitcher fatigue (pitch count, time through order)

**Interaction Features**:
- release_speed × pfx_z (effective velocity)
- plate_x × plate_z (precise location)
- count_advantage (balls - strikes)

### Data Considerations
- **Class Imbalance**: ~70% contact rate - use weighted loss or SMOTE
- **Sample Size**: Minimum 5,000 pitches per pitcher for individual models
- **Train/Test Split**: Temporal split (train on 2015-2023, test on 2024)

---

## Model 2: Hit Outcome Prediction (Post-Contact)

### Problem Definition
- **Input**: Batted ball characteristics immediately after contact
- **Output**: Multi-class (1B, 2B, 3B, HR, BB, IBB, HBP, GDP, SO, SF, SH) with probabilities
- **Challenge**: Highly imbalanced classes (triples <2% of all batted balls)

### Selected Models

#### Primary Model: XGBoost
- **Accuracy**: 77.7% for outcome classification
- **Reference**: Tyler Burch hit classifier, Statcast analysis (2020)
- **Advantage**: Handles multi-class ordinal outcomes effectively

**Key Features** (in order of importance):
1. Exit velocity (launch_speed) - 45% importance
2. Launch angle (launch_angle) - 35% importance
3. Spray angle (horizontal direction) - 10% importance
4. Park factors by hit type - 5% importance
5. Batter sprint speed - 3% importance
6. Hit location (hc_x, hc_y) - 2% importance

**Hyperparameters**:
```python
{
    'objective': 'multi:softprob',
    'num_class': 5,
    'max_depth': 12,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': [1, 1, 2, 10, 1.5]  # Weight for rare classes
}
```

#### Alternative Models:

**Random Forest**
- Accuracy: 94% for hit/out, 70% for outcome type
- Excellent feature importance visualization
- Reference: "Using Statcast Data to Predict Hits" (Hardball Times, 2016)

**Gradient Boosted Trees (scikit-learn)**
- Good baseline performance
- Faster training than XGBoost
- Built-in CV capabilities

**GAM + k-NN (MLB's xwOBA approach)**
- k-NN for line drives and fly balls
- GAM for ground balls
- Provides smooth probability surfaces
- Reference: Official MLB Statcast expected metrics

**Ensemble Deep Neural Network**
- 62.4% accuracy on 34-class problem
- Can predict multiple attributes simultaneously
- Reference: EP2 model (Journal of Sports Analytics, 2022)

### Feature Engineering Strategy

**Physical Features**:
- exit_velocity_centered (exit_velocity - mean)
- launch_angle_squared (quadratic effect)
- barrel_indicator (optimal EV/LA combination)

**Park-Specific Features**:
- park_factor_1B, park_factor_2B, park_factor_3B, park_factor_HR
- Adjusted by batter handedness
- Source: FanGraphs park factors

**Player Features**:
- batter_sprint_speed (Statcast metric)
- batter_handedness vs field dimensions
- Career GB/FB/LD rates

### Handling Class Imbalance

**Distribution** (approximate):
- Outs: 70%
- Singles: 15%
- Doubles: 8%
- Triples: 1.5%
- Home Runs: 5.5%

**Strategies**:
1. **SMOTE** (Synthetic Minority Oversampling) for triples
2. **Class weights** in loss function
3. **Focal loss** to focus on hard-to-classify examples
4. **Hierarchical classification**: First predict contact quality, then outcome

### Expected Metrics vs Actual

**Model outputs probability distribution**:
```python
# Example for a batted ball
{
    'out': 0.30,
    'single': 0.40,
    'double': 0.20,
    'triple': 0.05,
    'home_run': 0.05
}
```

**Use probabilities to calculate**:
- Expected wOBA (xwOBA)
- Expected batting average (xBA)
- Expected slugging percentage (xSLG)

--

## Evaluation Metrics

### Contact Prediction Model
**Primary**: 
- Accuracy
- ROC-AUC
- Precision-Recall curve

**Secondary**:
- Calibration plot (predicted prob vs actual rate)
- Brier score (probability accuracy)

### Hit Outcome Model
**Primary**:
- Overall accuracy
- Per-class F1 scores
- Confusion matrix

**Secondary**:
- Macro-averaged F1 (treats all classes equally)
- Weighted F1 (accounts for class imbalance)
- Mean Absolute Error for ordinal outcomes
- Correlation of xwOBA to actual wOBA

### Business Metrics
- Improvement over naive baseline (most common class)
- Improvement over traditional sabermetrics
- Real-time prediction latency

---

## Key References

### Academic Papers
1. "Machine Learning in Baseball Analytics: Sabermetrics and Beyond" (MDPI, 2025)
2. "Application of Machine Learning Models for Baseball Outcome Prediction" (MDPI, 2024)
3. "Singlearity: Using A Neural Network to Predict PA Outcomes" (Baseball Prospectus, 2020)
4. "Using multi-class classification methods to predict baseball pitch types" (Sidle & Tran, 2018)

### Technical Implementations
1. "Modeling Swing Probability" - LightGBM implementation (Towards Data Science, 2025)
2. Tyler Burch's Hit Classifier series (2020)
3. "Using Statcast Data to Predict Hits" (Hardball Times, 2016)

---

## Success Criteria

### Minimum Viable Performance
- Contact Model: >75% accuracy (beat naive baseline of 70%)
- Outcome Model: >70% accuracy (significantly beat ~65% "always predict out")

### Target Performance
- Contact Model: >80% accuracy (match published research)
- Outcome Model: >77% accuracy (match best published models)

### Stretch Goals
- Contact Model: >85% accuracy with real-time prediction
- Outcome Model: xwOBA correlation >0.85 with actual wOBA