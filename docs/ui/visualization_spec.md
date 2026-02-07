# Visualization Specification

## Overview

This document specifies the UI screens and interactive visualizations for the Baseball Prediction application. The app has two main pages:

1. **Contact Prediction Page** - Predicts whether a batter will make contact
2. **Hit Outcome Page** - Predicts the type of hit after contact

---

## Screen A: Contact Prediction (`/contact`)

### Input Panel

#### Pitch Type Section
- **pitch_type**: Dropdown select
  - Options: FF (Four-Seam), SI (Sinker), FC (Cutter), SL (Slider), CU (Curveball), CH (Changeup), FS (Splitter)
  - Default: FF

#### Pitch Characteristics
- **release_speed**: Number input
  - Label: "Velocity (mph)"
  - Range: 50-105
  - Default: 92
  - Step: 0.5

- **pfx_x**: Number input
  - Label: "Horizontal Movement (in)"
  - Range: -25 to 25
  - Default: 0
  - Step: 0.5

- **pfx_z**: Number input
  - Label: "Vertical Movement (in)"
  - Range: -25 to 25
  - Default: 10
  - Step: 0.5

#### Pitch Location
- **plate_x**: Number input
  - Label: "Horizontal Location (ft)"
  - Range: -2.5 to 2.5
  - Default: 0
  - Step: 0.1

- **plate_z**: Number input
  - Label: "Vertical Location (ft)"
  - Range: 0 to 5
  - Default: 2.5
  - Step: 0.1

#### Count
- **balls**: Select (0, 1, 2, 3)
  - Default: 0

- **strikes**: Select (0, 1, 2)
  - Default: 0

#### Handedness
- **stand**: Toggle (L / R)
  - Label: "Batter"
  - Default: R

- **p_throws**: Toggle (L / R)
  - Label: "Pitcher"
  - Default: R

### Output Panel

#### Primary Output
- **Contact Probability**: Large display
  - Format: "XX.X%"
  - Color-coded: Red (<40%), Yellow (40-60%), Green (>60%)

- **Prediction Label**: Badge
  - "CONTACT" (green) or "NO CONTACT" (red)

- **Threshold Toggle**: Slider
  - Range: 0.1 to 0.9
  - Default: 0.5
  - Shows threshold line on probability display

### Visualizations

#### 1. Strike Zone Plot
- **Type**: Scatter plot / heatmap overlay
- **Dimensions**: 250x300px
- **Features**:
  - Strike zone rectangle (1.66ft wide x 2ft tall)
  - Current pitch location as large dot
  - Color = contact probability (gradient)
  - Grid overlay (3x3 zones)
  - Axis labels: feet from center

#### 2. Probability Gauge
- **Type**: Horizontal progress bar or semicircle gauge
- **Features**:
  - Fill from left (red) to right (green)
  - Threshold indicator line
  - Percentage label

#### 3. Feature Contribution Chart (Optional)
- **Type**: Horizontal bar chart
- **Features**:
  - Top 10 features
  - Positive contributions (right, green)
  - Negative contributions (left, red)
  - Feature names on y-axis

---

## Screen B: Hit Outcome (`/outcome`)

### Input Panel

#### Primary Inputs
- **launch_speed**: Number input
  - Label: "Exit Velocity (mph)"
  - Range: 30-120
  - Default: 90
  - Step: 0.5

- **launch_angle**: Number input
  - Label: "Launch Angle (°)"
  - Range: -90 to 90
  - Default: 15
  - Step: 1

- **spray_angle**: Number input (optional)
  - Label: "Spray Angle (°)"
  - Range: -45 to 45
  - Default: 0
  - Step: 1
  - Help text: "Negative = pull, Positive = oppo"

#### Context Inputs (Optional, Collapsible)
- **stand**: Toggle (L / R)
  - Label: "Batter Handedness"
  - Default: R

- **sprint_speed**: Number input
  - Label: "Sprint Speed (ft/s)"
  - Range: 24-31
  - Default: 27
  - Step: 0.1

### Output Panel

#### Primary Output
- **Predicted Outcome**: Large badge
  - Text: "OUT" / "SINGLE" / "DOUBLE" / "TRIPLE" / "HOME RUN"
  - Color-coded by outcome type

- **Expected wOBA**: Secondary display
  - Format: ".XXX"
  - Interpretation: "Below Avg" / "Avg" / "Above Avg" / "Elite"

### Visualizations

#### 1. Exit Velocity vs Launch Angle Chart
- **Type**: Scatter plot with current point overlay
- **Dimensions**: 400x300px
- **Features**:
  - Background zones showing typical outcomes
  - Current input as highlighted point
  - Barrel zone highlighted
  - Sweet spot zone highlighted
  - Color legend for outcome types

#### 2. Outcome Probabilities Bar Chart
- **Type**: Horizontal bar chart
- **Features**:
  - 5 bars (one per outcome class)
  - Percentage labels
  - Color by outcome type:
    - Out: Gray
    - Single: Light Blue
    - Double: Blue
    - Triple: Purple
    - Home Run: Orange
  - Sorted by probability (highest first)

#### 3. Probability Distribution (Alternative)
- **Type**: Stacked horizontal bar or pie chart
- **Features**:
  - All probabilities in single visualization
  - Interactive tooltips

---

## Shared Components

### Navigation
- Top navigation bar
- Two links: "Contact Prediction" and "Hit Outcome"
- Active state indicator

### Example Presets
Button group to load example scenarios:

**Contact Page Presets:**
- "Fastball Down the Middle" (plate_x=0, plate_z=2.5, release_speed=95)
- "Breaking Ball on Corner" (plate_x=0.8, plate_z=1.8, release_speed=82)
- "Two-Strike Count" (strikes=2, balls=1)

**Outcome Page Presets:**
- "Barrel" (launch_speed=105, launch_angle=28)
- "Line Drive" (launch_speed=95, launch_angle=12)
- "Fly Ball" (launch_speed=85, launch_angle=35)
- "Ground Ball" (launch_speed=80, launch_angle=-5)

### Error States
- **Invalid Input**: Inline error message below field
- **Backend Unavailable**: Full-width banner at top
- **Server Error (500)**: Toast notification with retry button

### Loading States
- Button shows spinner while request in flight
- Results panel shows skeleton loader

---

## Technical Requirements

### API Integration
- Debounce form submissions (300ms)
- Show loading state during API call
- Handle timeout (5s) gracefully

### Validation
- Client-side validation before submit
- Range validation per data_dictionary.md
- Required field validation

### Performance
- Charts should update within 100ms of data change
- API latency target: <200ms
- Total round-trip: <500ms

### Accessibility
- All inputs have labels
- Color is not the only indicator of state
- Keyboard navigation support
- ARIA attributes for charts
