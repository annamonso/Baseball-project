# Data Dictionary - Baseball Statcast Prediction Models

## Overview

This document provides comprehensive definitions for all features used in the Contact Prediction and Hit Outcome models. All data originates from MLB Statcast unless otherwise noted.

**Data Source**: MLB Statcast via Baseball Savant and pybaseball library

**Coverage**: 2015-present (full Statcast metrics)

**Reference**: @sources.md for data acquisition details

---

## Raw Statcast Features - Pitch Data

### Pitch Location Features

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `plate_x` | Horizontal location of pitch as it crosses home plate | Float | -2.5 to 2.5 | feet | <1% | **High** (60%) |
| `plate_z` | Vertical location of pitch as it crosses home plate | Float | 0.0 to 5.0 | feet | <1% | **High** (60%) |
| `zone` | MLB defined strike zone grid (1-14) | Integer | 1-14 | categorical | <1% | Medium (8%) |

**Notes:**
- `plate_x`: 0 = center of plate, negative = catcher's left, positive = catcher's right
- `plate_z`: Height above ground; typical strike zone 1.5-3.5 feet
- `zone`: 1-9 are in strike zone, 11-14 are out of zone, 10 is undefined

### Pitch Movement Features

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `pfx_x` | Horizontal movement of pitch | Float | -25 to 25 | inches | <1% | Medium (12%) |
| `pfx_z` | Vertical movement of pitch | Float | -25 to 25 | inches | <1% | Medium (12%) |
| `release_speed` | Velocity of pitch at release | Float | 50 to 105 | mph | <1% | Medium (15%) |
| `effective_speed` | Speed accounting for extension | Float | 50 to 105 | mph | <1% | Low |
| `release_spin_rate` | Spin rate at release | Float | 0 to 3500 | rpm | 5-10% | Low |

**Notes:**
- `pfx_x`: Positive = moves toward right-handed batter (from pitcher's perspective)
- `pfx_z`: Positive = "rise" (fights gravity), negative = drops more than gravity alone
- Movement measured from release point to plate, accounting for gravity

### Pitch Release Features

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `release_pos_x` | Horizontal release point | Float | -4 to 4 | feet | <1% | Low |
| `release_pos_z` | Vertical release point | Float | 3 to 7 | feet | <1% | Low |
| `release_pos_y` | Distance from home plate at release | Float | 54 to 56 | feet | <1% | Low |
| `release_extension` | How far pitcher extends toward plate | Float | 4 to 8 | feet | <1% | Low |

**Notes:**
- Release points measured from center of rubber
- Longer extension = less reaction time for batter

### Count and Context Features

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `balls` | Number of balls in count | Integer | 0-3 | count | 0% | Medium (8%) |
| `strikes` | Number of strikes in count | Integer | 0-2 | count | 0% | Medium (8%) |
| `outs_when_up` | Number of outs when pitch thrown | Integer | 0-2 | count | 0% | Low |
| `inning` | Inning number | Integer | 1-20+ | count | 0% | Low |
| `pitch_number` | Sequential pitch number in at-bat | Integer | 1-20+ | count | 0% | Low |

### Pitch Type

| Feature Name | Description | Data Type | Values | Missing % | Importance |
|-------------|-------------|-----------|--------|-----------|------------|
| `pitch_type` | Type of pitch thrown | Categorical | FF, SL, CH, CU, SI, FC, etc. | <1% | Medium |
| `pitch_name` | Full pitch type name | String | Four-Seam Fastball, Slider, etc. | <1% | Medium |

**Common Pitch Types:**
- **FF**: Four-Seam Fastball
- **SI**: Sinker (Two-Seam Fastball)
- **FC**: Cutter
- **SL**: Slider
- **CU**: Curveball
- **CH**: Changeup
- **FS**: Splitter
- **KC**: Knuckle Curve
- **KN**: Knuckleball

### Player Features

| Feature Name | Description | Data Type | Values | Missing % | Importance |
|-------------|-------------|-----------|--------|-----------|------------|
| `pitcher` | MLBAM pitcher ID | Integer | Unique ID | 0% | Low |
| `batter` | MLBAM batter ID | Integer | Unique ID | 0% | Low |
| `stand` | Batter's stance | Categorical | R, L | 0% | Low (5%) |
| `p_throws` | Pitcher's throwing hand | Categorical | R, L | 0% | Low (5%) |

### Pitch Outcome

| Feature Name | Description | Data Type | Values | Missing % | Notes |
|-------------|-------------|-----------|--------|-----------|-------|
| `description` | Outcome of the pitch | Categorical | See below | 0% | Used to create target |
| `type` | Simplified outcome | Categorical | S, B, X | 0% | S=strike, B=ball, X=in play |
| `events` | Result of at-bat (if ended) | Categorical | See below | ~80% | Only populated on final pitch |

**Description Values:**
- `hit_into_play`: Ball was put in play (target for contact model)
- `ball`: Called ball
- `called_strike`: Called strike
- `swinging_strike`: Batter swung and missed
- `foul`: Foul ball
- `blocked_ball`: Pitch blocked by catcher
- `hit_by_pitch`: Batter hit by pitch

**Events Values (at-bat outcomes):**
- `single`, `double`, `triple`, `home_run`: Hits
- `field_out`, `strikeout`, `walk`: Non-hits
- Many others (see Hit Outcome section)

---

## Raw Statcast Features - Batted Ball Data

### Batted Ball Kinematics (Primary Features)

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `launch_speed` | Exit velocity off the bat | Float | 30-120 | mph | 5-10% | **Very High** (45%) |
| `launch_angle` | Vertical angle off bat | Float | -90 to 90 | degrees | 5-10% | **Very High** (35%) |
| `hit_distance_sc` | Projected hit distance | Float | 0-500 | feet | 10-15% | Medium |

**Notes:**
- `launch_speed`: Also called "exit velocity" (EV)
- `launch_angle`: 0° = line drive, positive = fly ball, negative = ground ball
- `hit_distance_sc`: Statcast calculated distance (not where fielder caught it)
- **Missing data more common 2015-2016** (early Statcast era)

**Optimal Ranges (Barrels from @research.md):**
- Exit velocity: >98 mph
- Launch angle: 26-30° (sweet spot for home runs)

### Batted Ball Location

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `hc_x` | Hit coordinate X (horizontal) | Float | 0-250 | feet | <5% | Low (2%) |
| `hc_y` | Hit coordinate Y (depth) | Float | 0-500 | feet | <5% | Low (2%) |

**Notes:**
- Coordinates measured from home plate
- `hc_x`: 125.42 is approximately center field
- `hc_y`: Distance down the field
- Used to calculate spray angle and depth

### Batted Ball Type

| Feature Name | Description | Data Type | Values | Missing % | Importance |
|-------------|-------------|-----------|--------|-----------|------------|
| `bb_type` | Type of batted ball | Categorical | ground_ball, fly_ball, line_drive, popup | <5% | Medium |

**Distribution (approximate):**
- Ground balls: ~45%
- Fly balls: ~35%
- Line drives: ~20%
- Pop ups: ~5%

### Hit Outcome

| Feature Name | Description | Data Type | Values | Missing % | Notes |
|-------------|-------------|-----------|--------|-----------|-------|
| `events` | Result of the batted ball | Categorical | See below | 0% | **Target variable** |

**Events Values (Hit Outcomes):**
- `single`: Single
- `double`: Double
- `triple`: Triple
- `home_run`: Home run
- `field_out`: Out on a batted ball
- `force_out`: Force out
- `grounded_into_double_play`: Double play
- `fielders_choice_out`: Fielder's choice
- `sac_fly`: Sacrifice fly
- `sac_bunt`: Sacrifice bunt

**Class Distribution (approximate):**
- Outs: ~70%
- Singles: ~15%
- Doubles: ~8%
- Triples: ~1.5% (very rare)
- Home runs: ~5.5%

### Player Features

| Feature Name | Description | Data Type | Range | Units | Missing % | Importance |
|-------------|-------------|-----------|-------|-------|-----------|------------|
| `sprint_speed` | Batter's sprint speed | Float | 24-31 | ft/sec | 30-40% | Low (3%) |

**Notes:**
- Sprint speed measured on competitive runs (home to first, etc.)
- Only available for runners, not all batters
- Critical for triple prediction (rare event)
- Missing for many players (especially pitchers in NL)

**Typical Ranges:**
- Elite speed: >29 ft/s
- Above average: 27-29 ft/s
- Average: 26-27 ft/s
- Below average: <26 ft/s

### Game Context

| Feature Name | Description | Data Type | Values/Range | Missing % | Importance |
|-------------|-------------|-----------|--------------|-----------|------------|
| `home_team` | Home team abbreviation | Categorical | NYY, BOS, LAD, etc. | 0% | Medium (5%) |
| `game_date` | Date of game | Date | YYYY-MM-DD | 0% | Low |
| `inning` | Inning number | Integer | 1-20+ | 0% | Low |

**Notes:**
- `home_team` used to join with park factors
- Park factors critical for HR/triple prediction

---

## Engineered Features - Contact Model

Reference: @research.md - "Model 1: Feature Engineering Strategy"

### Location-Based Features 

| Feature Name | Description | Formula/Logic | Data Type | Range | Importance |
|-------------|-------------|---------------|-----------|-------|------------|
| `distance_from_center` | Distance from center of strike zone | `sqrt(plate_x² + (plate_z - 2.5)²)` | Float | 0-5 | **High** |
| `in_zone` | Binary indicator if pitch in strike zone | `abs(plate_x) < 0.83 AND plate_z > 1.5 AND plate_z < 3.5` | Binary | 0 or 1 | **High** |

**Notes:**
- Strike zone center: (0, 2.5) in feet
- Simplified zone definition (actual zone varies by batter height)

### Movement Features 

| Feature Name | Description | Formula/Logic | Data Type | Range | Importance |
|-------------|-------------|---------------|-----------|-------|------------|
| `total_movement` | Total magnitude of pitch movement | `sqrt(pfx_x² + pfx_z²)` | Float | 0-35 | Medium |
| `effective_velocity` | Perceived velocity accounting for rise | `release_speed - (pfx_z / 12)` | Float | 45-110 | Medium |

**Notes:**
- `effective_velocity`: Pitches with more "rise" (positive pfx_z) appear faster
- Division by 12 is empirical scaling factor

### Count Situation Features 

| Feature Name | Description | Formula/Logic | Data Type | Range | Importance |
|-------------|-------------|---------------|-----------|-------|------------|
| `count_advantage` | Count advantage for pitcher | `strikes - balls` | Integer | -3 to 2 | Medium |
| `two_strikes` | Binary indicator for two-strike count | `strikes == 2` | Binary | 0 or 1 | Medium |
| `hitters_count` | Binary for advantageous batter count | `balls >= 2 AND strikes <= 1` | Binary | 0 or 1 | Medium |

**Hitter's Counts:**
- 2-0, 2-1, 3-0, 3-1 (batter expects fastball)

### Matchup Features 

| Feature Name | Description | Formula/Logic | Data Type | Range | Importance |
|-------------|-------------|---------------|-----------|-------|------------|
| `same_side` | Same-handed batter and pitcher | `stand == p_throws` | Binary | 0 or 1 | Low |
| `platoon_advantage` | Opposite-handed matchup (batter favor) | `stand != p_throws` | Binary | 0 or 1 | Low |

**Notes:**
- Same-side matchups harder for batters
- Example: RHP vs RHB, LHP vs LHB





