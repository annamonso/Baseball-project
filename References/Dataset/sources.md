# Data Sources for Baseball Prediction Models

## Primary Data Source: MLB Statcast

### Overview
**Statcast** is MLB's official tracking technology that captures detailed metrics on every pitch, hit, and defensive play across all 30 MLB stadiums. Introduced in 2015, it uses a combination of high-speed cameras and radar systems to generate granular baseball data.

**Official Source**: https://baseballsavant.mlb.com/statcast_search

---

## Data Access Methods


### pybaseball Python Library 

**Library**: `pybaseball` - Python wrapper for Statcast and other baseball databases

**GitHub**: https://github.com/jldbc/pybaseball

**Installation**:
```bash
pip install pybaseball
```

**Key Functions**:

#### 1. Statcast Pitch Data
```python
from pybaseball import statcast

# Download pitch data for date range
data = statcast(start_dt='2024-04-01', end_dt='2024-04-30')

# Returns DataFrame with ~120 columns including:
# - Pitch characteristics: release_speed, release_pos_x/y/z, pfx_x, pfx_z
# - Location: plate_x, plate_z, zone
# - Outcomes: description, events, type
# - Context: balls, strikes, outs_when_up, inning
# - Players: pitcher, batter, pitcher_name, player_name
```

**Available Data**:
- Date range: 2008-present (Statcast: 2015-present)
- ~700,000 pitches per season
- Real-time updates during season
- Historical data complete through previous day

#### 2. Statcast Batter Data
```python
from pybaseball import statcast_batter

# Get all batted balls for specific batter
batter_data = statcast_batter('2024-04-01', '2024-10-01', player_id=660271)  # Juan Soto

# Includes batted ball metrics:
# - launch_speed (exit velocity)
# - launch_angle
# - hit_distance_sc
# - hc_x, hc_y (hit coordinates)
# - bb_type (batted ball type: ground_ball, line_drive, fly_ball, popup)
```

#### 3. Statcast Pitcher Data
```python
from pybaseball import statcast_pitcher

# Get all pitches thrown by specific pitcher
pitcher_data = statcast_pitcher('2024-04-01', '2024-10-01', player_id=543037)  # Gerrit Cole
```

#### 4. Player ID Lookup
```python
from pybaseball import playerid_lookup

# Find player ID
player = playerid_lookup('judge', 'aaron')
# Returns: key_mlbam (Statcast ID), key_fangraphs, key_bbref, etc.
```

**Advantages**:
✅ Free and open source
✅ Programmatic access - fully automated
✅ No manual downloads needed
✅ Handles pagination automatically
✅ Returns clean pandas DataFrames
✅ Actively maintained (last update: 2024)
✅ Built-in caching to avoid re-downloading

**Limitations**:
⚠️ Rate limiting: ~10 requests/minute recommended
⚠️ Large date ranges take time (2-5 min per month)
⚠️ Some Statcast metrics only available 2015+
⚠️ Requires stable internet connection

**Best Practices**:
```python
import time
from pybaseball import statcast

# Download data month by month to avoid timeouts
months = [
    ('2024-04-01', '2024-04-30'),
    ('2024-05-01', '2024-05-31'),
    # etc.
]

all_data = []
for start, end in months:
    print(f"Downloading {start} to {end}...")
    data = statcast(start_dt=start, end_dt=end)
    all_data.append(data)
    time.sleep(10)  # Be nice to the server

import pandas as pd
full_season = pd.concat(all_data, ignore_index=True)
```

---

### Method 3: MLB Stats API (Direct API Access)

**Base URL**: https://statsapi.mlb.com/api/v1/

**Documentation**: https://github.com/toddrob99/MLB-StatsAPI

**Python Wrapper**:
```bash
pip install MLB-StatsAPI
```

**Capabilities**:
```python
import statsapi

# Get game data
game = statsapi.get('game', {'gamePk': 634577})

# Get play-by-play
plays = statsapi.get('game_playByPlay', {'gamePk': 634577})

# Get live game feed
live = statsapi.get('game_live', {'gamePk': 634577})
```

**Advantages**:
✅ Official MLB API
✅ Real-time game data
✅ More comprehensive than Statcast (includes lineup, substitutions, etc.)
✅ No rate limits (reasonable use)

**Limitations**:
⚠️ Does not include detailed Statcast metrics (exit velo, launch angle)
⚠️ Requires game IDs (not date ranges)
⚠️ More complex data structure (nested JSON)

**Best For**:
- Game context data
- Lineup information
- Real-time updates
- Non-Statcast historical data (pre-2015)


---

## Supplementary Data Sources

### 1. FanGraphs (Park Factors & Advanced Stats)

**URL**: https://www.fangraphs.com/

**Access Method**: 
- Manual CSV export
- `pybaseball.fangraphs` module (limited)

**Data Available**:
- Park factors by hit type (1B, 2B, 3B, HR)
- Advanced pitching metrics (FIP, xFIP, SIERA)
- Advanced hitting metrics (wOBA, wRC+, ISO)
- Player projections (Steamer, ZiPS, THE BAT)

**Example**:
```python
from pybaseball import park_factors

# Get park factors for recent seasons
pf = park_factors(2024)
# Returns: Basic, 1B, 2B, 3B, HR, SO, UIBB, GB, FB, LD park factors
```

**Use Case**: Add park factor features to hit outcome model

---

### 2. Baseball Reference (Historical Data)

**URL**: https://www.baseball-reference.com/

**Access Method**: pybaseball integration

**Data Available**:
```python
from pybaseball import batting_stats, pitching_stats

# Season-level batting statistics
batting = batting_stats(2024, qual=100)  # Min 100 PA

# Season-level pitching statistics
pitching = pitching_stats(2024, qual=50)  # Min 50 IP
```

**Use Case**: Player career statistics, historical context

---

### 3. Retrosheet (Play-by-Play Historical Data)

**URL**: https://www.retrosheet.org/

**Data Range**: 1871-present

**Format**: Event files (requires parsing)

**Access Method**:
```python
from pybaseball import get_retrosheet

# Download retrosheet data (requires custom parsing)
# Best for historical analysis pre-2015
```

**Use Case**: Training data augmentation (pre-Statcast era)

---

### 4. Savant Leaderboards

**URL**: https://baseballsavant.mlb.com/leaderboard/custom

**Access**:
```python
from pybaseball import statcast_batter_exitvelo_barrels, statcast_batter_expected_stats

# Exit velocity and barrel data
ev_data = statcast_batter_exitvelo_barrels(2024)

# Expected stats (xBA, xSLG, xwOBA)
expected = statcast_batter_expected_stats(2024)
```

**Features**:
- Expected batting average (xBA)
- Expected slugging (xSLG)
- Expected weighted on-base average (xwOBA)
- Barrel percentage
- Hard-hit percentage
- Sweet spot percentage

---

## Complete Data Acquisition Strategy

### Recommended Approach

**For Contact Prediction Model:**
```python
from pybaseball import statcast
import pandas as pd

# Download full season pitch-by-pitch data
def download_season_pitches(year):
    months = pd.date_range(f'{year}-03-01', f'{year}-11-01', freq='MS')
    
    all_data = []
    for i in range(len(months)-1):
        start = months[i].strftime('%Y-%m-%d')
        end = (months[i+1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Downloading {start} to {end}...")
        data = statcast(start_dt=start, end_dt=end)
        all_data.append(data)
        
        # Save intermediate results
        data.to_csv(f'raw/pitch_data/{year}_{months[i].month:02d}_pitches.csv', 
                    index=False)
    
    # Combine all months
    full_season = pd.concat(all_data, ignore_index=True)
    full_season.to_csv(f'raw/pitch_data/{year}_full_pitches.csv', index=False)
    
    return full_season

# Download multiple years
for year in [2023, 2024]:
    download_season_pitches(year)
```

**For Hit Outcome Model:**
```python
# Filter for batted balls only
def get_batted_balls(pitch_data):
    batted_balls = pitch_data[
        pitch_data['description'] == 'hit_into_play'
    ].copy()
    
    # Remove rows with missing critical features
    batted_balls = batted_balls.dropna(subset=[
        'launch_speed', 
        'launch_angle', 
        'events'
    ])
    
    return batted_balls

# Usage
pitches = pd.read_csv('raw/pitch_data/2024_full_pitches.csv')
batted = get_batted_balls(pitches)
batted.to_csv('raw/batted_ball_data/2024_batted_balls.csv', index=False)
```

---

## Data Quality Considerations

### Missing Data Issues

**By Year**:
- **2015-2016**: Early Statcast - some missing exit velocity/launch angle (~15%)
- **2017-present**: Complete Statcast coverage

**Common Missing Fields**:
- `launch_speed`: Missing on ~5% of batted balls (soft contact)
- `launch_angle`: Missing on ~5% of batted balls
- `hit_distance_sc`: Missing on ~10% (out of stadium, measurement failures)
- `sprint_speed`: Only available for runners (not all batters)

**Handling Strategy**:
```python
# Check missing data
def check_missing_data(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    return pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)

# Remove incomplete records or impute
df_clean = df.dropna(subset=['launch_speed', 'launch_angle', 'events'])
```

---


## Data Storage Recommendations

**Raw Data Storage**:
- Format: CSV (compressed with gzip)
- Partitioning: By year and month
- Size estimates:
  - Full season pitches: ~300 MB compressed
  - Full season batted balls: ~50 MB compressed
  - 10 years (2015-2024): ~3.5 GB total

**Database Option** (for large-scale projects):
```python
import sqlite3

# Store in SQLite for faster queries
conn = sqlite3.connect('data/statcast.db')
df.to_sql('pitches_2024', conn, if_exists='replace', index=False)

# Query
query = """
SELECT * FROM pitches_2024 
WHERE launch_speed > 100 
AND launch_angle BETWEEN 20 AND 35
"""
hard_hit = pd.read_sql(query, conn)
```

---

## API Rate Limits & Best Practices

### pybaseball
- **Recommended**: Max 10 requests/minute
- **Caching**: Enabled by default in `~/.pybaseball/cache`
- **Timeout**: 30 seconds per request

### Best Practices
```python
# Enable caching
from pybaseball import cache
cache.enable()

# Batch downloads
import time

def download_with_retry(start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = statcast(start_dt=start, end_dt=end)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries}")
                time.sleep(30)
            else:
                raise e
```

---


**Attribution**:
Always cite data sources:
```
Data provided by MLB Advanced Media, LP (Statcast).
Accessed via Baseball Savant (https://baseballsavant.mlb.com) 
and pybaseball library.
```

---

## Troubleshooting Common Issues

**Issue 1: SSL Certificate Error**
```python
# Solution
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

**Issue 2: Timeout on Large Requests**
```python
# Solution: Break into smaller date ranges
# Instead of full season, do month-by-month
```

**Issue 3: Missing Data Columns**
```python
# Check Statcast availability
if 'launch_speed' in data.columns:
    print("Statcast metrics available")
else:
    print("Pre-Statcast data or incomplete")
```


