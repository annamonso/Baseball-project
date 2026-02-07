"""
Data Download Script for Baseball Prediction Models

Downloads MLB Statcast pitch-by-pitch data using pybaseball library.
Follows best practices from sources.md for monthly downloads to avoid timeouts.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enable pybaseball caching
from pybaseball import cache
cache.enable()

from pybaseball import statcast
from tqdm import tqdm


def create_directories():
    """Create necessary data directories."""
    dirs = [
        PROJECT_ROOT / "data" / "raw" / "pitch_data",
        PROJECT_ROOT / "data" / "raw" / "batted_ball_data",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs[0], dirs[1]


def download_month(start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Download Statcast data for a specific date range with retry logic.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Number of retry attempts

    Returns:
        DataFrame with Statcast data
    """
    for attempt in range(max_retries):
        try:
            print(f"  Downloading {start_date} to {end_date}...")
            data = statcast(start_dt=start_date, end_dt=end_date)
            if data is not None and len(data) > 0:
                print(f"  Downloaded {len(data):,} pitches")
                return data
            else:
                print(f"  No data returned, retrying...")
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"  All retries exhausted for {start_date} to {end_date}")
                raise
    return pd.DataFrame()


def get_month_ranges(year: int) -> list:
    """
    Generate monthly date ranges for MLB season (April - October).

    Args:
        year: Season year

    Returns:
        List of (start_date, end_date) tuples
    """
    # MLB season typically runs from late March/early April to early November
    months = [
        (f"{year}-03-20", f"{year}-03-31"),  # Spring training / opening day
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-31"),  # Postseason
    ]
    return months


def download_season_pitches(year: int, pitch_dir: Path) -> pd.DataFrame:
    """
    Download full season pitch-by-pitch data.

    Args:
        year: Season year
        pitch_dir: Directory to save pitch data

    Returns:
        DataFrame with all pitches for the season
    """
    print(f"\n{'='*60}")
    print(f"Downloading {year} Season Pitch Data")
    print(f"{'='*60}")

    months = get_month_ranges(year)
    all_data = []

    for i, (start, end) in enumerate(tqdm(months, desc=f"Months ({year})")):
        try:
            month_data = download_month(start, end)

            if month_data is not None and len(month_data) > 0:
                # Save monthly file
                month_file = pitch_dir / f"{year}_{i+1:02d}_pitches.csv"
                month_data.to_csv(month_file, index=False)
                all_data.append(month_data)
                print(f"  Saved to {month_file.name}")

            # Be nice to the server
            time.sleep(5)

        except Exception as e:
            print(f"  Error downloading {start} to {end}: {e}")
            continue

    if all_data:
        # Combine all months
        full_season = pd.concat(all_data, ignore_index=True)

        # Save full season file
        full_file = pitch_dir / f"{year}_full_pitches.csv"
        full_season.to_csv(full_file, index=False)
        print(f"\nSaved full season: {full_file.name}")
        print(f"Total pitches: {len(full_season):,}")

        return full_season

    return pd.DataFrame()


def extract_batted_balls(pitch_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter pitch data for batted balls only.

    Args:
        pitch_data: DataFrame with all pitches

    Returns:
        DataFrame with only batted balls (hit_into_play)
    """
    # Filter for balls put in play
    batted_balls = pitch_data[
        pitch_data['description'] == 'hit_into_play'
    ].copy()

    # Remove rows with missing critical features
    critical_columns = ['launch_speed', 'launch_angle', 'events']
    initial_count = len(batted_balls)

    for col in critical_columns:
        if col in batted_balls.columns:
            batted_balls = batted_balls.dropna(subset=[col])

    final_count = len(batted_balls)
    dropped = initial_count - final_count

    print(f"  Batted balls: {initial_count:,}")
    print(f"  After removing missing critical features: {final_count:,}")
    print(f"  Dropped: {dropped:,} ({dropped/initial_count*100:.1f}%)")

    return batted_balls


def main():
    """Main function to download all data."""
    print("="*60)
    print("Baseball Prediction Models - Data Download")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    pitch_dir, batted_dir = create_directories()
    print(f"\nPitch data directory: {pitch_dir}")
    print(f"Batted ball data directory: {batted_dir}")

    # Download 2023 and 2024 seasons
    years = [2023, 2024]

    for year in years:
        # Download pitch data
        pitch_data = download_season_pitches(year, pitch_dir)

        if pitch_data is not None and len(pitch_data) > 0:
            # Extract batted balls
            print(f"\nExtracting batted balls for {year}...")
            batted_balls = extract_batted_balls(pitch_data)

            # Save batted ball data
            batted_file = batted_dir / f"{year}_batted_balls.csv"
            batted_balls.to_csv(batted_file, index=False)
            print(f"Saved batted balls: {batted_file.name}")
            print(f"Total batted balls: {len(batted_balls):,}")

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
