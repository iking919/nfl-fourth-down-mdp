
"""
Print basic statistics and inspect fourth-down plays in the
2024 NFL play-by-play dataset.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/play_by_play_2024.csv")


def main():
    print("Loading 2024 NFL play-by-play data...")

    # Load the full dataset for the initial inspection.
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Basic information about the dataset.
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Unique games: {df['game_id'].nunique():,}")

    # These are the main fields we may need for the fourth-down model.
    relevant_columns = [
        "season",
        "week",
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "down",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
        "score_differential",
        "play_type",
        "epa",
        "wp",
        "fourth_down_converted",
        "field_goal_result",
        "punt_blocked",
    ]

    # Only include columns that are present in the downloaded file.
    available = [c for c in relevant_columns if c in df.columns]

    print("\n" + "=" * 60)
    print("IMPORTANT COLUMNS")
    print("=" * 60)

    print(df[available].dtypes.to_string())

    # Check missing values before deciding how to handle them.
    # Some fields are only populated for certain types of plays.
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df[available].isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct,
    })

    print(missing_report.to_string())

    # Get a sense of which types of plays are included in the data.
    print("\n" + "=" * 60)
    print("PLAY TYPE COUNTS")
    print("=" * 60)

    print(df["play_type"].value_counts(dropna=False).to_string())

    # Check how many rows are recorded for each down.
    print("\n" + "=" * 60)
    print("DOWNS COUNTS")
    print("=" * 60)

    print(df["down"].value_counts(dropna=False).sort_index().to_string())

    # Isolate fourth-down rows for a closer look.
    # Not all of these will necessarily be actual decision opportunities.
    print("\n" + "=" * 60)
    print("FOURTH-DOWN PLAY TYPES")
    print("=" * 60)

    fourth_down = df[df["down"] == 4]

    print(f"Fourth-down rows: {len(fourth_down):,}")
    print("\nPlay types:")

    print(
        fourth_down["play_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Print a few examples so we can check how fourth-down situations
    # and their outcomes are represented in the dataset.
    print("\n" + "=" * 60)
    print("SAMPLE FOURTH-DOWN ROWS")
    print("=" * 60)

    sample_columns = [
        "game_id",
        "week",
        "posteam",
        "defteam",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
        "score_differential",
        "play_type",
        "epa",
        "wp",
        "fourth_down_converted",
        "field_goal_result",
        "punt_blocked",
    ]

    sample_columns = [c for c in sample_columns if c in df.columns]

    print(
        fourth_down[sample_columns]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
