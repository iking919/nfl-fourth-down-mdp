
from pathlib import Path

import pandas as pd


# Locate the project directory and raw play-by-play data.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "play_by_play_2024.csv"


COLUMNS = [
    "game_id",
    "play_id",
    "qtr",
    "game_seconds_remaining",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "posteam_score",
    "defteam_score",
    "posteam_score_post",
    "defteam_score_post",
]


def main():
    print("Loading play-by-play data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")

    # Only load the columns needed for these checks.
    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda col: col in COLUMNS,
        low_memory=False,
    )

    print(f"Loaded {len(df):,} plays.")

    # Check which expected columns are available.
    print("\n" + "=" * 75)
    print("AVAILABLE COLUMNS")
    print("=" * 75)

    for column in COLUMNS:
        print(f"{column:30} {'Found' if column in df.columns else 'MISSING'}")

    required = [column for column in COLUMNS if column not in df.columns]
    if required:
        raise ValueError(f"Missing required columns: {required}")

    # Check missing values in the fields used to define the state.
    print("\n" + "=" * 75)
    print("MISSING VALUES")
    print("=" * 75)

    print(df[COLUMNS].isna().sum().to_string())

    # Compare the reported score differential with the score columns.
    print("\n" + "=" * 75)
    print("SCORE DIFFERENTIAL CHECK")
    print("=" * 75)

    score_rows = df.dropna(
        subset=["score_differential", "posteam_score", "defteam_score"]
    ).copy()

    score_rows["calculated_diff"] = (
        score_rows["posteam_score"] - score_rows["defteam_score"]
    )

    score_rows["score_diff_match"] = (
        score_rows["score_differential"] == score_rows["calculated_diff"]
    )

    print(
        "Matches posteam_score - defteam_score:",
        f"{score_rows['score_diff_match'].mean():.2%}",
    )

    print("\nExamples where the values do not match:")
    mismatches = score_rows.loc[
        ~score_rows["score_diff_match"],
        [
            "game_id",
            "play_id",
            "posteam",
            "score_differential",
            "posteam_score",
            "defteam_score",
            "calculated_diff",
        ],
    ]

    if mismatches.empty:
        print("No mismatches found.")
    else:
        print(mismatches.head(10).to_string(index=False))

    # Display representative plays from both teams' perspectives.
    print("\n" + "=" * 75)
    print("SAMPLE PLAYS BY POSSESSING TEAM")
    print("=" * 75)

    sample_columns = [
        "game_id",
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "posteam",
        "defteam",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "posteam_score",
        "defteam_score",
    ]

    scrimmage = df.dropna(
        subset=["posteam", "down", "yardline_100"]
    ).copy()

    teams = scrimmage["posteam"].dropna().unique()

    for team in teams[:2]:
        print(f"\nTeam: {team}")

        sample = (
            scrimmage[scrimmage["posteam"] == team]
            .head(5)[sample_columns]
        )

        print(sample.to_string(index=False))

    # Inspect what happens to field position and score when possession changes.
    print("\n" + "=" * 75)
    print("POSSESSION CHANGE EXAMPLES")
    print("=" * 75)

    scrimmage = scrimmage.sort_values(["game_id", "play_id"]).copy()

    scrimmage["next_posteam"] = (
        scrimmage.groupby("game_id")["posteam"].shift(-1)
    )

    scrimmage["next_yardline_100"] = (
        scrimmage.groupby("game_id")["yardline_100"].shift(-1)
    )

    scrimmage["next_score_differential"] = (
        scrimmage.groupby("game_id")["score_differential"].shift(-1)
    )

    possession_changes = scrimmage[
        scrimmage["next_posteam"].notna()
        & (scrimmage["posteam"] != scrimmage["next_posteam"])
    ]

    transition_columns = [
        "game_id",
        "play_id",
        "posteam",
        "yardline_100",
        "score_differential",
        "next_posteam",
        "next_yardline_100",
        "next_score_differential",
    ]

    print(
        possession_changes[transition_columns]
        .head(15)
        .to_string(index=False)
    )

    # Show how the remaining game clock is represented.
    print("\n" + "=" * 75)
    print("GAME CLOCK CHECK")
    print("=" * 75)

    print("Quarter values:")
    print(df["qtr"].value_counts(dropna=False).sort_index().to_string())

    print("\nGame seconds remaining:")
    print(df["game_seconds_remaining"].describe().to_string())

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
