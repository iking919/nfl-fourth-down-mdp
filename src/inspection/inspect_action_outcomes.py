
"""
Inspect the outcome variables available for each fourth-down action.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/fourth_down_decisions_2024.csv")

# Candidate fields that may help describe what happened after each action.
CANDIDATE_COLUMNS = [
    "game_id",
    "play_id",
    "action",
    "play_type",
    "posteam",
    "defteam",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "yards_gained",
    "fourth_down_converted",
    "first_down",
    "touchdown",
    "field_goal_result",
    "kick_distance",
    "punt_blocked",
    "punt_inside_twenty",
    "punt_in_endzone",
    "punt_out_of_bounds",
    "punt_downed",
    "punt_fair_catch",
    "return_yards",
    "return_touchdown",
    "fumble_lost",
    "interception",
    "posteam_score_post",
    "defteam_score_post",
    "drive",
    "fixed_drive",
    "fixed_drive_result",
    "drive_end_yard_line",
    "drive_end_transition",
    "epa",
    "wp",
    "wpa",
    "penalty",
    "penalty_type",
    "desc",
]


def main():
    # Load only the fields we're interested in.
    available_columns = pd.read_csv(DATA_PATH, nrows=0).columns
    columns = [
        col for col in CANDIDATE_COLUMNS
        if col in available_columns
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        low_memory=False,
    )

    print(f"Loaded {len(df):,} decisions.")
    print(f"Available candidate columns: {len(columns)}")

    print("\n" + "=" * 60)
    print("AVAILABLE OUTCOME VARIABLES")
    print("=" * 60)

    outcome_columns = [
        col for col in columns
        if col not in {
            "game_id", "play_id", "action", "play_type",
            "posteam", "defteam", "ydstogo", "yardline_100",
            "game_seconds_remaining", "score_differential",
        }
    ]

    print(df[outcome_columns].dtypes.to_string())

    print("\n" + "=" * 60)
    print("MISSING OUTCOME VALUES")
    print("=" * 60)

    missing = df[outcome_columns].isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    report = pd.DataFrame({
        "missing": missing,
        "missing_pct": missing_pct,
    })

    print(report.to_string())

    print("\n" + "=" * 60)
    print("OUTCOME DATA BY ACTION")
    print("=" * 60)

    for action in ["go", "field_goal", "punt"]:
        subset = df[df["action"] == action]

        print(f"\n{action.upper()} ({len(subset):,} plays)")

        # Print a few fields that may help describe the result.
        relevant = [
            col for col in [
                "yards_gained",
                "fourth_down_converted",
                "first_down",
                "touchdown",
                "field_goal_result",
                "kick_distance",
                "punt_blocked",
                "return_yards",
                "return_touchdown",
                "fumble_lost",
                "interception",
                "posteam_score_post",
                "defteam_score_post",
                "drive_end_yard_line",
                "drive_end_transition",
                "epa",
                "wp",
                "wpa",
            ]
            if col in subset.columns
        ]

        print(subset[relevant].head(8).to_string(index=False))

    # Keep the descriptions available for checking unusual plays.
    if "desc" in df.columns:
        print("\n" + "=" * 60)
        print("SAMPLE PLAY DESCRIPTIONS")
        print("=" * 60)

        for action in ["go", "field_goal", "punt"]:
            print(f"\n{action.upper()}")
            samples = df.loc[df["action"] == action, "desc"].head(3)

            for description in samples:
                print("-", description)


if __name__ == "__main__":
    main()
