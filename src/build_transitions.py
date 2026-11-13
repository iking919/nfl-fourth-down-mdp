
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "play_by_play_2024.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "game_transitions_2024.csv"


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
    "play_type",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "posteam_score",
    "defteam_score",
    "score_differential",
]


def main():
    print("Loading play-by-play data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda col: col in COLUMNS,
        low_memory=False,
    )

    missing = [column for column in COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df):,} plays.")

    # Keep offensive scrimmage plays with a usable game state and score.
    scrimmage = df.dropna(
        subset=[
            "game_id",
            "play_id",
            "posteam",
            "defteam",
            "down",
            "ydstogo",
            "yardline_100",
            "game_seconds_remaining",
            "home_team",
            "away_team",
            "posteam_score",
            "defteam_score",
            "score_differential",
        ]
    ).copy()

    # Exclude nullified plays; their effects appear in the next valid state.
    scrimmage = scrimmage[scrimmage["play_type"] != "no_play"].copy()

    scrimmage = scrimmage.sort_values(
        ["game_id", "play_id"]
    ).reset_index(drop=True)

    print(f"Scrimmage plays with complete state information: {len(scrimmage):,}")

    # The offense-relative score difference should match these score fields.
    offense_score_diff = (
        scrimmage["posteam_score"] - scrimmage["defteam_score"]
    )

    if not offense_score_diff.equals(scrimmage["score_differential"]):
        if not offense_score_diff.eq(scrimmage["score_differential"]).all():
            raise ValueError(
                "The offense-relative score fields do not match "
                "score_differential."
            )

    # Convert the score difference to a consistent home-team perspective.
    home_possession = scrimmage["posteam"] == scrimmage["home_team"]
    away_possession = scrimmage["posteam"] == scrimmage["away_team"]

    if not (home_possession | away_possession).all():
        raise ValueError(
            "Found scrimmage plays where the offense is neither "
            "the home nor away team."
        )

    scrimmage["score_diff_home"] = (
        offense_score_diff.where(home_possession, -offense_score_diff)
    )

    # Identify the next offensive scrimmage state within each game.
    next_columns = [
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "posteam",
        "defteam",
        "down",
        "ydstogo",
        "yardline_100",
        "score_diff_home",
    ]

    for column in next_columns:
        scrimmage[f"next_{column}"] = (
            scrimmage.groupby("game_id")[column].shift(-1)
        )

    # Mark the last observed scrimmage state in each game.
    scrimmage["terminal"] = scrimmage["next_play_id"].isna()

    # The home_score and away_score columns contain final game scores.
    final_scores = (
        df.dropna(subset=["home_score", "away_score"])
        .sort_values(["game_id", "play_id"])
        .groupby("game_id")
        .tail(1)
        .set_index("game_id")
    )

    final_margin = (
        final_scores["home_score"] - final_scores["away_score"]
    )

    scrimmage["final_score_diff_home"] = (
        scrimmage["game_id"].map(final_margin)
    )

    # Use the next state's score for normal transitions and the final
    # game margin for terminal transitions.
    scrimmage["next_score_diff_home"] = scrimmage[
        "next_score_diff_home"
    ].fillna(scrimmage["final_score_diff_home"])

    # Reward is the change in home-team score differential.
    scrimmage["reward_home"] = (
        scrimmage["next_score_diff_home"]
        - scrimmage["score_diff_home"]
    )

    if scrimmage["reward_home"].isna().any():
        raise ValueError("Some transitions have missing rewards.")

    # Preserve the original play type and label fourth-down actions.
    scrimmage["fourth_down_action"] = pd.NA

    fourth_down = scrimmage["down"] == 4

    action_map = {
        "run": "go",
        "pass": "go",
        "field_goal": "field_goal",
        "punt": "punt",
    }

    scrimmage.loc[fourth_down, "fourth_down_action"] = (
        scrimmage.loc[fourth_down, "play_type"].map(action_map)
    )

    scrimmage["is_fourth_down_decision"] = (
        fourth_down
        & scrimmage["fourth_down_action"].notna()
    )

    # Keep the fields needed for the first transition dataset.
    output_columns = [
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "home_team",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "score_diff_home",
        "play_type",
        "fourth_down_action",
        "is_fourth_down_decision",
        "next_play_id",
        "next_qtr",
        "next_game_seconds_remaining",
        "next_posteam",
        "next_defteam",
        "next_down",
        "next_ydstogo",
        "next_yardline_100",
        "next_score_diff_home",
        "reward_home",
        "terminal",
    ]

    transitions = scrimmage[output_columns].copy()

    # Save the transition dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 75)
    print("TRANSITION DATASET SUMMARY")
    print("=" * 75)

    print(f"Total transitions: {len(transitions):,}")
    print(
        "Fourth-down decisions:",
        f"{transitions['is_fourth_down_decision'].sum():,}",
    )
    print(f"Terminal transitions: {transitions['terminal'].sum():,}")
    print(
        "Transitions with possession changes:",
        f"{(
            transitions['next_posteam'].notna()
            & (transitions['posteam'] != transitions['next_posteam'])
        ).sum():,}",
    )

    print("\nFourth-down actions:")
    print(
        transitions.loc[
            transitions["is_fourth_down_decision"],
            "fourth_down_action",
        ].value_counts(dropna=False).to_string()
    )

    print("\nReward summary (home-team score differential):")
    print(transitions["reward_home"].describe().to_string())

    print("\nReward frequencies:")
    print(transitions["reward_home"].value_counts().sort_index().to_string())

    print("\nSample transitions:")
    sample_columns = [
        "game_id",
        "play_id",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "fourth_down_action",
        "next_posteam",
        "next_down",
        "score_diff_home",
        "next_score_diff_home",
        "reward_home",
        "terminal",
    ]

    print(
        transitions[sample_columns]
        .head(15)
        .to_string(index=False)
    )

    print(f"\nSaved transition dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
