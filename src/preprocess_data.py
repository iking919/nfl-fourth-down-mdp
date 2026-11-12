
"""
Create a cleaned dataset of fourth-down decisions for 2024.
"""

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/raw/play_by_play_2024.csv")
OUTPUT_PATH = Path("data/processed/fourth_down_decisions_2024.csv")

STATE_COLUMNS = [
    "posteam",
    "defteam",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
]

ACTION_MAP = {
    "run": "go",
    "pass": "go",
    "field_goal": "field_goal",
    "punt": "punt",
}


def main():
    print("Loading play-by-play data...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Start with fourth-down plays.
    fourth_down = df[df["down"] == 4].copy()

    print(f"Original fourth-down rows: {len(fourth_down):,}")

    # Keep only the actions we want to compare.
    decisions = fourth_down[
        fourth_down["play_type"].isin(ACTION_MAP)
    ].copy()

    decisions["action"] = decisions["play_type"].map(ACTION_MAP)

    print(f"Rows after action filtering: {len(decisions):,}")

    # Keep the penalty fields so we can account for them later.
    # Missing penalty values are left as missing.
    decisions["is_penalized"] = decisions["penalty"].eq(1)

    # Check which rows have all the information needed to describe
    # the decision state.
    print("\nMissing state information:")
    print(decisions[STATE_COLUMNS].isna().sum().to_string())

    complete_state = decisions.dropna(subset=STATE_COLUMNS).copy()

    print(f"\nRows with complete state information: {len(complete_state):,}")
    print(f"Rows removed for missing state information: "
          f"{len(decisions) - len(complete_state):,}")

    # Save the cleaned decision rows.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    complete_state.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("ACTION COUNTS")
    print("=" * 60)
    print(complete_state["action"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("PENALTY COUNTS")
    print("=" * 60)
    print(complete_state["is_penalized"].value_counts().to_string())

    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print(f"Final rows: {len(complete_state):,}")


if __name__ == "__main__":
    main()
