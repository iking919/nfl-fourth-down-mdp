
"""
Check how fourth-down outcomes are recorded at the end of a drive.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/fourth_down_decisions_2024.csv")

COLUMNS = [
    "action",
    "posteam",
    "defteam",
    "yardline_100",
    "yards_gained",
    "fourth_down_converted",
    "field_goal_result",
    "drive_end_yard_line",
    "drive_end_transition",
    "posteam_score_post",
    "defteam_score_post",
    "desc",
]


def main():
    df = pd.read_csv(DATA_PATH, usecols=COLUMNS, low_memory=False)

    for action in ["go", "field_goal", "punt"]:
        subset = df[df["action"] == action]

        print("\n" + "=" * 65)
        print(f"{action.upper()} DRIVE OUTCOMES")
        print("=" * 65)

        print("\nDrive-ending transitions:")
        print(
            subset["drive_end_transition"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nDrive-ending field positions:")
        print(
            subset["drive_end_yard_line"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

        print("\nExample plays:")
        print(
            subset[
                [
                    "posteam",
                    "defteam",
                    "yardline_100",
                    "yards_gained",
                    "fourth_down_converted",
                    "field_goal_result",
                    "drive_end_yard_line",
                    "drive_end_transition",
                    "desc",
                ]
            ]
            .head(12)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
