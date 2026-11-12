
"""
Inspect fourth-down plays, including penalties and no-play situations.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/play_by_play_2024.csv")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep fourth-down rows for this inspection.
    fourth_down = df[df["down"] == 4].copy()

    print(f"Fourth-down rows: {len(fourth_down):,}")

    # Check which values appear in the penalty-related columns.
    penalty_columns = [
        "play_deleted",
        "penalty",
        "penalty_type",
    ]

    for column in penalty_columns:
        print("\n" + "=" * 60)
        print(f"{column.upper()} COUNTS")
        print("=" * 60)

        if column in fourth_down.columns:
            print(
                fourth_down[column]
                .value_counts(dropna=False)
                .to_string()
            )

    # Look at penalty information specifically for no-play rows.
    no_play = fourth_down[fourth_down["play_type"] == "no_play"]

    print("\n" + "=" * 60)
    print("NO-PLAY PENALTY SUMMARY")
    print("=" * 60)

    print(f"Total no-play rows: {len(no_play):,}")

    for column in penalty_columns:
        if column in no_play.columns:
            print(f"\n{column}:")
            print(
                no_play[column]
                .value_counts(dropna=False)
                .to_string()
            )

    # Show a sample of descriptions to help identify unusual cases.
    sample_columns = [
        "game_id",
        "week",
        "play_id",
        "posteam",
        "ydstogo",
        "yardline_100",
        "play_type",
        "desc",
        "penalty",
        "penalty_type",
        "penalty_yards",
        "play_deleted",
    ]

    available = [c for c in sample_columns if c in no_play.columns]

    print("\n" + "=" * 60)
    print("SAMPLE NO-PLAY ROWS")
    print("=" * 60)

    print(no_play[available].head(30).to_string(index=False))

    # A few no-play rows have missing penalty fields, so check
    # their descriptions before deciding how to handle them.
    ambiguous = no_play[no_play["penalty"].isna()]

    print("\n" + "=" * 60)
    print("NO-PLAY ROWS WITH MISSING PENALTY INFORMATION")
    print("=" * 60)

    print(
        ambiguous[
            ["game_id", "play_id", "posteam", "ydstogo",
             "yardline_100", "desc"]
        ].to_string(index=False)
    )


    # Check whether penalties also appear on rows classified as
    # actual plays. We should not automatically remove these.
    print("\n" + "=" * 60)
    print("PENALTIES BY PLAY TYPE")
    print("=" * 60)

    print(
        pd.crosstab(
            fourth_down["play_type"],
            fourth_down["penalty"],
            margins=True,
            dropna=False
        ).to_string()
    )

    # Show actual play rows that have a recorded penalty.
    penalized_plays = fourth_down[
        (fourth_down["penalty"] == 1)
        & (fourth_down["play_type"] != "no_play")
    ]

    print("\n" + "=" * 60)
    print("PENALIZED FOURTH-DOWN PLAYS")
    print("=" * 60)

    print(f"Rows: {len(penalized_plays):,}")

    print(
        penalized_plays[
            ["game_id", "play_id", "play_type", "ydstogo",
             "yardline_100", "penalty_type", "penalty_yards", "desc"]
        ].head(30).to_string(index=False)
    )



if __name__ == "__main__":
    main()
