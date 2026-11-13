
"""
Inspect the next scrimmage play following each fourth-down decision.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/play_by_play_2024.csv")

COLUMNS = [
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
]


def main():
    print("Loading play-by-play data...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=COLUMNS,
        low_memory=False,
    )

    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # Find the fourth-down decisions included in our cleaned dataset.
    decisions = df[
        (df["down"] == 4)
        & df["play_type"].isin(
            ["run", "pass", "field_goal", "punt"]
        )
    ].copy()

    # A scrimmage play has a down and an offensive team.
    scrimmage_plays = df[
        df["down"].notna()
        & df["posteam"].notna()
    ].copy()

    print(f"Fourth-down decisions: {len(decisions):,}")
    print(f"Scrimmage plays: {len(scrimmage_plays):,}")

    # Match each decision with the next scrimmage play in its game.
    next_plays = pd.merge_asof(
        decisions.sort_values(["play_id", "game_id"]),
        scrimmage_plays.sort_values(["play_id", "game_id"]),
        on="play_id",
        by="game_id",
        direction="forward",
        allow_exact_matches=False,
        suffixes=("_decision", "_next"),
    )

    # Only calculate possession changes when a next play was found.
    matched = next_plays["posteam_next"].notna()

    next_plays["possession_changed"] = pd.Series(
        pd.NA,
        index=next_plays.index,
        dtype="boolean",
    )

    next_plays.loc[matched, "possession_changed"] = (
        next_plays.loc[matched, "posteam_decision"]
        != next_plays.loc[matched, "posteam_next"]
    )

    print("\n" + "=" * 65)
    print("NEXT-PLAY MATCHING")
    print("=" * 65)

    print(
        "Decisions with a matched next play:",
        matched.sum(),
    )

    print(
        "Decisions without a matched next play:",
        (~matched).sum(),
    )

    print("\nPossession changes by original play type:")
    print(
        next_plays.groupby("play_type_decision")[
            "possession_changed"
        ].mean().mul(100).round(1).to_string()
    )

    # Combine runs and passes into the MDP's single "go" action.
    next_plays["mdp_action"] = next_plays["play_type_decision"].replace({
        "run": "go",
        "pass": "go",
    })

    print("\nPossession changes by MDP action:")
    print(
        next_plays.groupby("mdp_action")["possession_changed"]
        .agg(["count", "sum", "mean"])
        .assign(
            possession_change_pct=lambda x: (x["mean"] * 100).round(1)
        )
        .to_string()
    )

    print("\n" + "=" * 65)
    print("SAMPLE MATCHED DECISIONS")
    print("=" * 65)

    sample_columns = [
        "play_type_decision",
        "mdp_action",
        "posteam_decision",
        "yardline_100_decision",
        "posteam_next",
        "down_next",
        "ydstogo_next",
        "yardline_100_next",
        "game_seconds_remaining_next",
        "score_differential_next",
        "possession_changed",
    ]

    print(
        next_plays[sample_columns]
        .head(25)
        .to_string(index=False)
    )

    # Inspect decisions that did not have a subsequent scrimmage play.
    unmatched = next_plays[~matched]

    print("\n" + "=" * 65)
    print("DECISIONS WITHOUT A MATCHED NEXT PLAY")
    print("=" * 65)

    if unmatched.empty:
        print("All decisions had a matched next play.")
    else:
        print(
            unmatched[
                [
                    "game_id",
                    "play_id",
                    "play_type_decision",
                    "posteam_decision",
                    "yardline_100_decision",
                    "game_seconds_remaining_decision",
                    "score_differential_decision",
                ]
            ].to_string(index=False)
        )

    # Check what happened after decisions without a matched scrimmage play.
    print("\n" + "=" * 65)
    print("REMAINING PLAYS AFTER UNMATCHED DECISIONS")
    print("=" * 65)

    for _, decision in unmatched.iterrows():
        remaining_plays = df[
            (df["game_id"] == decision["game_id"])
            & (df["play_id"] > decision["play_id"])
        ].head(8)

        print(
            f"\nGame: {decision['game_id']} | "
            f"Play ID: {decision['play_id']} | "
            f"Action: {decision['play_type_decision']}"
        )

        if remaining_plays.empty:
            print("No remaining play records.")
        else:
            print(
                remaining_plays[
                    [
                        "play_id",
                        "play_type",
                        "posteam",
                        "down",
                        "game_seconds_remaining",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
