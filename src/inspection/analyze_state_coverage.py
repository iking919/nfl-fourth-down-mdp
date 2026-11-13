
import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/fourth_down_mdp_transitions_2024.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Identify whether the team making the decision is at home.
    df["is_home_team"] = (
        df["posteam"] == df["home_team"]
    )

    # Group remaining time into broad intervals.
    df["time_bin"] = pd.cut(
        df["game_seconds_remaining"],
        bins=[-1, 300, 900, 1800, 3600],
        labels=["0-5 min", "5-15 min", "15-30 min", "30-60 min"],
    )

    # Group yards to go.
    df["yards_bin"] = pd.cut(
        df["ydstogo"],
        bins=[0, 2, 5, 10, 100],
        labels=["1-2", "3-5", "6-10", "11+"],
    )

    # Group field position. Lower yardline_100 means closer
    # to the opponent's end zone.
    df["field_bin"] = pd.cut(
        df["yardline_100"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["1-20", "21-40", "41-60", "61-80", "81-100"],
    )

    # Group score differential from the current team's perspective.
    df["score_bin"] = np.select(
        [
            df["score_diff_team"] <= -14,
            df["score_diff_team"].between(-13, -8),
            df["score_diff_team"].between(-7, -1),
            df["score_diff_team"] == 0,
            df["score_diff_team"].between(1, 7),
            df["score_diff_team"] >= 8,
        ],
        [
            "Trailing 14+",
            "Trailing 8-13",
            "Trailing 1-7",
            "Tied",
            "Leading 1-7",
            "Leading 8+",
        ],
        default="Unknown",
    )

    print("\nSTATE VARIABLE CHECK")
    print("=" * 65)

    state_variables = [
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_diff_team",
        "is_home_team",
    ]

    print("Missing values:")
    print(df[state_variables].isna().sum())

    print("\nSTATE VARIABLE DISTRIBUTIONS")

    for column in [
        "qtr",
        "time_bin",
        "yards_bin",
        "field_bin",
        "score_bin",
        "is_home_team",
    ]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).sort_index())

    # A preliminary state representation for measuring coverage.
    state_columns = [
        "qtr",
        "time_bin",
        "is_home_team",
        "yards_bin",
        "field_bin",
        "score_bin",
    ]

    # Count observations for each state-action combination.
    state_action_counts = (
        df.groupby(
            state_columns + ["fourth_down_action"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    state_counts = state_action_counts.sum(axis=1)
    actions_available = (state_action_counts > 0).sum(axis=1)

    print("\nSTATE-ACTION COVERAGE")
    print("=" * 65)

    print(f"Distinct binned states: {len(state_counts):,}")
    print(
        "States with only one observed action:",
        f"{(actions_available == 1).sum():,}",
    )
    print(
        "States with two observed actions:",
        f"{(actions_available == 2).sum():,}",
    )
    print(
        "States with all three observed actions:",
        f"{(actions_available == 3).sum():,}",
    )

    print("\nOBSERVATIONS PER STATE")
    print(state_counts.describe().round(2))

    print("\nMOST FREQUENT STATES")
    frequent_states = state_action_counts.copy()
    frequent_states["total"] = state_counts
    print(
        frequent_states.sort_values("total", ascending=False)
        .head(15)
        .to_string()
    )

    print("\nACTION COUNTS BY SCORE SITUATION")
    print(
        pd.crosstab(
            df["score_bin"],
            df["fourth_down_action"],
            margins=True,
        )
    )

    print("\nACTION COUNTS BY FIELD POSITION")
    print(
        pd.crosstab(
            df["field_bin"],
            df["fourth_down_action"],
            margins=True,
        )
    )


if __name__ == "__main__":
    main()
