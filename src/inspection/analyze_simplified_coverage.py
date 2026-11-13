
import pandas as pd


INPUT_PATH = "data/processed/fourth_down_mdp_transitions_2024.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Group yards to go.
    df["yards_bin"] = pd.cut(
        df["ydstogo"],
        bins=[0, 2, 5, 10, 100],
        labels=["1-2", "3-5", "6-10", "11+"],
    )

    # Group field position.
    df["field_bin"] = pd.cut(
        df["yardline_100"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["1-20", "21-40", "41-60", "61-80", "81-100"],
    )

    # Group score differential from the team's perspective.
    df["score_bin"] = pd.cut(
        df["score_diff_team"],
        bins=[-float("inf"), -13, -7, -1, 0, 7, float("inf")],
        labels=[
            "Trailing 14+",
            "Trailing 8-13",
            "Trailing 1-7",
            "Tied",
            "Leading 1-7",
            "Leading 8+",
        ],
    )

    state_columns = [
        "field_bin",
        "yards_bin",
        "score_bin",
    ]

    state_actions = (
        df.groupby(
            state_columns + ["fourth_down_action"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    state_counts = state_actions.sum(axis=1)
    action_coverage = (state_actions > 0).sum(axis=1)

    print("\nSIMPLIFIED STATE-ACTION COVERAGE")
    print("=" * 65)

    print(f"Distinct states: {len(state_actions):,}")
    print(
        f"One observed action: {(action_coverage == 1).sum():,}"
    )
    print(
        f"Two observed actions: {(action_coverage == 2).sum():,}"
    )
    print(
        f"Three observed actions: {(action_coverage == 3).sum():,}"
    )

    print("\nOBSERVATIONS PER STATE")
    print(state_counts.describe().round(2))

    print("\nACTION COUNTS BY FIELD POSITION AND YARDS TO GO")
    print(
        pd.crosstab(
            [df["field_bin"], df["yards_bin"]],
            df["fourth_down_action"],
        )
    )

    print("\nSTATES WITH ALL THREE ACTIONS")
    supported_states = state_actions[
        (state_actions > 0).all(axis=1)
    ]
    print(supported_states.to_string())


if __name__ == "__main__":
    main()
