
import pandas as pd


DECISIONS_PATH = (
    "data/processed/fourth_down_decisions_2024.csv"
)

TRANSITIONS_PATH = (
    "data/processed/fourth_down_mdp_transitions_2024.csv"
)


def main():
    print("Loading decision and transition datasets...")

    decisions = pd.read_csv(DECISIONS_PATH, low_memory=False)
    transitions = pd.read_csv(TRANSITIONS_PATH, low_memory=False)

    # Immediate outcomes recorded on the fourth-down play.
    outcome_columns = [
        "fourth_down_converted",
        "yards_gained",
        "touchdown",
        "interception",
        "fumble_lost",
        "field_goal_result",
        "kick_distance",
        "punt_blocked",
        "punt_inside_twenty",
        "punt_in_endzone",
        "return_yards",
        "return_touchdown",
    ]

    available = [
        column for column in outcome_columns
        if column in decisions.columns
    ]

    immediate = decisions[
        ["game_id", "play_id"] + available
    ]

    # Match each decision with its same-team transition.
    # The action is already stored in the transition dataset.
    df = transitions.merge(
        immediate,
        on=["game_id", "play_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    print("\nDATASET MATCHING")
    print("=" * 65)
    print(df["_merge"].value_counts())

    if not (df["_merge"] == "both").all():
        raise ValueError("Some transitions did not match decisions.")

    df = df.drop(columns="_merge")

    print("\nIMMEDIATE OUTCOME MISSINGNESS")
    print("=" * 65)
    print(df[available].isna().sum())

    print("\nGO-FOR-IT OUTCOMES")
    print("=" * 65)

    go = df[df["fourth_down_action"] == "go"]

    for column in [
        "fourth_down_converted",
        "touchdown",
        "interception",
        "fumble_lost",
    ]:
        if column in go.columns:
            print(f"\n{column}:")
            print(go[column].value_counts(dropna=False))

    print("\nFIELD GOAL OUTCOMES")
    print("=" * 65)

    fg = df[df["fourth_down_action"] == "field_goal"]

    for column in ["field_goal_result", "kick_distance"]:
        if column in fg.columns:
            print(f"\n{column}:")
            print(fg[column].value_counts(dropna=False).head(15))

    print("\nPUNT OUTCOMES")
    print("=" * 65)

    punt = df[df["fourth_down_action"] == "punt"]

    for column in [
        "punt_blocked",
        "punt_inside_twenty",
        "punt_in_endzone",
        "return_touchdown",
    ]:
        if column in punt.columns:
            print(f"\n{column}:")
            print(punt[column].value_counts(dropna=False))

    if "return_yards" in punt.columns:
        print("\nReturn yards:")
        print(punt["return_yards"].describe().round(2))

    print("\nSAME-TEAM TRANSITION OUTCOMES BY ACTION")
    print("=" * 65)

    transition_targets = [
        "reward",
        "elapsed_seconds",
        "next_yardline_100",
        "next_ydstogo",
        "next_score_diff_team",
    ]

    for column in transition_targets:
        print(f"\n{column}:")
        print(
            df.groupby("fourth_down_action")[column]
            .describe()
            .round(2)
        )

    print("\nTERMINAL TRANSITIONS BY ACTION")
    print("=" * 65)
    print(
        pd.crosstab(
            df["fourth_down_action"],
            df["terminal"],
            margins=True,
        )
    )


if __name__ == "__main__":
    main()
