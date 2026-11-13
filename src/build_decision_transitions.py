
import os
import pandas as pd


INPUT_PATH = "data/processed/game_transitions_2024.csv"
OUTPUT_PATH = "data/processed/fourth_down_mdp_transitions_2024.csv"


def elapsed_game_time(qtr, seconds_remaining):
    """Convert the period clock into elapsed game time."""
    if qtr <= 4:
        return 3600 - seconds_remaining

    # 2024 regular-season overtime is a 10-minute period.
    return 3600 + (600 - seconds_remaining)


def main():
    print("Loading play-by-play transitions...")
    df = pd.read_csv(INPUT_PATH)

    # Keep only actual fourth-down decisions.
    decisions = df[df["is_fourth_down_decision"]].copy()
    decisions = decisions.sort_values(["game_id", "play_id"])

    # Find the next fourth-down decision by the same team.
    grouped = decisions.groupby(["game_id", "posteam"])

    next_columns = {
        "play_id": "next_play_id",
        "qtr": "next_qtr",
        "game_seconds_remaining": "next_time",
        "ydstogo": "next_ydstogo",
        "yardline_100": "next_yardline_100",
        "score_diff_home": "next_score_diff_home",
    }

    for source, destination in next_columns.items():
        decisions[destination] = grouped[source].shift(-1)

    decisions["terminal"] = decisions["next_play_id"].isna()

    # Convert home-team score differential into the perspective
    # of the team making the current fourth-down decision.
    decisions["team_sign"] = decisions.apply(
        lambda row: (
            1 if row["posteam"] == row["home_team"] else -1
        ),
        axis=1,
    )

    decisions["score_diff_team"] = (
        decisions["score_diff_home"] * decisions["team_sign"]
    )

    decisions["next_score_diff_team"] = (
        decisions["next_score_diff_home"] * decisions["team_sign"]
    )

    # Get the final home-team score differential for each game.
    final_scores = df.groupby("game_id")["next_score_diff_home"].last()

    decisions["final_score_diff_home"] = (
        decisions["game_id"].map(final_scores)
    )

    decisions["final_score_diff_team"] = (
        decisions["final_score_diff_home"] * decisions["team_sign"]
    )

    # Use the next same-team decision's score when available.
    # Otherwise, use the final game score.
    decisions["outcome_score_diff_team"] = (
        decisions["next_score_diff_team"].where(
            ~decisions["terminal"],
            decisions["final_score_diff_team"],
        )
    )

    # Reward is the change in the team's point differential.
    decisions["reward"] = (
        decisions["outcome_score_diff_team"]
        - decisions["score_diff_team"]
    )

    # Calculate elapsed time to the next same-team decision.
    decisions["elapsed_game_seconds"] = decisions.apply(
        lambda row: elapsed_game_time(
            row["qtr"], row["game_seconds_remaining"]
        ),
        axis=1,
    )

    decisions["next_elapsed_game_seconds"] = decisions.apply(
        lambda row: (
            elapsed_game_time(row["next_qtr"], row["next_time"])
            if not row["terminal"]
            else None
        ),
        axis=1,
    )

    decisions["elapsed_seconds"] = (
        decisions["next_elapsed_game_seconds"]
        - decisions["elapsed_game_seconds"]
    )

    # Keep the current state, action, reward, and next state.
    output_columns = [
        "game_id",
        "posteam",
        "home_team",
        "play_id",
        "fourth_down_action",
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_diff_team",
        "next_play_id",
        "next_qtr",
        "next_time",
        "next_ydstogo",
        "next_yardline_100",
        "next_score_diff_team",
        "terminal",
        "elapsed_seconds",
        "reward",
    ]

    mdp_data = decisions[output_columns].copy()

    # Basic validation.
    assert len(mdp_data) == 3997
    assert mdp_data["reward"].notna().all()
    assert mdp_data["terminal"].sum() == 570
    assert (~mdp_data["terminal"]).sum() == 3427

    assert (
        mdp_data.loc[~mdp_data["terminal"], "next_play_id"].notna().all()
    )

    assert (
        mdp_data.loc[mdp_data["terminal"], "next_play_id"].isna().all()
    )

    # Every nonterminal next state must belong to the same team and game.
    assert (
        mdp_data.loc[~mdp_data["terminal"], "posteam"].notna().all()
    )

    assert (mdp_data["elapsed_seconds"].dropna() > 0).all()

    # Save without overwriting an existing dataset.
    if os.path.exists(OUTPUT_PATH):
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Move or rename it before rebuilding."
        )

    mdp_data.to_csv(OUTPUT_PATH, index=False)

    print("\nDECISION-TRANSITION DATASET")
    print("=" * 60)
    print(f"Total decisions: {len(mdp_data):,}")
    print(f"Nonterminal transitions: {(~mdp_data['terminal']).sum():,}")
    print(f"Terminal transitions: {mdp_data['terminal'].sum():,}")

    print("\nACTION COUNTS")
    print(mdp_data["fourth_down_action"].value_counts())

    print("\nREWARD SUMMARY")
    print(mdp_data["reward"].describe().round(3))

    print("\nELAPSED TIME SUMMARY (NONTERMINAL)")
    print(
        mdp_data.loc[~mdp_data["terminal"], "elapsed_seconds"]
        .describe()
        .round(2)
    )

    print("\nSaved dataset to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
