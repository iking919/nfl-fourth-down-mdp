
import pandas as pd

df = pd.read_csv("data/processed/game_transitions_2024.csv")

# Keep only fourth-down decisions.
decisions = df[df["is_fourth_down_decision"]].copy()
decisions = decisions.sort_values(["game_id", "play_id"])

# Find each team's next fourth-down decision within the same game.
grouped = decisions.groupby(["game_id", "posteam"])

decisions["next_same_team_play_id"] = grouped["play_id"].shift(-1)
decisions["next_same_team_score_home"] = (
    grouped["score_diff_home"].shift(-1)
)

decisions["has_next_same_team"] = (
    decisions["next_same_team_play_id"].notna()
)

# Get each game's final home-team score differential.
final_margins = (
    df.groupby("game_id")["next_score_diff_home"].last()
)

decisions["final_score_home"] = (
    decisions["game_id"].map(final_margins)
)

# If there is another decision, use its score.
# Otherwise, use the final game score as the terminal outcome.
decisions["next_score_home"] = (
    decisions["next_same_team_score_home"].where(
        decisions["has_next_same_team"],
        decisions["final_score_home"]
    )
)

# Convert home-team score differential into the perspective
# of the team making the current fourth-down decision.
decisions["team_sign"] = decisions.apply(
    lambda row: 1 if row["posteam"] == row["home_team"] else -1,
    axis=1
)

decisions["current_score_team"] = (
    decisions["score_diff_home"] * decisions["team_sign"]
)

decisions["next_score_team"] = (
    decisions["next_score_home"] * decisions["team_sign"]
)

# Cumulative reward between this decision and the next
# decision by the same team (or the game's final outcome).
decisions["reward_team"] = (
    decisions["next_score_team"] - decisions["current_score_team"]
)

print("\nSAME-TEAM REWARD VALIDATION")
print("=" * 65)

print(f"Total decisions: {len(decisions):,}")
print(
    f"Nonterminal transitions: "
    f"{decisions['has_next_same_team'].sum():,}"
)
print(
    f"Terminal transitions: "
    f"{(~decisions['has_next_same_team']).sum():,}"
)

print("\nREWARD SUMMARY BY ACTION")
print(
    decisions.groupby("fourth_down_action")["reward_team"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3)
)

print("\nTERMINAL REWARD SUMMARY")
print(
    decisions.loc[
        ~decisions["has_next_same_team"], "reward_team"
    ].describe().round(3)
)

print("\nSAMPLE TRANSITIONS")
print(
    decisions[
        [
            "game_id",
            "posteam",
            "fourth_down_action",
            "score_diff_home",
            "next_score_home",
            "current_score_team",
            "next_score_team",
            "reward_team",
            "has_next_same_team",
        ]
    ].head(15).to_string(index=False)
)

# Check that all calculated rewards are valid.
assert decisions["reward_team"].notna().all()
assert decisions["next_score_team"].notna().all()

print("\nAll reward values successfully calculated.")


# Verify that rewards telescope for every team in every game.
decisions["final_score_team"] = (
    decisions["final_score_home"] * decisions["team_sign"]
)

team_checks = decisions.groupby(
    ["game_id", "posteam"]
).agg(
    reward_sum=("reward_team", "sum"),
    initial_score=("current_score_team", "first"),
    final_score=("final_score_team", "first"),
)

team_checks["expected_reward_sum"] = (
    team_checks["final_score"] - team_checks["initial_score"]
)

team_checks["difference"] = (
    team_checks["reward_sum"] - team_checks["expected_reward_sum"]
)

print("\nTEAM-LEVEL REWARD CONSISTENCY")
print("=" * 65)
print(f"Team-game sequences checked: {len(team_checks):,}")
print(
    "Maximum absolute difference:",
    team_checks["difference"].abs().max()
)
print(
    "Sequences with nonzero difference:",
    (team_checks["difference"].abs() > 1e-9).sum()
)

assert (team_checks["difference"].abs() < 1e-9).all()

print("All team-level reward checks passed.")

