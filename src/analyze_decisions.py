
"""
Look at fourth-down decisions and their outcomes in the 2024 season.
"""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/fourth_down_decisions_2024.csv")

# Only load the columns used in this analysis.
COLUMNS = [
    "action",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "fourth_down_converted",
    "field_goal_result",
]


def print_action_breakdown(df, column, title):
    # Show the percentage of each action within each group.
    counts = pd.crosstab(df[column], df["action"])
    percentages = pd.crosstab(
        df[column], df["action"], normalize="index"
    ) * 100

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("\nNumber of decisions:")
    print(counts.to_string())
    print("\nAction percentages:")
    print(percentages.round(1).to_string())


def main():
    df = pd.read_csv(DATA_PATH, usecols=COLUMNS)

    print(f"Loaded {len(df):,} fourth-down decisions.")

    # Group yards to go so the results are easier to compare.
    df["yards_to_go_group"] = pd.cut(
        df["ydstogo"],
        bins=[0, 2, 4, 6, float("inf")],
        labels=["1-2", "3-4", "5-6", "7+"],
    )

    # yardline_100 is the distance to the opponent's end zone.
    df["field_position"] = pd.cut(
        df["yardline_100"],
        bins=[0, 20, 50, 100],
        labels=["Red zone", "Opponent territory", "Own territory"],
        include_lowest=True,
    )

    # Group decisions by whether the offense was leading or trailing.
    df["score_situation"] = "Tied"
    df.loc[df["score_differential"] > 0, "score_situation"] = "Leading"
    df.loc[df["score_differential"] < 0, "score_situation"] = "Trailing"

    print_action_breakdown(
        df, "yards_to_go_group", "ACTION BY YARDS TO GO"
    )

    print_action_breakdown(
        df, "field_position", "ACTION BY FIELD POSITION"
    )

    print_action_breakdown(
        df, "score_situation", "ACTION BY SCORE SITUATION"
    )

    # Conversion rates only make sense for run/pass attempts here.
    go_plays = df[df["action"] == "go"].copy()
    go_plays["converted"] = go_plays["fourth_down_converted"] == 1

    print("\n" + "=" * 60)
    print("GO-FOR-IT CONVERSION RATES")
    print("=" * 60)

    print(
        go_plays.groupby("yards_to_go_group", observed=False)
        ["converted"]
        .agg(["count", "mean"])
        .assign(conversion_pct=lambda x: x["mean"] * 100)
        .drop(columns="mean")
        .round(1)
        .to_string()
    )

    # For field goals, a made kick is the immediate successful outcome.
    field_goals = df[df["action"] == "field_goal"].copy()
    field_goals["made"] = field_goals["field_goal_result"] == "made"

    print("\n" + "=" * 60)
    print("FIELD GOAL SUCCESS RATES")
    print("=" * 60)

    print(
        field_goals.groupby("field_position", observed=False)
        ["made"]
        .agg(["count", "mean"])
        .assign(make_pct=lambda x: x["mean"] * 100)
        .drop(columns="mean")
        .round(1)
        .to_string()
    )


if __name__ == "__main__":
    main()
