
import pandas as pd


INPUT_PATH = "data/processed/fourth_down_mdp_transitions_2024.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    print("\nFIELD POSITION BY ACTION")
    print("=" * 65)

    print(
        df.groupby("fourth_down_action")["yardline_100"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
    )

    print("\nYARDS TO GO BY ACTION")
    print("=" * 65)

    print(
        df.groupby("fourth_down_action")["ydstogo"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
    )

    print("\nSCORE DIFFERENTIAL BY ACTION")
    print("=" * 65)

    print(
        df.groupby("fourth_down_action")["score_diff_team"]
        .describe()
        .round(2)
    )

    print("\nFIELD POSITION RANGE BY ACTION")
    print("=" * 65)

    for action, group in df.groupby("fourth_down_action"):
        print(
            f"{action:12s} "
            f"yardline_100: "
            f"{group['yardline_100'].min():.0f}-"
            f"{group['yardline_100'].max():.0f}"
        )


if __name__ == "__main__":
    main()
