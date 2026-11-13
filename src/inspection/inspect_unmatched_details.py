
"""
Inspect raw play-by-play records following unmatched fourth-down decisions.
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
    "desc",
    "kickoff_attempt",
    "kickoff_return_yards",
    "touchback",
    "field_goal_result",
]


def main():
    print("Loading play-by-play data...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda col: col in COLUMNS,
        low_memory=False,
    )

    # These are the 15 decisions that did not have a matched next play.
    unmatched_ids = [
        ("2024_15_TB_LAC", 4016),
        ("2024_10_DEN_KC", 4057),
        ("2024_02_CIN_KC", 4175),
        ("2024_09_CHI_ARI", 4216),
        ("2024_08_ARI_MIA", 4273),
        ("2024_10_PHI_DAL", 4286),
        ("2024_16_DEN_LAC", 4292),
        ("2024_10_ATL_NO", 4311),
        ("2024_02_LV_BAL", 4375),
        ("2024_04_NO_ATL", 4387),
        ("2024_08_BAL_CLE", 4474),
        ("2024_14_BUF_LA", 4488),
        ("2024_16_NE_BUF", 4521),
        ("2024_10_NYG_CAR", 4609),
        ("2024_18_JAX_IND", 4726),
    ]

    for game_id, decision_play_id in unmatched_ids:
        print("\n" + "=" * 75)
        print(f"GAME: {game_id} | DECISION PLAY: {decision_play_id}")
        print("=" * 75)

        plays = df[
            (df["game_id"] == game_id)
            & (df["play_id"] >= decision_play_id)
        ].head(12)

        print(
            plays.to_string(index=False)
        )


if __name__ == "__main__":
    main()
