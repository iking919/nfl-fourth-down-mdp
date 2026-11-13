
import pandas as pd
from sklearn.metrics import brier_score_loss


INPUT_PATH = (
    "data/processed/action_propensity_predictions_2024.csv"
)

ACTIONS = ["field_goal", "go", "punt"]


def main():
    df = pd.read_csv(INPUT_PATH)

    print("\nPROPENSITY CALIBRATION")
    print("=" * 70)

    # Compare predicted probabilities with actual action frequencies.
    # Each action is evaluated as a separate one-versus-rest outcome.
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0]

    for action in ACTIONS:
        probability_column = f"prob_{action}"
        actual = (df["fourth_down_action"] == action).astype(int)

        df[f"{action}_probability_bin"] = pd.cut(
            df[probability_column],
            bins=bins,
            include_lowest=True,
        )

        calibration = (
            df.groupby(
                f"{action}_probability_bin",
                observed=True,
            )
            .agg(
                observations=(probability_column, "size"),
                mean_predicted=(probability_column, "mean"),
                actual_frequency=(probability_column, lambda x: (
                    actual.loc[x.index].mean()
                )),
            )
            .round(3)
        )

        print(f"\n{action.upper()}")
        print(
            f"Brier score: "
            f"{brier_score_loss(actual, df[probability_column]):.4f}"
        )
        print(calibration.to_string())

    print("\nACTION PROPENSITY DISTRIBUTIONS")
    print("=" * 70)

    for action in ACTIONS:
        column = f"prob_{action}"

        print(f"\nEstimated probability of {action}:")
        print(
            df[column]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .round(3)
        )

    print("\nCOMMON SUPPORT")
    print("=" * 70)

    # Count situations where every action has at least the
    # specified estimated probability.
    for threshold in [0.05, 0.10, 0.20]:
        supported = (
            df[[f"prob_{action}" for action in ACTIONS]]
            >= threshold
        ).all(axis=1)

        print(
            f"All three actions have propensity >= {threshold:.2f}: "
            f"{supported.sum():,} "
            f"({supported.mean():.1%})"
        )

    print("\nOBSERVED ACTIONS WITH LOW PROPENSITY")
    print("=" * 70)

    for action in ACTIONS:
        selected = df["fourth_down_action"] == action
        probabilities = df.loc[selected, f"prob_{action}"]

        print(f"\n{action}: {selected.sum():,} decisions")

        for threshold in [0.05, 0.10, 0.20]:
            fraction = (probabilities < threshold).mean()

            print(
                f"  Below {threshold:.2f}: {fraction:.1%}"
            )


if __name__ == "__main__":
    main()
