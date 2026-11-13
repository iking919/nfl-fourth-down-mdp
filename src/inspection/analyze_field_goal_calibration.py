
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


def main():
    decisions = pd.read_csv(
        "data/processed/fourth_down_decisions_2024.csv",
        low_memory=False,
    )

    data = decisions[
        decisions["play_type"] == "field_goal"
    ].copy()

    data["made"] = (
        data["field_goal_result"].str.lower() == "made"
    ).astype(int)

    features = [
        "kick_distance",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
    ]

    data = data.dropna(
        subset=features + ["made", "game_id"]
    )

    X = data[features]
    y = data["made"]
    groups = data["game_id"]

    numeric_features = [
        "kick_distance",
        "game_seconds_remaining",
        "score_differential",
    ]

    categorical_features = ["qtr"]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("FIELD-GOAL CALIBRATION")
    print("=" * 60)

    print("Test observations:", len(y_test))
    print(f"Actual make rate: {y_test.mean():.4f}")
    print(f"Mean predicted probability: {probabilities.mean():.4f}")
    print(f"Brier score: {brier_score_loss(y_test, probabilities):.4f}")
    print(f"Log loss: {log_loss(y_test, probabilities):.4f}")

    # Calibration by predicted-probability decile.
    calibration_data = pd.DataFrame({
        "predicted_probability": probabilities,
        "actual": y_test.to_numpy(),
    })

    calibration_data["probability_bin"] = pd.qcut(
        calibration_data["predicted_probability"],
        q=10,
        duplicates="drop",
    )

    print("\nCALIBRATION BY PREDICTED PROBABILITY")
    print("-" * 60)

    print(
        calibration_data.groupby(
            "probability_bin",
            observed=True,
        ).agg(
            observations=("actual", "size"),
            mean_predicted=("predicted_probability", "mean"),
            actual_make_rate=("actual", "mean"),
        ).to_string()
    )

    # Calibration by kick distance.
    distance_data = pd.DataFrame({
        "kick_distance": X_test["kick_distance"].to_numpy(),
        "predicted_probability": probabilities,
        "actual": y_test.to_numpy(),
    })

    distance_data["distance_bin"] = pd.cut(
        distance_data["kick_distance"],
        bins=[0, 39, 49, 59, 100],
        labels=["Under 40", "40-49", "50-59", "60+"],
        include_lowest=True,
    )

    print("\nCALIBRATION BY KICK DISTANCE")
    print("-" * 60)

    print(
        distance_data.groupby(
            "distance_bin",
            observed=True,
        ).agg(
            observations=("actual", "size"),
            mean_predicted=("predicted_probability", "mean"),
            actual_make_rate=("actual", "mean"),
        ).to_string()
    )


if __name__ == "__main__":
    main()
