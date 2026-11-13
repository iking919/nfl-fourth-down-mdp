
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = (
    "data/processed/fourth_down_mdp_transitions_2024.csv"
)


def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 50)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R-squared:", round(r2, 4))


def main():
    print("Loading fourth-down transition data...")

    df = pd.read_csv(DATA_PATH)

    # Next-state variables only exist for nonterminal transitions.
    df = df.loc[~df["terminal"]].copy()

    df["is_home_team"] = (
        df["posteam"] == df["home_team"]
    ).astype(int)

    numeric_features = [
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_diff_team",
        "is_home_team",
    ]

    categorical_features = [
        "fourth_down_action",
    ]

    features = numeric_features + categorical_features

    X = df[features]
    groups = df["game_id"]

    targets = [
        "next_yardline_100",
        "next_ydstogo",
    ]

    # Keep decisions from the same game in the same partition.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    print("\nDATA SPLIT")
    print("=" * 60)
    print("Training observations:", len(train_idx))
    print("Testing observations:", len(test_idx))
    print("Training games:", groups.iloc[train_idx].nunique())
    print("Testing games:", groups.iloc[test_idx].nunique())

    # Impute missing numeric values and encode the action.
    preprocessing = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ),
    ])

    for target in targets:
        print(f"\n\nTARGET: {target}")
        print("=" * 60)

        y_train = train[target]
        y_test = test[target]

        # Use the training mean for each action as the baseline.
        action_means = train.groupby(
            "fourth_down_action"
        )[target].mean()

        baseline_predictions = (
            test["fourth_down_action"].map(action_means)
        )

        model = Pipeline([
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=10,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

        print("Training random forest...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        evaluate(
            "ACTION-MEAN BASELINE",
            y_test,
            baseline_predictions,
        )

        evaluate(
            "RANDOM FOREST",
            y_test,
            predictions,
        )

        comparison = pd.DataFrame({
            "actual": y_test.values,
            "predicted": predictions,
        })

        print("\nACTUAL VS. PREDICTED SUMMARY")
        print(comparison.describe().round(2))


if __name__ == "__main__":
    main()
