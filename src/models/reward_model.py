
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = (
    "data/processed/fourth_down_mdp_transitions_2024.csv"
)


def main():
    print("Loading fourth-down transition data...")

    df = pd.read_csv(DATA_PATH)

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
    y = df["reward"]
    groups = df["game_id"]

    # Keep all decisions from each game in one partition.
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

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # Compare against a model that always predicts the training mean.
    baseline = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", DummyRegressor(strategy="mean")),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nTraining reward models...")

    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    predictions = model.predict(X_test)

    def report(name, actual, predicted):
        print(f"\n{name}")
        print("-" * 50)
        print("MAE:",
              round(mean_absolute_error(actual, predicted), 4))
        print("RMSE:",
              round(mean_squared_error(actual, predicted) ** 0.5, 4))
        print("R-squared:",
              round(r2_score(actual, predicted), 4))

    print("\nTEST SET")
    print("=" * 60)
    print("Training observations:", len(train_idx))
    print("Testing observations:", len(test_idx))
    print("Training games:", groups.iloc[train_idx].nunique())
    print("Testing games:", groups.iloc[test_idx].nunique())

    report("MEAN BASELINE", y_test, baseline_predictions)
    report("RANDOM FOREST", y_test, predictions)

    print("\nACTUAL VS. PREDICTED REWARD")
    print("=" * 60)

    comparison = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
    })

    print(comparison.describe().round(3))


if __name__ == "__main__":
    main()
