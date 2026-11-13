
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DECISIONS_PATH = "data/processed/fourth_down_decisions_2024.csv"
TRANSITIONS_PATH = "data/processed/game_transitions_2024.csv"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 55)
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.4f}")
    print(f"R^2:  {r2_score(y_test, predictions):.4f}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Mean actual:     {y_test.mean():.4f}")


def main():
    print("Loading punt transition data...")

    decisions = pd.read_csv(DECISIONS_PATH, low_memory=False)
    transitions = pd.read_csv(TRANSITIONS_PATH, low_memory=False)

    punt = decisions[
        decisions["play_type"] == "punt"
    ].copy()

    transition_cols = [
        "game_id",
        "play_id",
        "next_play_id",
        "next_posteam",
        "next_yardline_100",
    ]

    data = punt.merge(
        transitions[transition_cols],
        on=["game_id", "play_id"],
        how="left",
        validate="one_to_one",
    )

    # Keep punts where the receiving team obtained the next possession.
    data = data[
        data["next_play_id"].notna()
        & (data["next_posteam"] == data["defteam"])
    ].copy()

    print("\nDATASET")
    print("=" * 55)
    print("Ordinary receiving-team transitions:", len(data))
    print("Penalized punts:", data["is_penalized"].sum())
    print("Non-penalized punts:", (~data["is_penalized"]).sum())

    features = [
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_differential",
    ]

    target = "next_yardline_100"

    X = data[features].copy()
    y = data[target].copy()
    groups = data["game_id"]

    numeric_features = [
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_differential",
    ]

    categorical_features = ["qtr"]

    preprocessor = ColumnTransformer(
        transformers=[
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
        ]
    )

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

    print("\nDATA SPLIT")
    print("=" * 55)
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Training games:", groups.iloc[train_idx].nunique())
    print("Testing games:", groups.iloc[test_idx].nunique())

    print("\nTARGET DISTRIBUTION")
    print("=" * 55)
    print(y.describe())

    baseline_prediction = [y_train.mean()] * len(y_test)

    print("\nMEAN BASELINE")
    print("-" * 55)
    print(f"MAE:  {mean_absolute_error(y_test, baseline_prediction):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, baseline_prediction) ** 0.5:.4f}")
    print(f"R^2:  {r2_score(y_test, baseline_prediction):.4f}")

    linear_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    evaluate_model(
        "LINEAR REGRESSION",
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    forest_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    evaluate_model(
        "RANDOM FOREST REGRESSION",
        forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
