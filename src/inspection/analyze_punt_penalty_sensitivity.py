import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_model():
    numeric_features = [
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def report(name, model, X, y):
    predictions = model.predict(X)

    print(f"\n{name}")
    print("-" * 55)
    print(f"Observations: {len(y)}")
    print(f"MAE:  {mean_absolute_error(y, predictions):.4f}")
    print(f"RMSE: {mean_squared_error(y, predictions) ** 0.5:.4f}")
    print(f"R^2:  {r2_score(y, predictions):.4f}")


def main():
    decisions = pd.read_csv(
        "data/processed/fourth_down_decisions_2024.csv",
        low_memory=False,
    )

    transitions = pd.read_csv(
        "data/processed/game_transitions_2024.csv",
        low_memory=False,
    )

    punt = decisions[decisions["play_type"] == "punt"].copy()

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

    data = data[
        data["next_play_id"].notna()
        & (data["next_posteam"] == data["defteam"])
    ].copy()

    features = [
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_differential",
    ]

    target = "next_yardline_100"

    X = data[features]
    y = data[target]
    groups = data["game_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    clean_train = train[~train["is_penalized"]]
    clean_test = test[~test["is_penalized"]]

    print("PUNT PENALTY SENSITIVITY")
    print("=" * 60)
    print("All training punts:", len(train))
    print("Non-penalized training punts:", len(clean_train))
    print("Non-penalized test punts:", len(clean_test))
    print("Penalized test punts:", test["is_penalized"].sum())

    # Model 1: train using all available training punts.
    all_model = make_model()
    all_model.fit(train[features], train[target])

    # Model 2: train using only non-penalized training punts.
    clean_model = make_model()
    clean_model.fit(clean_train[features], clean_train[target])

    # Evaluate both on the same non-penalized test observations.
    report(
        "ALL-PUNTS TRAINING / CLEAN TEST",
        all_model,
        clean_test[features],
        clean_test[target],
    )

    report(
        "NON-PENALIZED TRAINING / CLEAN TEST",
        clean_model,
        clean_test[features],
        clean_test[target],
    )


if __name__ == "__main__":
    main()