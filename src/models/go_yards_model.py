
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/processed/fourth_down_decisions_2024.csv"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("=" * 60)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R-squared:", round(r2, 4))
    print("Prediction mean:", round(predictions.mean(), 4))
    print("Prediction std:", round(predictions.std(), 4))


def main():
    print("Loading fourth-down decision data...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    go = df[df["play_type"].isin(["run", "pass"])].copy()

    numeric_features = [
        "qtr",
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_differential",
    ]

    categorical_features = ["play_type"]
    features = numeric_features + categorical_features

    X = go[features]
    y = go["yards_gained"]
    groups = go["game_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nDATA SPLIT")
    print("=" * 60)
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Training games:", groups.iloc[train_idx].nunique())
    print("Testing games:", groups.iloc[test_idx].nunique())

    print("\nACTUAL TEST TARGET")
    print("=" * 60)
    print(y_test.describe())

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    mean_prediction = [y_train.mean()] * len(y_test)

    print("\nMEAN BASELINE")
    print("=" * 60)
    print("MAE:", round(mean_absolute_error(y_test, mean_prediction), 4))
    print("RMSE:", round(
        mean_squared_error(y_test, mean_prediction) ** 0.5, 4
    ))
    print("R-squared:", round(r2_score(y_test, mean_prediction), 4))

    linear_model = Pipeline([
        ("preprocessing", preprocessing),
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
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    evaluate_model(
        "RANDOM FOREST",
        forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
