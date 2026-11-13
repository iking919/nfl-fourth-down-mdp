
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/processed/fourth_down_decisions_2024.csv"


def main():
    print("Loading fourth-down decision data...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep only rushing and passing attempts.
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
    target = "fourth_down_converted"

    X = go[features]
    y = go[target].astype(int)
    groups = go["game_id"]

    # Keep all decisions from a game in the same split.
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
    print("Training conversion rate:", round(y_train.mean(), 4))
    print("Testing conversion rate:", round(y_test.mean(), 4))

    # Prepare numeric and categorical predictors separately.
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

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    # Use the training conversion rate as a simple probability baseline.
    baseline_probability = y_train.mean()
    baseline_predictions = [
        int(baseline_probability >= 0.5)
    ] * len(y_test)

    baseline_probabilities = [
        baseline_probability
    ] * len(y_test)

    model_predictions = model.predict(X_test)
    model_probabilities = model.predict_proba(X_test)[:, 1]

    print("\nCONSTANT BASELINE")
    print("=" * 60)
    print("Accuracy:", round(
        accuracy_score(y_test, baseline_predictions), 4
    ))
    print("Brier score:", round(
        brier_score_loss(y_test, baseline_probabilities), 4
    ))
    print("Log loss:", round(
        log_loss(y_test, baseline_probabilities), 4
    ))

    print("\nLOGISTIC REGRESSION")
    print("=" * 60)
    print("Accuracy:", round(
        accuracy_score(y_test, model_predictions), 4
    ))
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, model_predictions), 4
    ))
    print("ROC AUC:", round(
        roc_auc_score(y_test, model_probabilities), 4
    ))
    print("Precision:", round(
        precision_score(y_test, model_predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_test, model_predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_test, model_predictions, zero_division=0), 4
    ))
    print("Brier score:", round(
        brier_score_loss(y_test, model_probabilities), 4
    ))
    print("Log loss:", round(
        log_loss(y_test, model_probabilities), 4
    ))

    print("\nPREDICTED CONVERSION PROBABILITIES")
    print("=" * 60)
    print(pd.Series(model_probabilities).describe())

    print("\nACTUAL VS. PREDICTED CONVERSION RATE")
    print("=" * 60)
    print("Actual:", round(y_test.mean(), 4))
    print("Predicted:", round(model_probabilities.mean(), 4))


if __name__ == "__main__":
    main()
