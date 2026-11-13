
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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

    # Identify whether the team making the decision is at home.
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
    y = df["terminal"].astype(int)
    groups = df["game_id"]

    # Keep every decision from a game in the same split.
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

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("\nTraining terminal classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTEST SET PERFORMANCE")
    print("=" * 60)
    print("Training observations:", len(train_idx))
    print("Testing observations:", len(test_idx))
    print("Training games:", groups.iloc[train_idx].nunique())
    print("Testing games:", groups.iloc[test_idx].nunique())

    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 4))

    print("Balanced accuracy:",
          round(balanced_accuracy_score(y_test, predictions), 4))

    print("ROC AUC:",
          round(roc_auc_score(y_test, probabilities), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Nonterminal", "Terminal"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
