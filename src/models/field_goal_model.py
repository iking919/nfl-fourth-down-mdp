
import pandas as pd

from sklearn.metrics import confusion_matrix
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
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/processed/fourth_down_decisions_2024.csv"


def main():
    print("Loading fourth-down decision data...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    fg = df[df["play_type"] == "field_goal"].copy()

    # Treat missed and blocked kicks as unsuccessful attempts.
    fg["made"] = (fg["field_goal_result"] == "made").astype(int)

    features = [
        "kick_distance",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
    ]

    X = fg[features]
    y = fg["made"]
    groups = fg["game_id"]

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
    print("Training success rate:", round(y_train.mean(), 4))
    print("Testing success rate:", round(y_test.mean(), 4))

    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    # Predict the training-set success rate for every test attempt.
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

    print("\nPREDICTED SUCCESS PROBABILITIES")
    print("=" * 60)
    print(pd.Series(model_probabilities).describe())

    print("\nACTUAL VS. PREDICTED SUCCESS RATE")
    print("=" * 60)
    print("Actual:", round(y_test.mean(), 4))
    print("Predicted:", round(model_probabilities.mean(), 4))
    print("\nCONFUSION MATRIX AT 0.50")
    print("=" * 60)
    print(confusion_matrix(y_test, model_predictions))

    print("\nTHRESHOLD COMPARISON")
    print("=" * 60)

    for threshold in [0.50, 0.70, 0.80, 0.85, 0.90]:
        predictions = (model_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        print(f"\nThreshold: {threshold:.2f}")
        print("Predicted makes:", int(predictions.sum()))
        print("True positives:", tp)
        print("False positives:", fp)
        print("True negatives:", tn)
        print("False negatives:", fn)
        print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
        print("Recall:", round(recall_score(y_test, predictions), 4))


if __name__ == "__main__":
    main()
