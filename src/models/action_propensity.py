
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
)


INPUT_PATH = "data/processed/fourth_down_mdp_transitions_2024.csv"


def main():
    print("Loading fourth-down decision data...")
    df = pd.read_csv(INPUT_PATH)

    # Use game situation variables to predict the coach's action.
    numeric_features = [
        "game_seconds_remaining",
        "ydstogo",
        "yardline_100",
        "score_diff_team",
    ]

    categorical_features = ["qtr"]

    X = df[numeric_features + categorical_features]
    y = df["fourth_down_action"]

    # Standardize numerical variables and encode the quarter.
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    # Evaluate using five-fold stratified cross-validation.
    # Each decision receives a prediction from a model that
    # was not trained on that particular observation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )

    # Obtain the action labels in the same order as the probability columns.
    classes = sorted(y.unique())

    # Fit once to establish the classifier's class ordering.
    model.fit(X, y)
    classes = model.named_steps["classifier"].classes_

    predictions = classes[probabilities.argmax(axis=1)]

    print("\nACTION COUNTS")
    print("=" * 60)
    print(y.value_counts())

    print("\nCROSS-VALIDATION PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:          {accuracy_score(y, predictions):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )
    print(f"Log loss:          {log_loss(y, probabilities, labels=classes):.4f}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            y,
            predictions,
            labels=classes,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")
    print("=" * 60)
    print(
        pd.DataFrame(
            confusion_matrix(y, predictions, labels=classes),
            index=[f"Actual {c}" for c in classes],
            columns=[f"Predicted {c}" for c in classes],
        )
    )

    # Find the estimated probability assigned to the action
    # that the coach actually selected.
    class_indices = {
        action: index for index, action in enumerate(classes)
    }

    observed_action_probabilities = [
        probabilities[i, class_indices[action]]
        for i, action in enumerate(y)
    ]

    df["observed_action_probability"] = (
        observed_action_probabilities
    )

    # Save out-of-fold action probabilities for later diagnostics.
    for index, action in enumerate(sorted(y.unique())):
        df[f"prob_{action}"] = probabilities[:, index]

    df.to_csv(
        "data/processed/action_propensity_predictions_2024.csv",
        index=False,
    )

    print(
        "\nSaved out-of-fold probabilities to "
        "data/processed/action_propensity_predictions_2024.csv"
    )

    print("\nPROPENSITY OF THE OBSERVED ACTION")
    print("=" * 60)
    print(
        df.groupby("fourth_down_action")[
            "observed_action_probability"
        ]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(3)
    )

    print("\nLOW-PROPENSITY OBSERVATIONS")
    print("=" * 60)

    for threshold in [0.05, 0.10, 0.20]:
        fraction = (
            df["observed_action_probability"] < threshold
        ).mean()

        print(
            f"Observed action probability below {threshold:.2f}: "
            f"{fraction:.1%}"
        )


if __name__ == "__main__":
    main()
