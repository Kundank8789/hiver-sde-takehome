from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "weak_training.jsonl"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline_2_predictions.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

# Keep the smaller classes from being completely ignored.
# We will train on at most this many examples per class.
MAX_PER_CLASS = 2500


# ============================================================
# Load weak labels
# ============================================================

def load_training_data() -> pd.DataFrame:

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAINING_PATH}"
        )

    rows = []

    with TRAINING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            required = {
                "interaction_id",
                "text",
                "intent",
            }

            if not required.issubset(item):
                continue

            rows.append(
                {
                    "interaction_id": str(
                        item["interaction_id"]
                    ),
                    "text": str(
                        item["text"]
                    ).strip(),
                    "intent": str(
                        item["intent"]
                    ).strip(),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No training examples were loaded."
        )

    return df


# ============================================================
# Load frozen golden set
# ============================================================

def load_golden_set() -> pd.DataFrame:

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found:\n{GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required = {
        "interaction_id",
        "customer_message",
        "human_intent",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Golden set is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# Remove leakage
# ============================================================

def remove_golden_leakage(
    training: pd.DataFrame,
    golden: pd.DataFrame,
) -> pd.DataFrame:

    golden_ids = set(
        golden["interaction_id"]
        .astype(str)
        .str.strip()
    )

    before = len(training)

    training = training[
        ~training["interaction_id"].isin(
            golden_ids
        )
    ].copy()

    removed = before - len(training)

    print()
    print(
        f"Golden-set examples removed from "
        f"training: {removed}"
    )

    return training


# ============================================================
# Balance training data
# ============================================================

def balance_training_data(
    training: pd.DataFrame,
) -> pd.DataFrame:

    parts = []

    for intent, group in training.groupby(
        "intent",
        sort=True,
    ):

        if len(group) > MAX_PER_CLASS:

            group = group.sample(
                n=MAX_PER_CLASS,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    balanced = pd.concat(
        parts,
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced


# ============================================================
# Build model
# ============================================================

def build_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=4.0,
                ),
            ),
        ]
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 72)
    print("HIVER — BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 72)

    training = load_training_data()
    golden = load_golden_set()

    print()
    print(
        f"Weak training examples loaded: "
        f"{len(training):,}"
    )

    print(
        f"Golden evaluation examples: "
        f"{len(golden):,}"
    )

    # --------------------------------------------------------
    # Prevent evaluation leakage.
    # --------------------------------------------------------

    training = remove_golden_leakage(
        training,
        golden,
    )

    # --------------------------------------------------------
    # Balance training classes.
    # --------------------------------------------------------

    training = balance_training_data(
        training
    )

    print()
    print("Training distribution:")

    print(
        training["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    model = build_model()

    print()
    print("Training TF-IDF + Logistic Regression...")

    model.fit(
        training["text"],
        training["intent"],
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Evaluate against frozen golden set.
    # --------------------------------------------------------

    print()
    print("Evaluating against frozen golden set...")

    y_true = (
        golden["human_intent"]
        .astype(str)
        .str.strip()
    )

    y_pred = model.predict(
        golden["customer_message"]
        .fillna("")
        .astype(str),
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 72)
    print("BASELINE 2 RESULTS")
    print("=" * 72)

    print(
        f"Intent Accuracy : {accuracy:.4f}"
    )

    print(
        f"Intent Macro F1 : {macro_f1:.4f}"
    )

    print(
        f"Intent Weighted F1: {weighted_f1:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix.
    # --------------------------------------------------------

    labels = sorted(
        y_true.unique()
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print("Confusion matrix labels:")

    print(labels)

    print()
    print(matrix)

    # --------------------------------------------------------
    # Save predictions.
    # --------------------------------------------------------

    prediction_df = golden[
        [
            "interaction_id",
            "customer_message",
            "human_intent",
        ]
    ].copy()

    prediction_df[
        "predicted_intent"
    ] = y_pred

    # Store probability of predicted class.
    probabilities = model.predict_proba(
        golden["customer_message"]
        .fillna("")
        .astype(str),
    )

    prediction_df[
        "prediction_confidence"
    ] = probabilities.max(
        axis=1
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        f"Predictions saved to:\n"
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()