from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "evaluation" / "golden_set.csv"


GENERIC_REPLY = (
    "Thanks for reaching out. We're sorry you're experiencing "
    "this issue. Please contact Apple Support so we can look "
    "into it further."
)


def main() -> None:
    print("=" * 72)
    print("BASELINE 1 — TRIVIAL BASELINE")
    print("=" * 72)

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found:\n{GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required_columns = {
        "human_intent",
        "should_escalate",
        "customer_message",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------
    # Intent baseline
    # ------------------------------------------------------

    # Golden set is currently balanced, so use the first
    # deterministic class alphabetically as the trivial
    # fixed prediction.
    intents = sorted(
        df["human_intent"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if not intents:
        raise ValueError("No intent labels found.")

    predicted_intent = intents[0]

    y_true_intent = (
        df["human_intent"]
        .astype(str)
        .str.strip()
    )

    y_pred_intent = [
        predicted_intent
        for _ in range(len(df))
    ]

    intent_accuracy = accuracy_score(
        y_true_intent,
        y_pred_intent,
    )

    intent_macro_f1 = f1_score(
        y_true_intent,
        y_pred_intent,
        average="macro",
        zero_division=0,
    )

    # ------------------------------------------------------
    # Escalation baseline
    # ------------------------------------------------------

    y_true_escalation = (
        df["should_escalate"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    y_pred_escalation = [
        "TRUE"
        for _ in range(len(df))
    ]

    escalation_accuracy = accuracy_score(
        y_true_escalation,
        y_pred_escalation,
    )

    escalation_precision = precision_score(
        y_true_escalation,
        y_pred_escalation,
        pos_label="TRUE",
        zero_division=0,
    )

    escalation_recall = recall_score(
        y_true_escalation,
        y_pred_escalation,
        pos_label="TRUE",
        zero_division=0,
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("Dataset:")
    print(f"  Examples: {len(df)}")

    print()
    print("Intent baseline:")
    print(
        f"  Always predict: {predicted_intent}"
    )
    print(
        f"  Accuracy:       {intent_accuracy:.4f}"
    )
    print(
        f"  Macro F1:       {intent_macro_f1:.4f}"
    )

    print()
    print("Escalation baseline:")
    print("  Always predict: TRUE")
    print(
        f"  Accuracy:       {escalation_accuracy:.4f}"
    )
    print(
        f"  Precision:      {escalation_precision:.4f}"
    )
    print(
        f"  Recall:         {escalation_recall:.4f}"
    )

    print()
    print("Generic reply:")
    print(f"  {GENERIC_REPLY}")

    print()
    print("Intent classification report:")
    print(
        classification_report(
            y_true_intent,
            y_pred_intent,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()