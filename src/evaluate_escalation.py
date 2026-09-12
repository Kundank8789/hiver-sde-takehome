from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "llm_classifier_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "escalation_baseline_results.json"
)


def main() -> None:

    print("=" * 72)
    print("HIVER — ESCALATION POLICY EVALUATION")
    print("=" * 72)

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    merged = golden[
        [
            "interaction_id",
            "should_escalate",
            "customer_message",
        ]
    ].merge(
        predictions[
            [
                "interaction_id",
                "predicted_intent",
                "confidence",
            ]
        ],
        on="interaction_id",
        how="inner",
    )

    if len(merged) != len(golden):
        raise RuntimeError(
            "Not all golden examples were matched."
        )

    y_true = (
        merged["should_escalate"]
        .astype(str)
        .str.upper()
    )

    y_pred = []

    for _, row in merged.iterrows():

        intent = row["predicted_intent"]
        confidence = float(
            row["confidence"]
        )

        message = str(
            row["customer_message"]
        )

        # Import locally so the script has a simple startup path.
        from escalation import decide_escalation

        decision = decide_escalation(
            intent=intent,
            confidence=confidence,
            customer_message=message,
        )

        y_pred.append(
            "TRUE"
            if decision.should_escalate
            else "FALSE"
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print()
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    results = {
        "examples": len(merged),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()