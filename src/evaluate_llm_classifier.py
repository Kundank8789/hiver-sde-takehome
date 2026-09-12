from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from intent_classifier import IntentClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "llm_classifier_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "llm_classifier_results.json"
)


RANDOM_STATE = 42


def build_context(row: pd.Series) -> str:
    context = row.get("context", "")

    if pd.isna(context):
        return ""

    return str(context).strip()


def main() -> None:

    print("=" * 72)
    print("HIVER — LLM INTENT CLASSIFIER EVALUATION")
    print("=" * 72)

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found:\n{GOLDEN_PATH}"
        )

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    required_columns = {
        "interaction_id",
        "customer_message",
        "context",
        "human_intent",
    }

    missing = (
        required_columns
        - set(golden.columns)
    )

    if missing:
        raise ValueError(
            f"Golden set missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"\nGolden examples: {len(golden)}"
    )

    classifier = IntentClassifier()

    predictions = []

    print("\nRunning classifier...")

    for position, (_, row) in enumerate(
        golden.iterrows(),
        start=1,
    ):

        customer_message = str(
            row["customer_message"]
        ).strip()

        context = build_context(row)

        started = time.perf_counter()

        try:
            result = classifier.classify(
                customer_message=customer_message,
                context=context,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            predictions.append(
                {
                    "interaction_id": str(
                        row["interaction_id"]
                    ),
                    "customer_message": customer_message,
                    "gold_intent": str(
                        row["human_intent"]
                    ).strip(),
                    "predicted_intent": result[
                        "intent"
                    ],
                    "confidence": float(
                        result["confidence"]
                    ),
                    "reason": result["reason"],
                    "latency_seconds": elapsed,
                    "error": "",
                }
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            predictions.append(
                {
                    "interaction_id": str(
                        row["interaction_id"]
                    ),
                    "customer_message": customer_message,
                    "gold_intent": str(
                        row["human_intent"]
                    ).strip(),
                    "predicted_intent": "OTHER_UNCLEAR",
                    "confidence": 0.0,
                    "reason": "",
                    "latency_seconds": elapsed,
                    "error": str(exc),
                }
            )

        if position % 10 == 0 or position == len(golden):
            print(
                f"Processed {position}/{len(golden)}"
            )

    predictions_df = pd.DataFrame(
        predictions
    )

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    y_true = predictions_df[
        "gold_intent"
    ]

    y_pred = predictions_df[
        "predicted_intent"
    ]

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

    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # ------------------------------------------------------
    # Additional operational metrics
    # ------------------------------------------------------

    errors = int(
        (
            predictions_df["error"]
            .astype(str)
            .str.len()
            > 0
        ).sum()
    )

    average_latency = float(
        predictions_df[
            "latency_seconds"
        ].mean()
    )

    exact_accuracy = float(
        (
            predictions_df["gold_intent"]
            == predictions_df["predicted_intent"]
        ).mean()
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print()
    print("=" * 72)
    print("LLM CLASSIFIER RESULTS")
    print("=" * 72)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : {weighted_f1:.4f}"
    )

    print(
        f"Average latency: {average_latency:.3f}s"
    )

    print(
        f"API/runtime errors: {errors}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix labels:")

    print(labels)

    print("\nConfusion matrix:")

    print(matrix)

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # Save machine-readable summary
    # ------------------------------------------------------

    summary = {
        "model": classifier.client.model,
        "examples": len(predictions_df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "average_latency_seconds": average_latency,
        "runtime_errors": errors,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        f"Predictions saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()