from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "hybrid_classifier_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "hybrid_classifier_results.json"
)


def main() -> None:

    from hybrid_intent_classifier import (
        HybridIntentClassifier
    )

    print("=" * 72)
    print("HIVER — HYBRID INTENT CLASSIFIER EVALUATION")
    print("=" * 72)

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    if len(golden) != 200:
        raise ValueError(
            f"Expected 200 golden examples, found {len(golden)}."
        )

    classifier = HybridIntentClassifier(
        llm_classifier=None
    )

    results = []

    for position, (_, row) in enumerate(
        golden.iterrows(),
        start=1,
    ):

        message = str(
            row["customer_message"]
        ).strip()

        context = ""

        if pd.notna(
            row.get("context")
        ):
            context = str(
                row["context"]
            ).strip()

        result = classifier.classify(
            customer_message=message,
            context=context,
        )

        results.append(
            {
                "interaction_id": str(
                    row["interaction_id"]
                ),
                "gold_intent": str(
                    row["human_intent"]
                ).strip(),
                "predicted_intent": result[
                    "intent"
                ],
                "confidence": float(
                    result["confidence"]
                ),
                "method": result.get(
                    "method",
                    "",
                ),
                "reason": result["reason"],
            }
        )

        if position % 20 == 0:
            print(
                f"Processed {position}/200"
            )

    df = pd.DataFrame(
        results
    )

    y_true = df[
        "gold_intent"
    ]

    y_pred = df[
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

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    method_distribution = (
        df["method"]
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 72)
    print("HYBRID CLASSIFIER RESULTS")
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
        "\nMethod distribution:"
    )

    for method, count in method_distribution.items():
        print(
            f"  {method:20s} {count}"
        )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = {
        "examples": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "method_distribution": method_distribution,
        "labels": labels,
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

    print(
        f"\nPredictions saved to:\n{OUTPUT_PATH}"
    )

    print(
        f"Summary saved to:\n{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()