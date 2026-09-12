from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "llm_classifier_predictions.csv"
)


def main() -> None:
    print("=" * 72)
    print("HIVER — LLM CLASSIFIER FAILURE ANALYSIS")
    print("=" * 72)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found:\n{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    required = {
        "interaction_id",
        "customer_message",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "reason",
        "error",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["correct"] = (
        df["gold_intent"]
        == df["predicted_intent"]
    )

    errors = df[~df["correct"]].copy()

    print(
        f"\nTotal examples: {len(df)}"
    )

    print(
        f"Incorrect predictions: {len(errors)}"
    )

    print(
        f"Correct predictions: {df['correct'].sum()}"
    )

    print("\nMost common confusion pairs:")

    confusion = (
        errors.groupby(
            [
                "gold_intent",
                "predicted_intent",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    print(
        confusion.head(15).to_string(
            index=False
        )
    )

    print("\nLowest-confidence errors:")

    low_conf = errors.sort_values(
        "confidence",
        ascending=True,
    )

    columns = [
        "interaction_id",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "customer_message",
        "reason",
    ]

    print(
        low_conf[columns]
        .head(15)
        .to_string(index=False)
    )

    print("\nHighest-confidence errors:")

    high_conf = errors.sort_values(
        "confidence",
        ascending=False,
    )

    print(
        high_conf[columns]
        .head(15)
        .to_string(index=False)
    )

    print("\nRuntime/API errors:")

    runtime_errors = df[
        df["error"].fillna("").astype(str).str.strip() != ""
    ]

    print(
        f"Count: {len(runtime_errors)}"
    )

    if not runtime_errors.empty:
        print(
            runtime_errors[
                [
                    "interaction_id",
                    "error",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()