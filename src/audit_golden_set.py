from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

LLM_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "llm_classifier_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set_audit.csv"
)


def main() -> None:

    print("=" * 72)
    print("HIVER — GOLDEN SET QUALITY AUDIT")
    print("=" * 72)

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    predictions = pd.read_csv(
        LLM_PATH
    )

    required_golden = {
        "interaction_id",
        "customer_message",
        "human_intent",
    }

    required_predictions = {
        "interaction_id",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "reason",
    }

    missing_golden = (
        required_golden
        - set(golden.columns)
    )

    missing_predictions = (
        required_predictions
        - set(predictions.columns)
    )

    if missing_golden:
        raise ValueError(
            f"Golden set missing: "
            f"{sorted(missing_golden)}"
        )

    if missing_predictions:
        raise ValueError(
            f"Predictions missing: "
            f"{sorted(missing_predictions)}"
        )

    audit = predictions[
        [
            "interaction_id",
            "gold_intent",
            "predicted_intent",
            "confidence",
            "reason",
        ]
    ].merge(
        golden[
            [
                "interaction_id",
                "customer_message",
                "context",
            ]
        ],
        on="interaction_id",
        how="left",
    )

    audit["disagreement"] = (
        audit["gold_intent"]
        != audit["predicted_intent"]
    )

    # High-confidence model disagreements are the first
    # candidates for human re-review.
    audit["high_confidence_disagreement"] = (
        audit["disagreement"]
        & (audit["confidence"] >= 0.85)
    )

    audit["medium_confidence_disagreement"] = (
        audit["disagreement"]
        & (
            (audit["confidence"] >= 0.70)
            & (audit["confidence"] < 0.85)
        )
    )

    audit = audit.sort_values(
        [
            "high_confidence_disagreement",
            "confidence",
        ],
        ascending=[False, False],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Total examples: "
        f"{len(audit)}"
    )

    print(
        f"Disagreements: "
        f"{audit['disagreement'].sum()}"
    )

    print(
        f"High-confidence disagreements "
        f"(>= 0.85): "
        f"{audit['high_confidence_disagreement'].sum()}"
    )

    print(
        f"Medium-confidence disagreements "
        f"(0.70-0.84): "
        f"{audit['medium_confidence_disagreement'].sum()}"
    )

    print()
    print("Top high-confidence disagreements:")

    columns = [
        "interaction_id",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "customer_message",
        "reason",
    ]

    high = audit[
        audit["high_confidence_disagreement"]
    ][columns]

    print(
        high.head(30).to_string(
            index=False
        )
    )

    print()
    print(
        f"Full audit saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()