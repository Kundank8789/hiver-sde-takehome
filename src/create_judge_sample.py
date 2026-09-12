from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "full_agent_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_sample.csv"
)


def main() -> None:

    print("=" * 72)
    print("HIVER — CREATE HUMAN/JUDGE REPLY-QUALITY SAMPLE")
    print("=" * 72)

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    if len(df) != 200:
        raise ValueError(
            f"Expected 200 agent predictions, found {len(df)}"
        )

    eligible = df[
        df["reply"].fillna("").astype(str).str.strip() != ""
    ].copy()

    random.seed(42)

    sample = (
        eligible
        .sample(
            n=min(30, len(eligible)),
            random_state=42,
        )
        .copy()
    )

    output = sample[
        [
            "interaction_id",
            "customer_message",
            "reply",
            "grounded",
            "evidence_used",
            "evidence_score",
        ]
    ].copy()

    output["human_overall_score"] = ""
    output["human_grounded"] = ""
    output["human_relevant"] = ""
    output["human_helpful"] = ""
    output["human_notes"] = ""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSample created: {len(output)}"
    )

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )

    print(
        "\nHuman reviewer fields:"
    )

    print(
        "human_overall_score: 1-5"
    )

    print(
        "human_grounded: 1-5"
    )

    print(
        "human_relevant: 1-5"
    )

    print(
        "human_helpful: 1-5"
    )

    print(
        "human_notes: optional"
    )


if __name__ == "__main__":
    main()