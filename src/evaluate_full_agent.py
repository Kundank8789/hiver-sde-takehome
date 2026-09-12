from __future__ import annotations

import json
import time
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

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "full_agent_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "full_agent_results.json"
)


def normalize_bool(value: object) -> str:
    return (
        str(value)
        .strip()
        .upper()
    )


def main() -> None:

    from support_agent import SupportAgent

    print("=" * 72)
    print("HIVER — FINAL FULL-AGENT EVALUATION")
    print("=" * 72)

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found:\n{GOLDEN_PATH}"
        )

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    required = {
        "interaction_id",
        "customer_message",
        "context",
        "human_intent",
        "should_escalate",
    }

    missing = required - set(
        golden.columns
    )

    if missing:
        raise ValueError(
            f"Golden set missing columns: {sorted(missing)}"
        )

    if len(golden) != 200:
        raise ValueError(
            f"Expected exactly 200 frozen examples, "
            f"found {len(golden)}."
        )

    agent = SupportAgent()

    results: list[dict] = []

    print(
        f"\nFrozen golden examples: {len(golden)}"
    )

    print(
        "\nWARNING: this performs live Groq calls."
    )

    for position, (_, row) in enumerate(
        golden.iterrows(),
        start=1,
    ):

        customer_message = (
            str(
                row["customer_message"]
            )
            .strip()
        )

        context = ""

        if pd.notna(
            row["context"]
        ):
            context = (
                str(
                    row["context"]
                )
                .strip()
            )

        started = time.perf_counter()

        error = ""

        try:

            output = agent.handle(
                customer_message=customer_message,
                context=context,
                top_k=5,
            )

        except Exception as exc:

            error = str(exc)

            output = {
                "intent": "OTHER_UNCLEAR",
                "confidence": 0.0,
                "classification_reason": "",
                "should_escalate": True,
                "escalation_reason": (
                    "Agent execution failed; "
                    "human review required."
                ),
                "reply": (
                    "We'd like to help. "
                    "Please contact support so the issue "
                    "can be reviewed."
                ),
                "grounded": False,
                "evidence_used": 0,
                "evidence_score": 0.0,
                "retrieved_cases": [],
            }

        latency = (
            time.perf_counter()
            - started
        )

        results.append(
            {
                "interaction_id": str(
                    row["interaction_id"]
                ),
                "customer_message": customer_message,
                "gold_intent": str(
                    row["human_intent"]
                ).strip(),
                "predicted_intent": str(
                    output["intent"]
                ).strip(),
                "intent_confidence": float(
                    output["confidence"]
                ),
                "classification_reason": (
                    output.get(
                        "classification_reason",
                        "",
                    )
                ),
                "gold_escalation": normalize_bool(
                    row["should_escalate"]
                ),
                "predicted_escalation": (
                    "TRUE"
                    if output["should_escalate"]
                    else "FALSE"
                ),
                "escalation_reason": output[
                    "escalation_reason"
                ],
                "reply": output["reply"],
                "grounded": bool(
                    output["grounded"]
                ),
                "evidence_used": int(
                    output["evidence_used"]
                ),
                "evidence_score": float(
                    output["evidence_score"]
                ),
                "latency_seconds": latency,
                "error": error,
            }
        )

        if position % 10 == 0:
            print(
                f"Processed {position}/{len(golden)}"
            )

    df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------
    # Intent metrics
    # ------------------------------------------------------

    intent_true = df[
        "gold_intent"
    ]

    intent_pred = df[
        "predicted_intent"
    ]

    intent_accuracy = accuracy_score(
        intent_true,
        intent_pred,
    )

    intent_macro_f1 = f1_score(
        intent_true,
        intent_pred,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        intent_true,
        intent_pred,
        average="weighted",
        zero_division=0,
    )

    # ------------------------------------------------------
    # Escalation metrics
    # ------------------------------------------------------

    escalation_true = df[
        "gold_escalation"
    ]

    escalation_pred = df[
        "predicted_escalation"
    ]

    escalation_accuracy = accuracy_score(
        escalation_true,
        escalation_pred,
    )

    escalation_precision = precision_score(
        escalation_true,
        escalation_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    escalation_recall = recall_score(
        escalation_true,
        escalation_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    escalation_f1 = f1_score(
        escalation_true,
        escalation_pred,
        pos_label="TRUE",
        zero_division=0,
    )

    runtime_errors = int(
        (
            df["error"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ).sum()
    )

    average_latency = float(
        df["latency_seconds"].mean()
    )

    median_latency = float(
        df["latency_seconds"].median()
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL AGENT RESULTS")
    print("=" * 72)

    print(
        f"\nINTENT"
    )

    print(
        f"  Accuracy   : {intent_accuracy:.4f}"
    )

    print(
        f"  Macro F1   : {intent_macro_f1:.4f}"
    )

    print(
        f"  Weighted F1: {intent_weighted_f1:.4f}"
    )

    print(
        "\nESCALATION"
    )

    print(
        f"  Accuracy : {escalation_accuracy:.4f}"
    )

    print(
        f"  Precision: {escalation_precision:.4f}"
    )

    print(
        f"  Recall   : {escalation_recall:.4f}"
    )

    print(
        f"  F1       : {escalation_f1:.4f}"
    )

    print(
        "\nOPERATIONS"
    )

    print(
        f"  Average latency: {average_latency:.3f}s"
    )

    print(
        f"  Median latency : {median_latency:.3f}s"
    )

    print(
        f"  Runtime errors : {runtime_errors}"
    )

    print(
        f"  Grounded replies: "
        f"{df['grounded'].mean():.2%}"
    )

    print(
        "\nIntent classification report:"
    )

    print(
        classification_report(
            intent_true,
            intent_pred,
            zero_division=0,
        )
    )

    print(
        "\nEscalation classification report:"
    )

    print(
        classification_report(
            escalation_true,
            escalation_pred,
            zero_division=0,
        )
    )

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    summary = {
        "model": "openai/gpt-oss-20b",
        "golden_examples": len(df),
        "intent": {
            "accuracy": intent_accuracy,
            "macro_f1": intent_macro_f1,
            "weighted_f1": intent_weighted_f1,
        },
        "escalation": {
            "accuracy": escalation_accuracy,
            "precision": escalation_precision,
            "recall": escalation_recall,
            "f1": escalation_f1,
        },
        "operations": {
            "average_latency_seconds": average_latency,
            "median_latency_seconds": median_latency,
            "runtime_errors": runtime_errors,
            "grounded_reply_rate": float(
                df["grounded"].mean()
            ),
        },
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