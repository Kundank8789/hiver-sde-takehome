from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_AGENT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "full_agent_predictions.csv"
)

HYBRID_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "hybrid_classifier_predictions.csv"
)

JUDGE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_comparison.csv"
)

RETRIEVAL_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_evaluation.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "failure_analysis.md"
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


def main() -> None:
    print("=" * 72)
    print("HIVER — FINAL FAILURE ANALYSIS")
    print("=" * 72)

    full_agent = load_csv(FULL_AGENT_PATH)
    hybrid = load_csv(HYBRID_PATH)
    judge = load_csv(JUDGE_PATH)

    retrieval = None

    if RETRIEVAL_PATH.exists():
        with RETRIEVAL_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            retrieval = json.load(file)

    # --------------------------------------------------
    # 1. Intent failure analysis
    # --------------------------------------------------

    full_agent["intent_correct"] = (
        full_agent["gold_intent"]
        == full_agent["predicted_intent"]
    )

    intent_errors = full_agent[
        ~full_agent["intent_correct"]
    ].copy()

    confusion = (
        intent_errors.groupby(
            [
                "gold_intent",
                "predicted_intent",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # --------------------------------------------------
    # 2. Escalation failure analysis
    # --------------------------------------------------

    full_agent["escalation_correct"] = (
        full_agent["gold_escalation"]
        == full_agent["predicted_escalation"]
    )

    escalation_errors = full_agent[
        ~full_agent["escalation_correct"]
    ].copy()

    escalation_confusion = (
        escalation_errors.groupby(
            [
                "gold_escalation",
                "predicted_escalation",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # --------------------------------------------------
    # 3. High-confidence intent failures
    # --------------------------------------------------

    if "intent_confidence" in full_agent.columns:

        high_confidence_errors = (
            intent_errors
            .sort_values(
                "intent_confidence",
                ascending=False,
            )
            .head(10)
        )

    else:
        high_confidence_errors = (
            intent_errors.head(10)
        )

    # --------------------------------------------------
    # 4. Runtime failures
    # --------------------------------------------------

    runtime_errors = full_agent[
        full_agent["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]

    # --------------------------------------------------
    # 5. Judge/human evidence
    # --------------------------------------------------

    judge_summary = {}

    if not judge.empty:

        judge_summary = {
            "sample_size": len(judge),
            "human_overall_mean": float(
                judge[
                    "human_overall_score"
                ].mean()
            ),
            "judge_overall_mean": float(
                judge[
                    "judge_overall"
                ].mean()
            ),
            "unsupported_claim_rate": float(
                judge[
                    "judge_unsupported_claim"
                ].mean()
            ),
        }

    # --------------------------------------------------
    # 6. Write report
    # --------------------------------------------------

    lines: list[str] = []

    lines.append(
        "# Hiver AI Support Agent — Failure Analysis"
    )

    lines.append("")

    lines.append(
        "This report summarizes observed failure modes from the "
        "frozen 200-example evaluation and the 30-example human/judge "
        "reply-quality validation sample."
    )

    lines.append("")

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    lines.append("## 1. Evaluation Snapshot")
    lines.append("")

    intent_accuracy = float(
        full_agent["intent_correct"].mean()
    )

    escalation_accuracy = float(
        full_agent["escalation_correct"].mean()
    )

    grounded_rate = float(
        full_agent["grounded"].mean()
    )

    lines.append(
        f"- Full-agent intent accuracy: "
        f"{intent_accuracy:.2%}"
    )

    lines.append(
        f"- Full-agent escalation accuracy: "
        f"{escalation_accuracy:.2%}"
    )

    lines.append(
        f"- Self-reported grounded-output rate: "
        f"{grounded_rate:.2%}"
    )

    lines.append(
        f"- Runtime errors: "
        f"{len(runtime_errors)}"
    )

    if retrieval:
        lexical = retrieval.get(
            "lexical",
            {}
        )

        reranked = retrieval.get(
            "intent_aware_reranking",
            retrieval.get(
                "intent_aware",
                {}
            ),
        )

        lines.append("")

        lines.append(
            "### Retrieval"
        )

        if lexical:
            lines.append(
                f"- Lexical Recall@1: "
                f"{lexical.get('recall_at_1', 'N/A')}"
            )
            lines.append(
                f"- Lexical Recall@3: "
                f"{lexical.get('recall_at_3', 'N/A')}"
            )
            lines.append(
                f"- Lexical Recall@5: "
                f"{lexical.get('recall_at_5', 'N/A')}"
            )

        if reranked:
            lines.append(
                f"- Intent-aware Recall@1: "
                f"{reranked.get('recall_at_1', 'N/A')}"
            )
            lines.append(
                f"- Intent-aware Recall@3: "
                f"{reranked.get('recall_at_3', 'N/A')}"
            )
            lines.append(
                f"- Intent-aware Recall@5: "
                f"{reranked.get('recall_at_5', 'N/A')}"
            )

    # --------------------------------------------------
    # Top failure modes
    # --------------------------------------------------

    lines.append("")
    lines.append("## 2. Top Failure Modes")
    lines.append("")

    if confusion.empty:
        lines.append(
            "No intent errors were observed."
        )
    else:

        top_confusions = confusion.head(5)

        for number, row in enumerate(
            top_confusions.itertuples(
                index=False
            ),
            start=1,
        ):

            gold = row.gold_intent
            predicted = row.predicted_intent
            count = row.count

            lines.append(
                f"### Failure Mode {number}: "
                f"{gold} → {predicted}"
            )

            lines.append("")

            lines.append(
                f"Observed {count} example(s) with "
                f"gold intent `{gold}` predicted as "
                f"`{predicted}`."
            )

            examples = intent_errors[
                (
                    intent_errors["gold_intent"]
                    == gold
                )
                & (
                    intent_errors[
                        "predicted_intent"
                    ]
                    == predicted
                )
            ].head(2)

            lines.append("")

            lines.append(
                "**Examples:**"
            )

            for example in examples.itertuples(
                index=False
            ):

                message = str(
                    example.customer_message
                ).replace(
                    "\n",
                    " ",
                )

                lines.append(
                    f"- `{example.interaction_id}` — "
                    f"{message[:240]}"
                )

            lines.append("")

            lines.append(
                "**Hypothesis:** "
                "The classifier is confusing overlapping intent boundaries "
                "or relying on a symptom that is not the customer's primary "
                "support need."
            )

            lines.append("")

    # --------------------------------------------------
    # Escalation failures
    # --------------------------------------------------

    lines.append(
        "## 3. Escalation Failure Modes"
    )

    lines.append("")

    if escalation_confusion.empty:

        lines.append(
            "No escalation errors were observed."
        )

    else:

        for row in escalation_confusion.itertuples(
            index=False
        ):

            lines.append(
                f"- Gold `{row.gold_escalation}` → "
                f"Predicted `{row.predicted_escalation}`: "
                f"{row.count} example(s)"
            )

    lines.append("")

    # --------------------------------------------------
    # High-confidence failures
    # --------------------------------------------------

    lines.append(
        "## 4. High-Confidence Intent Errors"
    )

    lines.append("")

    for example in high_confidence_errors.itertuples(
        index=False
    ):

        confidence = getattr(
            example,
            "intent_confidence",
            None,
        )

        lines.append(
            f"- `{example.interaction_id}`: "
            f"{example.gold_intent} → "
            f"{example.predicted_intent}; "
            f"confidence={confidence}"
        )

        lines.append(
            f"  - {str(example.customer_message)[:300]}"
        )

    # --------------------------------------------------
    # Judge limitations
    # --------------------------------------------------

    lines.append("")
    lines.append(
        "## 5. Reply-Quality Judge Limitations"
    )

    lines.append("")

    lines.append(
        "The LLM judge was evaluated against 30 human-rated "
        "responses. Judge-human agreement was weak, so judge scores "
        "are treated as diagnostic rather than ground truth."
    )

    if judge_summary:

        lines.append("")

        lines.append(
            f"- Human review sample: "
            f"{judge_summary['sample_size']}"
        )

        lines.append(
            f"- Human overall mean: "
            f"{judge_summary['human_overall_mean']:.2f}"
        )

        lines.append(
            f"- Judge overall mean: "
            f"{judge_summary['judge_overall_mean']:.2f}"
        )

        lines.append(
            f"- Judge unsupported-claim rate: "
            f"{judge_summary['unsupported_claim_rate']:.2%}"
        )

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    lines.append("")
    lines.append(
        "## 6. Operational Issues"
    )

    lines.append("")

    if len(runtime_errors) == 0:

        lines.append(
            "No runtime errors were recorded in the final "
            "full-agent evaluation."
        )

    else:

        for row in runtime_errors.head(10).itertuples(
            index=False
        ):

            lines.append(
                f"- `{row.interaction_id}`: "
                f"{row.error}"
            )

    lines.append("")

    lines.append(
        "## 7. What We Would Improve With One More Week"
    )

    lines.append("")

    lines.extend(
        [
            "1. Replace the lexical retrieval layer with a semantic "
            "embedding retriever and evaluate Recall@K again.",
            "2. Expand human evaluation beyond 200 intent examples and "
            "stratify by ambiguous intent boundaries.",
            "3. Calibrate classifier confidence using a held-out "
            "calibration set instead of raw model confidence.",
            "4. Improve primary-intent precedence for multi-symptom "
            "Twitter messages.",
            "5. Add stronger response-grounding checks before a reply "
            "is exposed to a customer.",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 72)

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()