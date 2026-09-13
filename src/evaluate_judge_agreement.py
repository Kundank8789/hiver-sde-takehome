from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_sample.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_comparison.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_agreement_results.json"
)


def build_historical_evidence(
    retriever,
    customer_message: str,
) -> str:

    cases = retriever.retrieve(
        query=customer_message,
        intent=None,
        candidate_k=10,
        top_k=3,
    )

    blocks = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        customer = str(
            case.get(
                "customer_message",
                "",
            )
        ).strip()

        response = str(
            case.get(
                "agent_response",
                "",
            )
        ).strip()

        if not response:
            continue

        blocks.append(
            f"""
Historical case {index}

Customer:
{customer}

AppleSupport response:
{response}
""".strip()
        )

    return (
        "\n\n".join(blocks)
        if blocks
        else "No historical evidence retrieved."
    )


def main() -> None:

    from response_judge import ResponseJudge
    from retriever import HistoricalRetriever

    print("=" * 72)
    print("HIVER — HUMAN VS LLM JUDGE AGREEMENT")
    print("=" * 72)

    df = pd.read_csv(
        SAMPLE_PATH
    )

    required = {
        "interaction_id",
        "customer_message",
        "reply",
        "human_overall_score",
        "human_grounded",
        "human_relevant",
        "human_helpful",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(df) != 30:
        raise ValueError(
            f"Expected 30 rows, found {len(df)}"
        )

    rating_columns = [
        "human_overall_score",
        "human_grounded",
        "human_relevant",
        "human_helpful",
    ]

    for column in rating_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if (
            df[column].isna().any()
            or ~df[column].between(1, 5).all()
        ):
            raise ValueError(
                f"Invalid ratings in {column}"
            )

    retriever = HistoricalRetriever()
    judge = ResponseJudge()

    results = []

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        customer_message = str(
            row["customer_message"]
        ).strip()

        reply = str(
            row["reply"]
        ).strip()

        evidence = build_historical_evidence(
            retriever,
            customer_message,
        )

        started = time.perf_counter()

        try:

            judgment = judge.judge(
                customer_message=customer_message,
                historical_evidence=evidence,
                draft_reply=reply,
                should_escalate=(
                    str(
                        row.get(
                            "gold_escalation",
                            "",
                        )
                    ).upper()
                    == "TRUE"
                ),
            )

            error = ""

        except Exception as exc:

            judgment = {
                "correctness": 0,
                "groundedness": 0,
                "relevance": 0,
                "helpfulness": 0,
                "style": 0,
                "unsupported_claim": False,
                "overall": 0,
                "reason": "",
            }

            error = str(exc)

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
                "reply": reply,

                "human_overall_score": int(
                    row["human_overall_score"]
                ),
                "human_grounded": int(
                    row["human_grounded"]
                ),
                "human_relevant": int(
                    row["human_relevant"]
                ),
                "human_helpful": int(
                    row["human_helpful"]
                ),

                "judge_overall": int(
                    judgment.get(
                        "overall",
                        0,
                    )
                ),
                "judge_groundedness": int(
                    judgment.get(
                        "groundedness",
                        0,
                    )
                ),
                "judge_relevance": int(
                    judgment.get(
                        "relevance",
                        0,
                    )
                ),
                "judge_helpfulness": int(
                    judgment.get(
                        "helpfulness",
                        0,
                    )
                ),
                "judge_correctness": int(
                    judgment.get(
                        "correctness",
                        0,
                    )
                ),
                "judge_style": int(
                    judgment.get(
                        "style",
                        0,
                    )
                ),
                "judge_unsupported_claim": bool(
                    judgment.get(
                        "unsupported_claim",
                        False,
                    )
                ),
                "judge_reason": str(
                    judgment.get(
                        "reason",
                        "",
                    )
                ),
                "latency_seconds": latency,
                "error": error,
            }
        )

        print(
            f"Processed {position}/30"
        )

    comparison = pd.DataFrame(
        results
    )

    valid = comparison[
        comparison["judge_overall"] > 0
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No valid judge results were produced."
        )

    # --------------------------------------------------
    # Overall-score agreement
    # --------------------------------------------------

    human_overall = valid[
        "human_overall_score"
    ].astype(int)

    judge_overall = valid[
        "judge_overall"
    ].astype(int)

    exact_agreement = float(
        (
            human_overall
            == judge_overall
        ).mean()
    )

    within_one = float(
        (
            (
                human_overall
                - judge_overall
            ).abs()
            <= 1
        ).mean()
    )

    kappa = float(
        cohen_kappa_score(
            human_overall,
            judge_overall,
            weights="quadratic",
        )
    )

    correlation, correlation_p = (
        spearmanr(
            human_overall,
            judge_overall,
        )
    )

    # --------------------------------------------------
    # Dimension agreement
    # --------------------------------------------------

    dimension_results = {}

    for human_column, judge_column in [
        (
            "human_grounded",
            "judge_groundedness",
        ),
        (
            "human_relevant",
            "judge_relevance",
        ),
        (
            "human_helpful",
            "judge_helpfulness",
        ),
    ]:

        human = valid[
            human_column
        ].astype(int)

        judged = valid[
            judge_column
        ].astype(int)

        dimension_results[
            human_column.replace(
                "human_",
                "",
            )
        ] = {
            "exact_agreement": float(
                (
                    human == judged
                ).mean()
            ),
            "within_one": float(
                (
                    (
                        human - judged
                    ).abs()
                    <= 1
                ).mean()
            ),
        }

    runtime_errors = int(
        (
            comparison["error"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ).sum()
    )

    unsupported_claim_rate = float(
        valid[
            "judge_unsupported_claim"
        ].mean()
    )

    average_judge_latency = float(
        comparison[
            "latency_seconds"
        ].mean()
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print()
    print("=" * 72)
    print("AGREEMENT RESULTS")
    print("=" * 72)

    print(
        f"Valid judge results: "
        f"{len(valid)}/{len(comparison)}"
    )

    print(
        f"Exact agreement: "
        f"{exact_agreement:.2%}"
    )

    print(
        f"Within ±1 point: "
        f"{within_one:.2%}"
    )

    print(
        f"Quadratic weighted kappa: "
        f"{kappa:.4f}"
    )

    print(
        f"Spearman correlation: "
        f"{correlation:.4f}"
    )

    print(
        f"Correlation p-value: "
        f"{correlation_p:.4f}"
    )

    print(
        f"Judge unsupported-claim rate: "
        f"{unsupported_claim_rate:.2%}"
    )

    print(
        f"Judge runtime errors: "
        f"{runtime_errors}"
    )

    print(
        f"Average judge latency: "
        f"{average_judge_latency:.3f}s"
    )

    print(
        "\nDimension agreement:"
    )

    for dimension, values in (
        dimension_results.items()
    ):

        print(
            f"  {dimension:12s} "
            f"exact={values['exact_agreement']:.2%} "
            f"within±1={values['within_one']:.2%}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = {
        "sample_size": len(comparison),
        "valid_judgments": len(valid),
        "overall": {
            "exact_agreement": exact_agreement,
            "within_one": within_one,
            "quadratic_weighted_kappa": kappa,
            "spearman_correlation": float(
                correlation
            ),
            "spearman_p_value": float(
                correlation_p
            ),
        },
        "dimensions": dimension_results,
        "unsupported_claim_rate": (
            unsupported_claim_rate
        ),
        "runtime_errors": runtime_errors,
        "average_latency_seconds": (
            average_judge_latency
        ),
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
        f"\nComparison saved to:\n{OUTPUT_PATH}"
    )

    print(
        f"Summary saved to:\n{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()