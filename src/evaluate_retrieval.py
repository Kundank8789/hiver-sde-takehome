from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

INTERACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "apple_support_interactions.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_evaluation.json"
)


def load_interaction_intents() -> dict[str, str]:
    """
    Build interaction_id -> historical intent mapping.

    We use a deterministic keyword heuristic only to identify
    the intent of retrieved historical examples.

    These labels are NOT golden labels.
    """

    keyword_map = {
        "IOS_UPDATE": [
            "ios",
            "update",
            "upgrade",
            "software update",
        ],
        "BATTERY_POWER": [
            "battery",
            "charging",
            "charger",
            "drain",
        ],
        "PERFORMANCE_STABILITY": [
            "slow",
            "lag",
            "freeze",
            "crash",
            "restart",
            "stuck",
        ],
        "CONNECTIVITY": [
            "wifi",
            "bluetooth",
            "cellular",
            "connect",
            "connection",
        ],
        "APPS_MEDIA": [
            "apple music",
            "itunes",
            "imovie",
            "facetime",
            "messages",
            "photos",
            "youtube",
        ],
        "DEVICE_HARDWARE": [
            "screen",
            "display",
            "speaker",
            "keyboard",
            "camera",
            "broken",
            "cracked",
            "hardware",
        ],
        "FEATURE_HOW_TO": [
            "how do i",
            "how can i",
            "how to",
            "where is",
            "how does",
        ],
        "ACCOUNT_SECURITY": [
            "apple id",
            "password",
            "account",
            "verification",
            "verify",
            "locked",
            "login",
        ],
        "PURCHASE_REPAIR_WARRANTY": [
            "order",
            "purchase",
            "buy",
            "refund",
            "warranty",
            "repair",
            "appointment",
            "reservation",
            "replacement",
        ],
    }

    mapping = {}

    with INTERACTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            text = item[
                "customer_message"
            ].lower()

            scores = {}

            for intent, keywords in keyword_map.items():

                score = sum(
                    keyword in text
                    for keyword in keywords
                )

                if score > 0:
                    scores[intent] = score

            if not scores:
                continue

            best_score = max(
                scores.values()
            )

            winners = [
                intent
                for intent, score
                in scores.items()
                if score == best_score
            ]

            # Only keep unambiguous historical cases.
            if len(winners) != 1:
                continue

            mapping[
                str(item["interaction_id"])
            ] = winners[0]

    return mapping


def evaluate(
    retriever,
    golden: pd.DataFrame,
    historical_intents: dict[str, str],
    use_intent_reranking: bool,
) -> dict[str, float]:

    hits = {
        1: 0,
        3: 0,
        5: 0,
    }

    evaluated = 0

    for _, row in golden.iterrows():

        query = str(
            row["customer_message"]
        )

        gold_intent = str(
            row["human_intent"]
        ).strip()

        results = retriever.retrieve(
            query=query,
            intent=(
                gold_intent
                if use_intent_reranking
                else None
            ),
            candidate_k=20,
            top_k=5,
        )

        if not results:
            continue

        evaluated += 1

        retrieved_intents = [
            historical_intents.get(
                str(result["interaction_id"])
            )
            for result in results
        ]

        retrieved_intents = [
            intent
            for intent in retrieved_intents
            if intent is not None
        ]

        for k in (1, 3, 5):

            if gold_intent in retrieved_intents[:k]:
                hits[k] += 1

    if evaluated == 0:
        raise RuntimeError(
            "No golden examples could be evaluated."
        )

    return {
        "evaluated_examples": evaluated,
        "recall_at_1": hits[1] / evaluated,
        "recall_at_3": hits[3] / evaluated,
        "recall_at_5": hits[5] / evaluated,
    }


def main() -> None:

    from retriever import HistoricalRetriever

    print("=" * 72)
    print("HIVER — RETRIEVAL EVALUATION")
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
        "human_intent",
    }

    missing = required - set(
        golden.columns
    )

    if missing:
        raise ValueError(
            f"Golden set missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"\nGolden examples: {len(golden)}"
    )

    historical_intents = (
        load_interaction_intents()
    )

    print(
        "Historical examples with "
        f"unambiguous weak intent: "
        f"{len(historical_intents):,}"
    )

    retriever = HistoricalRetriever()

    print("\nEvaluating lexical retrieval...")

    lexical = evaluate(
        retriever=retriever,
        golden=golden,
        historical_intents=historical_intents,
        use_intent_reranking=False,
    )

    print("\nLexical retrieval:")

    print(
        f"  Recall@1: "
        f"{lexical['recall_at_1']:.4f}"
    )

    print(
        f"  Recall@3: "
        f"{lexical['recall_at_3']:.4f}"
    )

    print(
        f"  Recall@5: "
        f"{lexical['recall_at_5']:.4f}"
    )

    print("\nEvaluating intent-aware reranking...")

    reranked = evaluate(
        retriever=retriever,
        golden=golden,
        historical_intents=historical_intents,
        use_intent_reranking=True,
    )

    print("\nIntent-aware reranking:")

    print(
        f"  Recall@1: "
        f"{reranked['recall_at_1']:.4f}"
    )

    print(
        f"  Recall@3: "
        f"{reranked['recall_at_3']:.4f}"
    )

    print(
        f"  Recall@5: "
        f"{reranked['recall_at_5']:.4f}"
    )

    output = {
        "lexical": lexical,
        "intent_aware": reranked,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved results to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()