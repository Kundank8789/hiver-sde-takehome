from __future__ import annotations

from retriever import HistoricalRetriever


TEST_CASES = [
    (
        "My iPhone battery is draining very quickly",
        "BATTERY_POWER",
    ),
    (
        "My iPhone cannot install the latest iOS update",
        "IOS_UPDATE",
    ),
    (
        "Bluetooth won't connect to my car",
        "CONNECTIVITY",
    ),
    (
        "My Apple ID password is locked",
        "ACCOUNT_SECURITY",
    ),
    (
        "How do I use the notification feature on my iPhone?",
        "FEATURE_HOW_TO",
    ),
]


def main() -> None:

    print("=" * 80)
    print("HIVER — INTENT-AWARE RETRIEVAL SANITY CHECK")
    print("=" * 80)

    retriever = HistoricalRetriever()

    print(
        f"\nLoaded documents: "
        f"{len(retriever.documents):,}"
    )

    for query, intent in TEST_CASES:

        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"INTENT: {intent}")
        print("=" * 80)

        results = retriever.retrieve(
            query=query,
            intent=intent,
            candidate_k=20,
            top_k=3,
        )

        if not results:
            print("\nNo results.")
            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"RESULT {rank} "
                f"(lexical={result['retrieval_score']:.4f}, "
                f"rerank={result['rerank_score']:.4f})"
            )

            print(
                "\nCustomer:"
            )

            print(
                result["customer_message"]
            )

            print(
                "\nAppleSupport:"
            )

            print(
                result["agent_response"]
            )


if __name__ == "__main__":
    main()