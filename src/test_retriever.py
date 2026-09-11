from __future__ import annotations

from retriever import HistoricalRetriever


def print_results(
    query: str,
    retriever: HistoricalRetriever,
    top_k: int = 3,
) -> None:

    print()
    print("=" * 80)
    print(f"QUERY:\n{query}")
    print("=" * 80)

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
    )

    if not results:
        print("\nNo relevant historical cases found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        score = result["retrieval_score"]

        print()
        print(
            f"RESULT {rank} "
            f"(cosine={score:.4f})"
        )

        print(
            "\nCustomer:"
        )

        print(
            result["customer_message"]
        )

        print(
            "\nHistorical AppleSupport:"
        )

        print(
            result["agent_response"]
        )


def main() -> None:

    print("=" * 80)
    print("HIVER — RETRIEVER SANITY CHECK")
    print("=" * 80)

    retriever = HistoricalRetriever()

    print(
        f"\nLoaded documents: "
        f"{len(retriever.documents):,}"
    )

    queries = [
        "My iPhone battery is draining very quickly",

        "My iPhone cannot install the latest iOS update",

        "Bluetooth won't connect to my car",

        "My Apple ID password is locked",

        "How do I use the notification feature on my iPhone?",
    ]

    for query in queries:

        print_results(
            query=query,
            retriever=retriever,
            top_k=3,
        )


if __name__ == "__main__":
    main()