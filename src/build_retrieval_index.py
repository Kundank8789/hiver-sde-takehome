from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "apple_support_interactions.jsonl"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "retrieval_index"
)

MATRIX_PATH = INDEX_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"
DOCUMENTS_PATH = INDEX_DIR / "documents.jsonl"
METADATA_PATH = INDEX_DIR / "metadata.json"

TOKEN_PATTERN = re.compile(
    r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_'-]*\b"
)


def normalize_text(text: str) -> str:
    """
    Normalize noisy Twitter text while preserving
    useful lexical information.
    """
    text = str(text).lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove Twitter mentions.
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def main() -> None:

    print("=" * 72)
    print("HIVER — BUILD LEAKAGE-SAFE TF-IDF RETRIEVAL INDEX")
    print("=" * 72)

    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Interactions not found:\n"
            f"{INTERACTIONS_PATH}"
        )

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found:\n"
            f"{GOLDEN_PATH}"
        )

    # ------------------------------------------------------
    # Load frozen evaluation IDs.
    # ------------------------------------------------------

    golden = pd.read_csv(
        GOLDEN_PATH,
        usecols=["interaction_id"],
    )

    golden_ids = {
        str(value).strip()
        for value in golden["interaction_id"]
        if pd.notna(value)
    }

    print(
        f"\nGolden IDs excluded: "
        f"{len(golden_ids)}"
    )

    # ------------------------------------------------------
    # Stream interactions and remove leakage.
    # ------------------------------------------------------

    documents = []
    seen_customer_messages = set()

    total = 0
    excluded_golden = 0
    excluded_duplicate = 0
    excluded_empty = 0

    with INTERACTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as source:

        for line in source:

            if not line.strip():
                continue

            total += 1

            item = json.loads(line)

            interaction_id = str(
                item["interaction_id"]
            ).strip()

            if interaction_id in golden_ids:
                excluded_golden += 1
                continue

            customer_message = normalize_text(
                item["customer_message"]
            )

            if len(customer_message) < 5:
                excluded_empty += 1
                continue

            # Deduplicate exact customer messages.
            message_key = customer_message

            if message_key in seen_customer_messages:
                excluded_duplicate += 1
                continue

            seen_customer_messages.add(
                message_key
            )

            documents.append(
                {
                    "interaction_id": interaction_id,
                    "customer_tweet_id": str(
                        item["customer_tweet_id"]
                    ),
                    "customer_message": item[
                        "customer_message"
                    ],
                    "normalized_text": customer_message,
                    "context": item.get(
                        "context",
                        [],
                    ),
                    "agent_response": item[
                        "agent_response"
                    ],
                }
            )

    if not documents:
        raise RuntimeError(
            "No retrieval documents remain."
        )

    print(
        f"\nRaw interactions: "
        f"{total:,}"
    )

    print(
        f"Excluded golden examples: "
        f"{excluded_golden:,}"
    )

    print(
        f"Excluded duplicates: "
        f"{excluded_duplicate:,}"
    )

    print(
        f"Excluded empty/noisy messages: "
        f"{excluded_empty:,}"
    )

    print(
        f"Final retrieval documents: "
        f"{len(documents):,}"
    )

    # ------------------------------------------------------
    # Build sparse TF-IDF matrix.
    # ------------------------------------------------------

    texts = [
        document["normalized_text"]
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN.pattern,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=150_000,
        norm="l2",
    )

    print(
        "\nFitting TF-IDF vectorizer..."
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    print(
        f"Matrix shape: {matrix.shape}"
    )

    # ------------------------------------------------------
    # Persist index.
    # ------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving retrieval index..."
    )

    save_npz(
        MATRIX_PATH,
        matrix,
    )

    with VECTORIZER_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            vectorizer,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with DOCUMENTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for document in documents:

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata = {
        "version": 2,
        "brand": "AppleSupport",
        "document_count": len(documents),
        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
        "golden_set_excluded": True,
        "golden_set_size": len(golden_ids),
        "duplicate_customer_messages_removed": True,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("RETRIEVAL INDEX COMPLETE")
    print("=" * 72)

    print(
        f"Documents: "
        f"{len(documents):,}"
    )

    print(
        f"Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Golden leakage excluded: "
        f"{metadata['golden_set_excluded']}"
    )

    print(
        f"\nIndex directory:\n"
        f"{INDEX_DIR}"
    )


if __name__ == "__main__":
    main()