from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "apple_support_interactions.jsonl"
)

WEAK_PATH = (
    PROJECT_ROOT
    / "data"
    / "weak_training.jsonl"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "retrieval_index"
)

MATRIX_PATH = OUTPUT_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = OUTPUT_DIR / "vectorizer.pkl"
DOCUMENTS_PATH = OUTPUT_DIR / "documents.jsonl"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def normalize(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def load_golden_ids() -> set[str]:
    import pandas as pd

    golden = pd.read_csv(
        GOLDEN_PATH,
        usecols=["interaction_id"],
    )

    return {
        str(value).strip()
        for value in golden["interaction_id"]
        if str(value).strip()
        and str(value).strip() != "nan"
    }


def load_weak_intents() -> dict[str, str]:
    """
    Build:
        interaction_id -> weakly assigned intent

    Only examples present in weak_training.jsonl are included.
    """

    mapping: dict[str, str] = {}

    with WEAK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            interaction_id = str(
                item.get(
                    "interaction_id",
                    "",
                )
            ).strip()

            intent = str(
                item.get(
                    "intent",
                    "",
                )
            ).strip()

            if (
                interaction_id
                and intent
            ):
                mapping[
                    interaction_id
                ] = intent

    return mapping


def main() -> None:

    print("=" * 72)
    print("HIVER — BUILD LEAKAGE-SAFE TF-IDF RETRIEVAL INDEX")
    print("=" * 72)

    golden_ids = load_golden_ids()
    weak_intents = load_weak_intents()

    print(
        f"\nGolden IDs excluded: "
        f"{len(golden_ids)}"
    )

    print(
        f"Weak intent labels available: "
        f"{len(weak_intents):,}"
    )

    documents = []
    seen_texts = set()

    raw = 0
    excluded_golden = 0
    excluded_duplicates = 0
    excluded_empty = 0
    labeled_documents = 0

    print(
        "\nReading support interactions..."
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            raw += 1

            item = json.loads(line)

            interaction_id = str(
                item.get(
                    "interaction_id",
                    "",
                )
            ).strip()

            if interaction_id in golden_ids:
                excluded_golden += 1
                continue

            customer_message = str(
                item.get(
                    "customer_message",
                    "",
                )
            ).strip()

            agent_response = str(
                item.get(
                    "agent_response",
                    "",
                )
            ).strip()

            normalized = normalize(
                customer_message
            )

            if not normalized:
                excluded_empty += 1
                continue

            # Remove duplicate customer messages.
            if normalized in seen_texts:
                excluded_duplicates += 1
                continue

            seen_texts.add(normalized)

            weak_intent = weak_intents.get(
                interaction_id
            )

            if weak_intent:
                labeled_documents += 1

            documents.append(
                {
                    "interaction_id": interaction_id,
                    "customer_message": customer_message,
                    "agent_response": agent_response,
                    "intent": weak_intent,
                }
            )

    if not documents:
        raise RuntimeError(
            "No retrieval documents were created."
        )

    texts = [
        normalize(
            document[
                "customer_message"
            ]
        )
        for document in documents
    ]

    print(
        "\nWriting retrieval index..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 2),
        max_features=150_000,
    )

    matrix = vectorizer.fit_transform(
        texts
    ).tocsr()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        "version": 3,
        "brand": "AppleSupport",
        "raw_interactions": raw,
        "document_count": len(documents),
        "labeled_documents": labeled_documents,
        "excluded_golden": excluded_golden,
        "excluded_duplicates": excluded_duplicates,
        "excluded_empty": excluded_empty,
        "vocabulary_size": len(
            vectorizer.vocabulary_
        ),
        "golden_set_excluded": True,
        "weak_intents_attached": True,
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
        f"Raw interactions: {raw:,}"
    )

    print(
        f"Excluded golden examples: "
        f"{excluded_golden:,}"
    )

    print(
        f"Excluded duplicates: "
        f"{excluded_duplicates:,}"
    )

    print(
        f"Excluded empty/noisy messages: "
        f"{excluded_empty:,}"
    )

    print(
        f"Final retrieval documents: "
        f"{len(documents):,}"
    )

    print(
        f"Documents with weak intent: "
        f"{labeled_documents:,}"
    )

    print(
        f"Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"\nIndex directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()