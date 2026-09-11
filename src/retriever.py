from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "retrieval_index"
)


INTENT_TERMS = {
    "IOS_UPDATE": [
        "ios",
        "update",
        "upgrade",
        "install",
        "software",
    ],
    "BATTERY_POWER": [
        "battery",
        "charge",
        "charging",
        "charger",
        "drain",
        "power",
    ],
    "PERFORMANCE_STABILITY": [
        "slow",
        "lag",
        "freeze",
        "crash",
        "restart",
        "stuck",
        "unresponsive",
    ],
    "CONNECTIVITY": [
        "wifi",
        "bluetooth",
        "cellular",
        "network",
        "connect",
        "connection",
        "hotspot",
    ],
    "APPS_MEDIA": [
        "music",
        "itunes",
        "imovie",
        "facetime",
        "messages",
        "photos",
        "youtube",
        "app",
    ],
    "DEVICE_HARDWARE": [
        "screen",
        "display",
        "speaker",
        "camera",
        "keyboard",
        "broken",
        "cracked",
        "hardware",
    ],
    "FEATURE_HOW_TO": [
        "how",
        "use",
        "enable",
        "disable",
        "where",
        "feature",
        "setting",
    ],
    "ACCOUNT_SECURITY": [
        "apple id",
        "password",
        "account",
        "verification",
        "verify",
        "locked",
        "sign in",
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
    "OTHER_UNCLEAR": [],
}


class HistoricalRetriever:
    """
    Leakage-safe historical support retriever.

    Loads a persisted TF-IDF sparse matrix and returns
    historically similar AppleSupport interactions.

    Retrieval score is cosine similarity in [0, 1].
    """

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> None:

        self.index_dir = Path(index_dir)

        matrix_path = (
            self.index_dir
            / "tfidf_matrix.npz"
        )

        vectorizer_path = (
            self.index_dir
            / "vectorizer.pkl"
        )

        documents_path = (
            self.index_dir
            / "documents.jsonl"
        )

        metadata_path = (
            self.index_dir
            / "metadata.json"
        )

        self._validate_files(
            matrix_path,
            vectorizer_path,
            documents_path,
            metadata_path,
        )

        # Load sparse TF-IDF matrix.
        self.matrix = load_npz(
            matrix_path
        ).tocsr()

        # Load fitted vectorizer.
        with vectorizer_path.open(
            "rb"
        ) as file:

            self.vectorizer = pickle.load(
                file
            )

        # Load document metadata.
        self.documents: list[dict[str, Any]] = []

        with documents_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                self.documents.append(
                    json.loads(line)
                )

        # Load index metadata.
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(
                file
            )

        self._validate_index()

    @staticmethod
    def _validate_files(
        matrix_path: Path,
        vectorizer_path: Path,
        documents_path: Path,
        metadata_path: Path,
    ) -> None:

        required_files = (
            matrix_path,
            vectorizer_path,
            documents_path,
            metadata_path,
        )

        missing = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Retrieval index is incomplete.\n"
                + "\n".join(missing)
            )

    def _validate_index(self) -> None:

        expected_documents = int(
            self.metadata[
                "document_count"
            ]
        )

        actual_documents = len(
            self.documents
        )

        if actual_documents != expected_documents:
            raise RuntimeError(
                "Document count mismatch: "
                f"metadata={expected_documents}, "
                f"actual={actual_documents}"
            )

        if self.matrix.shape[0] != actual_documents:
            raise RuntimeError(
                "TF-IDF matrix/document mismatch: "
                f"matrix={self.matrix.shape[0]}, "
                f"documents={actual_documents}"
            )

        if not self.metadata.get(
            "golden_set_excluded",
            False,
        ):
            raise RuntimeError(
                "Refusing to use an index that does not "
                "explicitly exclude the frozen golden set."
            )

    @staticmethod
    def normalize(text: str) -> str:

        text = str(text).lower()

        # Remove URLs.
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
        )

        # Remove @mentions.
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

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        min_score: float = 0.05,
        intent: str | None = None,
    ) -> list[dict[str, Any]]:

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be >= top_k."
            )

        normalized_query = self.normalize(query)

        if not normalized_query:
            return []

        query_vector = self.vectorizer.transform(
            [normalized_query]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        candidate_indices = np.flatnonzero(
            scores >= min_score
        )

        if candidate_indices.size == 0:
            return []

        lexical_order = np.argsort(
            scores[candidate_indices]
        )[::-1]

        candidate_indices = candidate_indices[
            lexical_order[:candidate_k]
        ]

        query_terms = set(
            self.tokenize(normalized_query)
        )

        intent_terms = set(
            INTENT_TERMS.get(
                intent,
                [],
            )
        )

        reranked = []

        for index in candidate_indices:

            lexical_score = float(
                scores[index]
            )

            document = self.documents[
                int(index)
            ]

            document_text = self.normalize(
                document["customer_message"]
            )

            document_terms = set(
                self.tokenize(document_text)
            )

            # Intent overlap.
            intent_overlap = 0.0

            if intent_terms:
                matched_intent_terms = (
                    query_terms
                    & document_terms
                    & intent_terms
                )

                intent_overlap = (
                    len(matched_intent_terms)
                    / max(
                        len(intent_terms),
                        1,
                    )
                )

            # Query overlap.
            query_overlap = 0.0

            if query_terms:
                query_overlap = (
                    len(
                        query_terms
                        & document_terms
                    )
                    / len(query_terms)
                )

            # Combined score.
            final_score = (
                0.70 * lexical_score
                + 0.20 * query_overlap
                + 0.10 * intent_overlap
            )

            result = dict(document)

            result[
                "retrieval_score"
            ] = lexical_score

            result[
                "rerank_score"
            ] = final_score

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]


def main() -> None:
    """
    Tiny smoke test.

    This allows:
        python src/retriever.py
    """

    retriever = HistoricalRetriever()

    results = retriever.retrieve(
        "my iphone battery is draining very quickly",
        top_k=3,
        intent="BATTERY_POWER",
    )

    print(
        f"Loaded documents: "
        f"{len(retriever.documents):,}"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"{rank}. "
            f"score={result['retrieval_score']:.4f} "
            f"rerank={result['rerank_score']:.4f}"
        )

        print(
            result["customer_message"]
        )


if __name__ == "__main__":
    main()