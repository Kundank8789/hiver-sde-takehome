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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[dict[str, Any]]:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0 and 1."
            )

        normalized_query = self.normalize(
            query
        )

        if not normalized_query:
            return []

        query_vector = (
            self.vectorizer.transform(
                [normalized_query]
            )
        )

        # Proper cosine similarity.
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        # Only keep positive/relevant matches.
        candidate_indices = np.flatnonzero(
            scores >= min_score
        )

        if candidate_indices.size == 0:
            return []

        # Rank highest similarity first.
        order = np.argsort(
            scores[candidate_indices]
        )[::-1]

        results = []

        for position in order[:top_k]:

            index = int(
                candidate_indices[position]
            )

            score = float(
                scores[index]
            )

            # Defensive numerical guard.
            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            document = dict(
                self.documents[index]
            )

            document[
                "retrieval_score"
            ] = score

            results.append(
                document
            )

        return results


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
            f"score={result['retrieval_score']:.4f}"
        )

        print(
            result["customer_message"]
        )


if __name__ == "__main__":
    main()