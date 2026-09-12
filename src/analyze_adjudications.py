from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set_adjudication.csv"
)


def main() -> None:

    print("=" * 72)
    print("HIVER — ADJUDICATION ANALYSIS")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    reviewed = df[
        df["adjudication"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()

    print(
        f"\nTotal golden examples: {len(df)}"
    )

    print(
        f"Reviewed examples: {len(reviewed)}"
    )

    if reviewed.empty:
        print("No adjudications found.")
        return

    print("\nAdjudication distribution:")

    print(
        reviewed[
            "adjudication"
        ]
        .value_counts()
        .to_string()
    )

    print("\nCorrected intent distribution:")

    corrected = reviewed[
        reviewed["adjudication"]
        == "CHANGE_GOLD"
    ]

    if corrected.empty:
        print("No corrected labels.")
    else:
        print(
            corrected[
                "adjudicated_intent"
            ]
            .value_counts()
            .to_string()
        )

    print("\nOriginal → adjudicated changes:")

    if corrected.empty:
        print("No changes.")
    else:

        changes = (
            corrected.groupby(
                [
                    "gold_intent",
                    "adjudicated_intent",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
        )

        print(
            changes.to_string(
                index=False
            )
        )

    print("\nExamples marked AMBIGUOUS:")

    ambiguous = reviewed[
        reviewed["adjudication"]
        == "AMBIGUOUS"
    ]

    if ambiguous.empty:
        print("None.")
    else:

        columns = [
            "interaction_id",
            "gold_intent",
            "predicted_intent",
            "customer_message",
        ]

        print(
            ambiguous[columns]
            .to_string(index=False)
        )

    print("\nAdjudication coverage:")

    print(
        f"{len(reviewed) / len(df):.2%}"
    )


if __name__ == "__main__":
    main()