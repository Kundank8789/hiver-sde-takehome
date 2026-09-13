from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_sample.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "judge_human_sample.csv"
)


def ask_rating(
    label: str,
    minimum: int = 1,
    maximum: int = 5,
) -> int:
    while True:
        value = input(
            f"{label} ({minimum}-{maximum}): "
        ).strip()

        try:
            number = int(value)
        except ValueError:
            print("Please enter a number.")
            continue

        if minimum <= number <= maximum:
            return number

        print(
            f"Please enter a value from "
            f"{minimum} to {maximum}."
        )


def main() -> None:

    print("=" * 80)
    print("HIVER — HUMAN REPLY QUALITY REVIEW")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    if len(df) != 30:
        raise ValueError(
            f"Expected 30 rows, found {len(df)}"
        )

    required = {
        "interaction_id",
        "customer_message",
        "reply",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print(
        "\nYou will review 30 replies."
    )

    print(
        "\nFor every case enter four numbers:"
    )

    print(
        "Overall   : 1 poor -> 5 excellent"
    )

    print(
        "Grounded  : 1 unsupported -> 5 fully grounded"
    )

    print(
        "Relevant  : 1 irrelevant -> 5 highly relevant"
    )

    print(
        "Helpful   : 1 not helpful -> 5 very helpful"
    )

    print(
        "\nIMPORTANT: judge the actual customer message "
        "and actual reply shown for each case."
    )

    input(
        "\nPress ENTER to begin..."
    )

    for index in range(
        len(df)
    ):

        row = df.iloc[index]

        print("\n" + "=" * 80)
        print(
            f"CASE {index + 1}/"
            f"{len(df)}"
        )
        print("=" * 80)

        print(
            "\nInteraction ID:"
        )
        print(
            row["interaction_id"]
        )

        print(
            "\nCUSTOMER:"
        )
        print(
            str(
                row["customer_message"]
            )
        )

        print(
            "\nAGENT REPLY:"
        )
        print(
            str(
                row["reply"]
            )
        )

        print(
            "\nRATINGS"
        )

        overall = ask_rating(
            "Overall"
        )

        grounded = ask_rating(
            "Grounded"
        )

        relevant = ask_rating(
            "Relevant"
        )

        helpful = ask_rating(
            "Helpful"
        )

        note = input(
            "Optional note (press ENTER to skip): "
        ).strip()

        df.loc[
            df.index[index],
            "human_overall_score"
        ] = overall

        df.loc[
            df.index[index],
            "human_grounded"
        ] = grounded

        df.loc[
            df.index[index],
            "human_relevant"
        ] = relevant

        df.loc[
            df.index[index],
            "human_helpful"
        ] = helpful

        df.loc[
            df.index[index],
            "human_notes"
        ] = note

        # Save after every case so progress isn't lost.
        df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8",
        )

        print(
            "\nSaved."
        )

    print("\n" + "=" * 80)
    print("REVIEW COMPLETE")
    print("=" * 80)

    print(
        f"\nRows reviewed: {len(df)}"
    )

    print(
        f"Saved to:\n{OUTPUT_PATH}"
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

    missing = int(
        df[rating_columns]
        .isna()
        .sum()
        .sum()
    )

    invalid = int(
        (
            ~df[rating_columns]
            .apply(
                lambda column:
                    column.between(1, 5)
            )
        )
        .sum()
        .sum()
    )

    print(
        f"Missing ratings: {missing}"
    )

    print(
        f"Invalid ratings: {invalid}"
    )


if __name__ == "__main__":
    main()