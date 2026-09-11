from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "blind_golden_zoho_export.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set.csv"
)

BACKUP_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_set_backup.csv"
)

EXPECTED_ROWS = 200

ALLOWED_INTENTS = {
    "IOS_UPDATE",
    "BATTERY_POWER",
    "PERFORMANCE_STABILITY",
    "CONNECTIVITY",
    "APPS_MEDIA",
    "DEVICE_HARDWARE",
    "FEATURE_HOW_TO",
    "ACCOUNT_SECURITY",
    "PURCHASE_REPAIR_WARRANTY",
    "OTHER_UNCLEAR",
}

EXPECTED_COLUMNS = [
    "interaction_id",
    "customer_tweet_id",
    "agent_tweet_id",
    "customer_message",
    "context",
    "historical_agent_response",
    "human_intent",
    "should_escalate",
    "escalation_reason",
]


def clean(value: str | None) -> str:
    return (value or "").strip()


def main() -> None:
    print("=" * 72)
    print("HIVER — FREEZE GOLDEN EVALUATION SET")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        columns = [
            clean(column)
            for column in reader.fieldnames
        ]

        if columns != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected CSV schema.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {columns}"
            )

        rows = list(reader)

    print(f"\nRows loaded: {len(rows)}")

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(rows)}."
        )

    # ------------------------------------------------------
    # Validate every example before freezing.
    # ------------------------------------------------------

    interaction_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):

        interaction_id = clean(
            row["interaction_id"]
        )

        if not interaction_id:
            raise ValueError(
                f"Row {row_number}: empty interaction_id."
            )

        if interaction_id in interaction_ids:
            raise ValueError(
                f"Duplicate interaction_id: "
                f"{interaction_id}"
            )

        interaction_ids.add(interaction_id)

        customer_message = clean(
            row["customer_message"]
        )

        if not customer_message:
            raise ValueError(
                f"Row {row_number}: empty customer_message."
            )

        intent = clean(
            row["human_intent"]
        )

        if intent not in ALLOWED_INTENTS:
            raise ValueError(
                f"Row {row_number}: invalid intent "
                f"'{intent}'."
            )

        escalation = clean(
            row["should_escalate"]
        ).upper()

        if escalation not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Row {row_number}: invalid escalation "
                f"value '{escalation}'."
            )

        reason = clean(
            row["escalation_reason"]
        )

        if not reason:
            raise ValueError(
                f"Row {row_number}: empty escalation_reason."
            )

    # ------------------------------------------------------
    # Preserve a backup if an older golden_set exists.
    # ------------------------------------------------------

    if OUTPUT_PATH.exists():
        shutil.copy2(
            OUTPUT_PATH,
            BACKUP_PATH,
        )
        print(
            f"Existing golden set backed up to:\n"
            f"{BACKUP_PATH}"
        )

    # ------------------------------------------------------
    # Write frozen evaluation set.
    # ------------------------------------------------------

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EXPECTED_COLUMNS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: clean(row.get(column))
                    for column in EXPECTED_COLUMNS
                }
            )

    # ------------------------------------------------------
    # Verify the written file.
    # ------------------------------------------------------

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)
        frozen_rows = list(reader)

    if len(frozen_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            "Frozen file verification failed."
        )

    print()
    print("=" * 72)
    print("FROZEN GOLDEN SET")
    print("=" * 72)

    print(
        f"Examples: {len(frozen_rows)}"
    )

    print(
        f"Unique IDs: "
        f"{len({row['interaction_id'] for row in frozen_rows})}"
    )

    print("\nIntent distribution:")

    intent_counts: dict[str, int] = {}

    for row in frozen_rows:
        intent = row["human_intent"]
        intent_counts[intent] = (
            intent_counts.get(intent, 0) + 1
        )

    for intent, count in sorted(intent_counts.items()):
        print(
            f"  {intent:<28} {count:>3}"
        )

    print("\nEscalation distribution:")

    escalation_counts = {
        "TRUE": 0,
        "FALSE": 0,
    }

    for row in frozen_rows:
        value = row["should_escalate"].upper()
        escalation_counts[value] += 1

    for value, count in escalation_counts.items():
        print(
            f"  {value:<28} {count:>3}"
        )

    print()
    print("✅ Golden set successfully frozen.")
    print(f"Saved to:\n{OUTPUT_PATH}")
    print(
        "\nFrom this point forward, do not manually "
        "edit golden_set.csv."
    )


if __name__ == "__main__":
    main()