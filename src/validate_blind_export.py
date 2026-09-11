from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "blind_golden_zoho_export.csv"
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

ALLOWED_ESCALATION = {
    "TRUE",
    "FALSE",
    "true",
    "false",
}


def clean(value: str | None) -> str:
    return (value or "").strip()


def main() -> None:

    print("=" * 72)
    print("HIVER — BLIND GOLDEN SET VALIDATION")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{INPUT_PATH}\n\n"
            "Download the completed Zoho sheet first."
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        expected_columns = [
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

        actual_columns = [
            clean(column)
            for column in reader.fieldnames
        ]

        if actual_columns != expected_columns:
            raise ValueError(
                "\nCSV schema does not match expected schema.\n\n"
                f"Expected:\n{expected_columns}\n\n"
                f"Found:\n{actual_columns}"
            )

        rows = list(reader)

    print(f"\nRows found: {len(rows)}")

    errors: list[str] = []

    # ------------------------------------------------------
    # Row count
    # ------------------------------------------------------

    if len(rows) != EXPECTED_ROWS:
        errors.append(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(rows)}."
        )

    # ------------------------------------------------------
    # Required fields
    # ------------------------------------------------------

    required = [
        "interaction_id",
        "customer_message",
        "human_intent",
        "should_escalate",
        "escalation_reason",
    ]

    for row_number, row in enumerate(rows, start=2):

        for column in required:

            if not clean(row.get(column)):
                errors.append(
                    f"Row {row_number}: "
                    f"{column} is empty."
                )

    # ------------------------------------------------------
    # Intent values
    # ------------------------------------------------------

    for row_number, row in enumerate(rows, start=2):

        intent = clean(
            row.get("human_intent")
        )

        if intent not in ALLOWED_INTENTS:
            errors.append(
                f"Row {row_number}: invalid intent "
                f"'{intent}'."
            )

    # ------------------------------------------------------
    # Escalation values
    # ------------------------------------------------------

    for row_number, row in enumerate(rows, start=2):

        value = clean(
            row.get("should_escalate")
        )

        if value not in ALLOWED_ESCALATION:
            errors.append(
                f"Row {row_number}: invalid "
                f"should_escalate value '{value}'."
            )

    # ------------------------------------------------------
    # Duplicate interaction IDs
    # ------------------------------------------------------

    interaction_ids = [
        clean(row.get("interaction_id"))
        for row in rows
    ]

    seen: set[str] = set()

    for interaction_id in interaction_ids:

        if interaction_id in seen:
            errors.append(
                f"Duplicate interaction_id: "
                f"{interaction_id}"
            )

        seen.add(interaction_id)

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("=" * 72)
    print("VALIDATION RESULT")
    print("=" * 72)

    if errors:

        print(
            f"\n❌ VALIDATION FAILED "
            f"({len(errors)} issue(s))\n"
        )

        for error in errors[:50]:
            print(f"- {error}")

        if len(errors) > 50:
            print(
                f"\n... {len(errors) - 50} "
                "additional issues ..."
            )

        raise SystemExit(1)

    print("\n✅ Structural validation passed.")

    # ------------------------------------------------------
    # Distribution
    # ------------------------------------------------------

    intent_counts: dict[str, int] = {}

    for row in rows:

        intent = clean(
            row["human_intent"]
        )

        intent_counts[intent] = (
            intent_counts.get(intent, 0) + 1
        )

    print("\nIntent distribution:")

    for intent, count in sorted(
        intent_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(
            f"  {intent:<28} {count:>3}"
        )

    escalation_counts = {
        "TRUE": 0,
        "FALSE": 0,
    }

    for row in rows:

        value = clean(
            row["should_escalate"]
        ).upper()

        escalation_counts[value] += 1

    print("\nEscalation distribution:")

    for value, count in escalation_counts.items():
        print(
            f"  {value:<28} {count:>3}"
        )

    print("\n✅ Golden set is structurally valid.")
    print(
        "Next step: create the frozen "
        "evaluation/golden_set.csv."
    )


if __name__ == "__main__":
    main()