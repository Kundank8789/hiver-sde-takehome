from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/golden_annotation_sheet.csv")

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


print("Loading golden set...")

df = pd.read_csv(INPUT_PATH)

print(f"Rows found: {len(df)}")

errors = []

required_columns = {
    "interaction_id",
    "customer_message",
    "human_intent",
    "should_escalate",
    "escalation_reason",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    errors.append(
        f"Missing columns: {sorted(missing_columns)}"
    )

if errors:
    print("\n❌ VALIDATION FAILED")

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)


# Check number of rows
if len(df) != 200:
    errors.append(
        f"Expected 200 rows, found {len(df)}."
    )


# Check missing values
for column in [
    "human_intent",
    "should_escalate",
    "escalation_reason",
]:
    missing = (
        df[column].isna()
        | (df[column].astype(str).str.strip() == "")
    )

    count = int(missing.sum())

    if count:
        errors.append(
            f"{column}: {count} missing values"
        )


# Check intent values
actual_intents = set(
    df["human_intent"]
    .dropna()
    .astype(str)
    .str.strip()
)

invalid_intents = actual_intents - ALLOWED_INTENTS

if invalid_intents:
    errors.append(
        f"Invalid intents: {sorted(invalid_intents)}"
    )


# Check escalation values
actual_escalation = set(
    df["should_escalate"]
    .dropna()
    .astype(str)
    .str.strip()
)

invalid_escalation = (
    actual_escalation - ALLOWED_ESCALATION
)

if invalid_escalation:
    errors.append(
        "Invalid should_escalate values: "
        f"{sorted(invalid_escalation)}"
    )


print()
print("=" * 70)
print("GOLDEN SET VALIDATION")
print("=" * 70)

if errors:

    print("\n❌ VALIDATION FAILED\n")

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)


print("\n✅ All basic validation checks passed.")

print("\nIntent distribution:")
print(
    df["human_intent"]
    .value_counts()
    .to_string()
)

print("\nEscalation distribution:")
print(
    df["should_escalate"]
    .astype(str)
    .str.upper()
    .value_counts()
    .to_string()
)

print("\nGolden set is ready for evaluation.")