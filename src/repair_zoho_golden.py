from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/golden_zoho_export.csv")
OUTPUT_PATH = Path("evaluation/golden_set.csv")

COLUMNS = [
    "interaction_id",
    "customer_tweet_id",
    "agent_tweet_id",
    "customer_message",
    "context",
    "historical_agent_response",
    "human_intent",
    "should_escalate",
    "escalation_reason",
    "provisional_intent",
]


print("Loading Zoho export...")

raw = pd.read_csv(
    INPUT_PATH,
    header=None,
    dtype=str,
)

print(f"Raw rows: {len(raw)}")
print(f"Raw columns: {len(raw.columns)}")

if raw.shape != (201, 10):
    raise ValueError(
        f"Expected 201 rows x 10 columns, got {raw.shape}"
    )


# ---------------------------------------------------------
# The Zoho export has:
#
# raw row 0:
#   correct columns A-F
#   FIRST EXAMPLE'S labels in G-I
#   "provisional_intent" in J
#
# raw rows 1..200:
#   the 200 actual data examples
#   but their G-I labels are shifted UP by one row
#
# Therefore:
#   A-F + J come from rows 1..200
#   G-I come from rows 0..199
# ---------------------------------------------------------

data = raw.iloc[1:].copy()
data.columns = COLUMNS


# Preserve the actual data columns.
final_df = data.copy()


# Recover labels by shifting G-I down by one row.
human_intent = raw.iloc[0:200, 6].tolist()
should_escalate = raw.iloc[0:200, 7].tolist()
escalation_reason = raw.iloc[0:200, 8].tolist()

final_df["human_intent"] = human_intent
final_df["should_escalate"] = should_escalate
final_df["escalation_reason"] = escalation_reason


# Clean whitespace.
for column in COLUMNS:
    final_df[column] = final_df[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )


# ---------------------------------------------------------
# Validate annotations
# ---------------------------------------------------------

print()
print("=" * 70)
print("REPAIRED GOLDEN SET")
print("=" * 70)

print(f"Examples: {len(final_df)}")

for column in [
    "human_intent",
    "should_escalate",
    "escalation_reason",
]:
    missing = (
        final_df[column].isna()
        | (final_df[column].astype(str).str.strip() == "")
    )

    print(
        f"{column} missing: {int(missing.sum())}"
    )


# Allowed values.
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


actual_intents = set(
    final_df["human_intent"]
    .dropna()
    .astype(str)
    .str.strip()
)

invalid_intents = actual_intents - ALLOWED_INTENTS

if invalid_intents:
    raise ValueError(
        f"Invalid intents found: {sorted(invalid_intents)}"
    )


actual_escalation = set(
    final_df["should_escalate"]
    .dropna()
    .astype(str)
    .str.strip()
)

invalid_escalation = actual_escalation - ALLOWED_ESCALATION

if invalid_escalation:
    raise ValueError(
        "Invalid escalation values found: "
        f"{sorted(invalid_escalation)}"
    )


# Check all labels exist.
for column in [
    "human_intent",
    "should_escalate",
    "escalation_reason",
]:
    missing = (
        final_df[column].isna()
        | (final_df[column].astype(str).str.strip() == "")
    )

    if missing.any():
        raise ValueError(
            f"{column} still contains missing values."
        )


# ---------------------------------------------------------
# Remove provisional_intent from final benchmark.
# ---------------------------------------------------------

FINAL_COLUMNS = [
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

golden_set = final_df[FINAL_COLUMNS].copy()

golden_set.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print()
print("✅ Repair successful.")
print()
print("Intent distribution:")
print(
    golden_set["human_intent"]
    .value_counts()
    .to_string()
)

print()
print("Escalation distribution:")
print(
    golden_set["should_escalate"]
    .astype(str)
    .str.upper()
    .value_counts()
    .to_string()
)

print()
print(f"Saved to: {OUTPUT_PATH}")