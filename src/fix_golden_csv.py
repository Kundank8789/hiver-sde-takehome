from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/golden_annotation_sheet.csv")
OUTPUT_PATH = Path("evaluation/golden_annotation_sheet_fixed.csv")

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
    "provisional_intent",
]


print("Loading Zoho-exported CSV...")

# Read WITHOUT assuming the first row is the header.
df = pd.read_csv(
    INPUT_PATH,
    header=None,
)

print(f"Raw rows including header: {len(df)}")
print(f"Raw columns: {len(df.columns)}")

if len(df) != 201:
    raise ValueError(
        f"Expected 201 rows (1 header + 200 examples), "
        f"but found {len(df)}."
    )

if len(df.columns) != 10:
    raise ValueError(
        f"Expected 10 columns, found {len(df.columns)}."
    )

# The first row is the original header.
df = df.iloc[1:].copy()

# Restore the intended schema.
df.columns = EXPECTED_COLUMNS

# Clean whitespace from text columns.
for column in EXPECTED_COLUMNS:
    if df[column].dtype == "object":
        df[column] = df[column].apply(
            lambda x: x.strip()
            if isinstance(x, str)
            else x
        )

# Save corrected file.
df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print()
print("=" * 70)
print("FIXED GOLDEN CSV")
print("=" * 70)

print(f"Examples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Saved to: {OUTPUT_PATH}")