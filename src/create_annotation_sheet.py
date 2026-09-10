import json
import random
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/golden_candidates.jsonl")
OUTPUT_PATH = Path("evaluation/golden_annotation_sheet.csv")

TARGET_SIZE = 200
SEED = 42


INTENTS = [
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
]


print("Loading golden candidates...")

items = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))

print(f"Loaded {len(items)} candidates.")


# ---------------------------------------------------------
# Prefer broad coverage across provisional categories.
# These provisional labels are ONLY used for sampling.
# ---------------------------------------------------------

random.seed(SEED)

groups = {}

for item in items:
    intent = item.get(
        "provisional_intent",
        "OTHER_UNCLEAR",
    )

    groups.setdefault(intent, []).append(item)


selected = []

# Approximately equal representation from the provisional groups.
base_per_group = TARGET_SIZE // len(INTENTS)
remainder = TARGET_SIZE % len(INTENTS)

for index, intent in enumerate(INTENTS):

    group = groups.get(intent, [])

    if not group:
        continue

    target = base_per_group

    if index < remainder:
        target += 1

    target = min(target, len(group))

    selected.extend(
        random.sample(group, target)
    )


# Fill any remaining slots.
selected_ids = {
    item["interaction_id"]
    for item in selected
}

remaining = [
    item
    for item in items
    if item["interaction_id"] not in selected_ids
]

random.shuffle(remaining)

while len(selected) < TARGET_SIZE and remaining:
    selected.append(remaining.pop())


random.shuffle(selected)


# ---------------------------------------------------------
# Build annotation rows
# ---------------------------------------------------------

rows = []

for item in selected:

    context_text = "\n".join(
        f"[{message['role'].upper()}] {message['text']}"
        for message in item.get("context", [])
        if message["tweet_id"] != item["customer_tweet_id"]
    )

    rows.append(
        {
            "interaction_id": item["interaction_id"],
            "customer_tweet_id": item["customer_tweet_id"],
            "agent_tweet_id": item["agent_tweet_id"],
            "customer_message": item["customer_message"],
            "context": context_text,
            "historical_agent_response": item["agent_response"],

            # Human annotation fields:
            "human_intent": "",
            "should_escalate": "",
            "escalation_reason": "",

            # Keep this only as metadata.
            # DO NOT use it as ground truth.
            "provisional_intent": item.get(
                "provisional_intent",
                "OTHER_UNCLEAR",
            ),
        }
    )


df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 70)
print("GOLDEN ANNOTATION SHEET")
print("=" * 70)

print(f"Rows created: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")