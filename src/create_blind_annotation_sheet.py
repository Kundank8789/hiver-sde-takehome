import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "evaluation/blind_golden_candidates.jsonl"
)

OUTPUT_PATH = Path(
    "evaluation/blind_golden_annotation.csv"
)


print("Loading blind candidates...")

items = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))


rows = []

for item in items:

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

            # Human annotation fields.
            "human_intent": "",
            "should_escalate": "",
            "escalation_reason": "",
        }
    )


df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print()
print("=" * 70)
print("BLIND GOLDEN ANNOTATION SHEET")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print()
print("Saved to:")
print(OUTPUT_PATH)