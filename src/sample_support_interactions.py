import json
from pathlib import Path

INPUT_PATH = Path("data/apple_support_interactions.jsonl")

NUM_SAMPLES = 60


print("Loading support interactions...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    interactions = [
        json.loads(line)
        for line in f
    ]

print(
    f"Total interactions: {len(interactions):,}"
)

print("\n" + "=" * 100)
print("SAMPLE APPLESUPPORT SUPPORT INTERACTIONS")
print("=" * 100)


# Deterministic sampling spread across the dataset.
step = max(len(interactions) // NUM_SAMPLES, 1)

samples = interactions[::step][:NUM_SAMPLES]


for i, item in enumerate(samples, 1):

    print(f"\n{'-' * 100}")
    print(f"INTERACTION {i}")
    print(f"ID: {item['interaction_id']}")

    print("\nCUSTOMER:")
    print(item["customer_message"])

    if item["context"]:
        print("\nPREVIOUS CONTEXT:")

        for message in item["context"][:-1]:
            print(
                f"[{message['role'].upper()}] "
                f"{message['text']}"
            )

    print("\nAPPLE SUPPORT:")
    print(item["agent_response"])


print("\nDone.")