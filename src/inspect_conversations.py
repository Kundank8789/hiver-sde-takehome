import json
from pathlib import Path

INPUT_PATH = Path("data/apple_conversations.jsonl")

NUM_CONVERSATIONS = 10

print("Loading reconstructed conversations...\n")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    conversations = [
        json.loads(line)
        for line in f
    ]

print(f"Total conversations: {len(conversations):,}")

print("\n" + "=" * 100)
print("SAMPLE RECONSTRUCTED CONVERSATIONS")
print("=" * 100)

for i, conversation in enumerate(
    conversations[:NUM_CONVERSATIONS],
    start=1
):
    print(f"\n\nCONVERSATION {i}")
    print(f"Conversation ID: {conversation['conversation_id']}")
    print("-" * 100)

    for message in conversation["messages"]:

        role = message["role"].upper()

        print(
            f"\n[{role}] "
            f"(tweet_id={message['tweet_id']})"
        )

        print(message["text"])

print("\nDone.")