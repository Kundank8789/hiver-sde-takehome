import json
from pathlib import Path

INPUT_PATH = Path("data/apple_conversations.jsonl")

MAX_CONVERSATIONS = 5
MAX_MESSAGES_TO_PRINT = 30

print("Loading conversations...")

conversations = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        conversations.append(json.loads(line))

large = sorted(
    conversations,
    key=lambda c: len(c["messages"]),
    reverse=True
)[:MAX_CONVERSATIONS]

print(
    f"\nInspecting {len(large)} largest conversations..."
)

print("=" * 100)

for i, conversation in enumerate(large, 1):

    print(
        f"\nCONVERSATION {i}"
    )

    print(
        f"Root ID: {conversation['conversation_id']}"
    )

    print(
        f"Message count: {len(conversation['messages'])}"
    )

    print("-" * 100)

    messages = conversation["messages"]

    for j, message in enumerate(
        messages[:MAX_MESSAGES_TO_PRINT],
        1
    ):

        print(
            f"\n{j}. [{message['role'].upper()}] "
            f"tweet_id={message['tweet_id']}"
        )

        print(message["text"])

    if len(messages) > MAX_MESSAGES_TO_PRINT:
        print(
            f"\n... "
            f"{len(messages) - MAX_MESSAGES_TO_PRINT} "
            f"more messages not shown ..."
        )

print("\nDone.")