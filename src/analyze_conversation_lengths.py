import json
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("data/apple_conversations.jsonl")

print("Loading conversations...")

conversations = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        conversations.append(json.loads(line))

lengths = [len(c["messages"]) for c in conversations]

print(f"\nTotal conversations: {len(conversations):,}")

print("\nConversation length distribution:")
print(f"1 message : {sum(x == 1 for x in lengths):,}")
print(f"2 messages: {sum(x == 2 for x in lengths):,}")
print(f"3 messages: {sum(x == 3 for x in lengths):,}")
print(f"4 messages: {sum(x == 4 for x in lengths):,}")
print(f"5 messages: {sum(x == 5 for x in lengths):,}")
print(f"6-10     : {sum(6 <= x <= 10 for x in lengths):,}")
print(f"11-20    : {sum(11 <= x <= 20 for x in lengths):,}")
print(f"21-50    : {sum(21 <= x <= 50 for x in lengths):,}")
print(f"51+      : {sum(x > 50 for x in lengths):,}")

print("\nLongest conversations:")
top = sorted(
    conversations,
    key=lambda c: len(c["messages"]),
    reverse=True
)[:10]

for i, conversation in enumerate(top, 1):
    print(
        f"{i}. ID={conversation['conversation_id']} "
        f"messages={len(conversation['messages'])}"
    )