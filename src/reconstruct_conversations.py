import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/twcs.csv")
OUTPUT_PATH = Path("data/apple_conversations.jsonl")

BRAND = "AppleSupport"

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
)

print(f"Loaded {len(df):,} tweets.")

# -------------------------------------------------------------------
# 1. Create tweet lookup tables
# -------------------------------------------------------------------

tweets = df.set_index("tweet_id").to_dict("index")

# response_tweet_id can contain comma-separated tweet IDs.
# Build forward links: parent tweet -> replies.
children = {}

for _, row in df.iterrows():
    tweet_id = int(row["tweet_id"])
    parent_id = row["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    parent_id = int(parent_id)

    children.setdefault(parent_id, []).append(tweet_id)


# -------------------------------------------------------------------
# 2. Find AppleSupport tweets
# -------------------------------------------------------------------

brand_ids = set(
    df.loc[df["author_id"] == BRAND, "tweet_id"].astype(int)
)

print(f"AppleSupport tweets: {len(brand_ids):,}")


# -------------------------------------------------------------------
# 3. Find customer tweets directly answered by AppleSupport
# -------------------------------------------------------------------

customer_entry_ids = set()

for tweet_id in brand_ids:

    tweet = tweets[tweet_id]

    parent_id = tweet["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    customer_entry_ids.add(int(parent_id))

print(
    "Customer tweets with direct AppleSupport replies: "
    f"{len(customer_entry_ids):,}"
)


# -------------------------------------------------------------------
# 4. For every customer entry, walk backwards to the root
# -------------------------------------------------------------------

def find_root(tweet_id: int) -> int:
    """
    Walk backwards through in_response_to_tweet_id
    until reaching the beginning of the thread.
    """

    current_id = tweet_id
    visited = set()

    while current_id in tweets:

        if current_id in visited:
            break

        visited.add(current_id)

        parent_id = tweets[current_id]["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            break

        parent_id = int(parent_id)

        if parent_id not in tweets:
            break

        current_id = parent_id

    return current_id


root_to_entries = {}

for customer_id in customer_entry_ids:

    root_id = find_root(customer_id)

    root_to_entries.setdefault(root_id, set()).add(customer_id)


print(
    f"Unique root threads containing AppleSupport replies: "
    f"{len(root_to_entries):,}"
)


# -------------------------------------------------------------------
# 5. Reconstruct each unique thread
# -------------------------------------------------------------------

conversations = []

for root_id in sorted(root_to_entries):

    # Traverse forward through the thread.
    # We recursively collect all descendants from the root.
    collected_ids = []
    queue = [root_id]
    visited = set()

    while queue:

        current_id = queue.pop(0)

        if current_id in visited:
            continue

        if current_id not in tweets:
            continue

        visited.add(current_id)
        collected_ids.append(current_id)

        for child_id in children.get(current_id, []):
            queue.append(child_id)

    # Keep only threads that contain AppleSupport.
    if not any(
        tweets[tweet_id]["author_id"] == BRAND
        for tweet_id in collected_ids
    ):
        continue

    # Sort by tweet ID is NOT safe for chronology.
    # Use created_at instead.
    collected_ids.sort(
        key=lambda tweet_id: tweets[tweet_id]["created_at"]
    )

    messages = []

    for tweet_id in collected_ids:

        tweet = tweets[tweet_id]

        role = (
            "agent"
            if tweet["author_id"] == BRAND
            else "customer"
        )

        messages.append(
            {
                "tweet_id": tweet_id,
                "role": role,
                "author_id": tweet["author_id"],
                "created_at": tweet["created_at"],
                "text": tweet["text"],
            }
        )

    roles = {message["role"] for message in messages}

    # Need both customer and AppleSupport messages.
    if roles != {"customer", "agent"}:
        continue

    conversations.append(
        {
            "conversation_id": root_id,
            "brand": BRAND,
            "messages": messages,
        }
    )


# -------------------------------------------------------------------
# 6. Save
# -------------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for conversation in conversations:

        f.write(
            json.dumps(
                conversation,
                ensure_ascii=False
            )
            + "\n"
        )


# -------------------------------------------------------------------
# 7. Statistics
# -------------------------------------------------------------------

message_counts = [
    len(conversation["messages"])
    for conversation in conversations
]

print()
print("=" * 70)
print("RECONSTRUCTION RESULTS")
print("=" * 70)

print(
    f"Unique conversations: "
    f"{len(conversations):,}"
)

if message_counts:

    print(
        f"Average messages/conversation: "
        f"{sum(message_counts) / len(message_counts):.2f}"
    )

    print(
        f"Median messages/conversation: "
        f"{sorted(message_counts)[len(message_counts) // 2]}"
    )

    print(
        f"Longest conversation: "
        f"{max(message_counts)} messages"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")