import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/twcs.csv")
OUTPUT_PATH = Path("data/apple_support_interactions.jsonl")

BRAND = "AppleSupport"
MAX_CONTEXT_MESSAGES = 6


print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ]
)

print(f"Loaded {len(df):,} tweets.")


# ---------------------------------------------------------
# Build fast tweet lookup
# ---------------------------------------------------------

tweets = df.set_index("tweet_id").to_dict("index")

brand_tweets = df[
    df["author_id"] == BRAND
].copy()

print(
    f"AppleSupport responses: "
    f"{len(brand_tweets):,}"
)


# ---------------------------------------------------------
# Build one interaction for every AppleSupport response
# ---------------------------------------------------------

interactions = []

for _, agent_row in brand_tweets.iterrows():

    agent_id = int(agent_row["tweet_id"])

    parent_id = agent_row["in_response_to_tweet_id"]

    # An AppleSupport tweet without a parent cannot give us
    # a customer message to evaluate.
    if pd.isna(parent_id):
        continue

    parent_id = int(parent_id)

    if parent_id not in tweets:
        continue

    customer = tweets[parent_id]

    context = []

    current_id = parent_id
    visited = set()

    # Walk backwards through the actual parent chain.
    while (
        current_id in tweets
        and current_id not in visited
        and len(context) < MAX_CONTEXT_MESSAGES
    ):

        visited.add(current_id)

        message = tweets[current_id]

        role = (
            "agent"
            if message["author_id"] == BRAND
            else "customer"
        )

        context.append(
            {
                "tweet_id": int(current_id),
                "role": role,
                "author_id": message["author_id"],
                "created_at": message["created_at"],
                "text": message["text"],
            }
        )

        previous_id = message["in_response_to_tweet_id"]

        if pd.isna(previous_id):
            break

        current_id = int(previous_id)

    # Put context in chronological order.
    context.reverse()

    # The direct parent is always our current customer message.
    interactions.append(
        {
            "interaction_id": agent_id,
            "brand": BRAND,
            "customer_tweet_id": parent_id,
            "agent_tweet_id": agent_id,
            "customer_message": customer["text"],
            "agent_response": agent_row["text"],
            "context": context,
        }
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for interaction in interactions:
        f.write(
            json.dumps(
                interaction,
                ensure_ascii=False
            ) + "\n"
        )


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

context_lengths = [
    len(item["context"])
    for item in interactions
]

print()
print("=" * 70)
print("SUPPORT INTERACTION DATASET")
print("=" * 70)

print(
    f"AppleSupport responses: "
    f"{len(brand_tweets):,}"
)

print(
    f"Usable support interactions: "
    f"{len(interactions):,}"
)

if context_lengths:

    print(
        f"Average context messages: "
        f"{sum(context_lengths) / len(context_lengths):.2f}"
    )

    print(
        f"Maximum context messages: "
        f"{max(context_lengths)}"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")