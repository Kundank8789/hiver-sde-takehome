import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")

BRANDS = [
    "AppleSupport",
    "AmazonHelp",
    "Uber_Support",
]

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
)

# Fast lookup by tweet ID
tweets = df.set_index("tweet_id")

for brand in BRANDS:

    print("\n")
    print("=" * 100)
    print(f"BRAND: {brand}")
    print("=" * 100)

    brand_tweets = df[
        df["author_id"] == brand
    ]

    # Brand tweets which directly respond to customers
    replies = brand_tweets[
        brand_tweets["in_response_to_tweet_id"].notna()
    ]

    # Take a deterministic sample so results are reproducible
    sample = replies.sample(
        n=min(15, len(replies)),
        random_state=42
    )

    for i, (_, brand_reply) in enumerate(sample.iterrows(), 1):

        parent_id = int(
            brand_reply["in_response_to_tweet_id"]
        )

        if parent_id not in tweets.index:
            continue

        customer = tweets.loc[parent_id]

        print(f"\n--- Conversation {i} ---")

        print(
            f"CUSTOMER [{parent_id}]:\n"
            f"{customer['text']}"
        )

        print(
            f"\n{brand} [{brand_reply['tweet_id']}]:\n"
            f"{brand_reply['text']}"
        )

print("\nDone.")