import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")

# Top candidates from our first analysis
BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "AskPlayStation",
]

print("Loading required columns...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
)

print(f"Loaded {len(df):,} tweets.")

# Brand tweets
brand_df = df[df["author_id"].isin(BRANDS)].copy()

# Customer tweets
customer_df = df[~df["author_id"].isin(BRANDS)].copy()

results = []

for brand in BRANDS:

    brand_tweets = brand_df[
        brand_df["author_id"] == brand
    ]

    # Brand tweets that directly reply to another tweet
    brand_replies = brand_tweets[
        brand_tweets["in_response_to_tweet_id"].notna()
    ]

    # IDs of customers receiving a direct brand reply
    customer_ids = set(
        brand_replies["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
    )

    # Customer messages that received a direct reply
    customer_messages = customer_df[
        customer_df["tweet_id"].isin(customer_ids)
    ]

    results.append(
        {
            "brand": brand,
            "brand_tweets": len(brand_tweets),
            "brand_replies": len(brand_replies),
            "customer_messages_replied_to": len(customer_messages),
            "reply_rate": (
                len(brand_replies) / len(brand_tweets)
                if len(brand_tweets) else 0
            ),
        }
    )

result = pd.DataFrame(results)

result = result.sort_values(
    "customer_messages_replied_to",
    ascending=False
)

print("\n" + "=" * 90)
print("BRAND SUPPORT INTERACTION ANALYSIS")
print("=" * 90)

print(
    result.to_string(
        index=False,
        formatters={
            "reply_rate": "{:.2%}".format
        }
    )
)

result.to_csv(
    "data/brand_interactions.csv",
    index=False
)

print("\nSaved:")
print("data/brand_interactions.csv")