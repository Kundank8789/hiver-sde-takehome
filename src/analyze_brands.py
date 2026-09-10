import pandas as pd
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")

print("Analyzing dataset in chunks...\n")

author_total = Counter()
author_inbound = Counter()
author_outbound = Counter()

CHUNK_SIZE = 100_000

for i, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=["author_id", "inbound"],
        chunksize=CHUNK_SIZE
    ),
    start=1
):
    author_total.update(chunk["author_id"])

    inbound_counts = chunk.loc[chunk["inbound"], "author_id"].value_counts()
    outbound_counts = chunk.loc[~chunk["inbound"], "author_id"].value_counts()

    author_inbound.update(inbound_counts.to_dict())
    author_outbound.update(outbound_counts.to_dict())

    print(f"Processed chunk {i}...")


# A likely brand account should have significant outbound activity.
candidates = []

for author, outbound_count in author_outbound.items():
    candidates.append(
        {
            "brand": author,
            "total_tweets": author_total[author],
            "inbound_tweets": author_inbound[author],
            "outbound_tweets": outbound_count,
        }
    )

result = pd.DataFrame(candidates)

result["outbound_ratio"] = (
    result["outbound_tweets"] / result["total_tweets"]
)

# Sort by outbound/support activity
result = result.sort_values(
    "outbound_tweets",
    ascending=False
)

print("\n" + "=" * 80)
print("TOP 50 ACCOUNTS BY OUTBOUND / SUPPORT ACTIVITY")
print("=" * 80)

print(
    result.head(50).to_string(index=False)
)

# Save results
Path("data").mkdir(exist_ok=True)

result.to_csv(
    "data/brand_candidates.csv",
    index=False
)

print("\nSaved:")
print("data/brand_candidates.csv")