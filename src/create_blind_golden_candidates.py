import json
import random
from pathlib import Path

INPUT_PATH = Path("data/apple_support_interactions.jsonl")
OUTPUT_PATH = Path("evaluation/blind_golden_candidates.jsonl")

SEED = 42
TARGET_PER_BUCKET = 25


KEYWORDS = {
    "IOS_UPDATE": [
        "ios update",
        "ios",
        "update",
        "updating",
        "upgrade",
        "software update",
        "install update",
        "can't update",
        "cannot update",
        "macos update",
        "update error",
        "update stuck",
    ],

    "BATTERY_POWER": [
        "battery",
        "battery life",
        "battery drain",
        "draining",
        "charge",
        "charging",
        "charger",
        "power",
        "dies",
    ],

    "PERFORMANCE_STABILITY": [
        "slow",
        "slower",
        "lag",
        "lagging",
        "freeze",
        "freezing",
        "crash",
        "crashing",
        "restart",
        "restarting",
        "stuck",
        "hang",
        "hanging",
        "unresponsive",
        "keeps restarting",
    ],

    "CONNECTIVITY": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "network",
        "connect",
        "connection",
        "connected",
        "no service",
        "hotspot",
    ],

    "APPS_MEDIA": [
        "app",
        "apps",
        "apple music",
        "music",
        "itunes",
        "imovie",
        "facetime",
        "messages",
        "message",
        "mail",
        "photos",
        "photo",
        "youtube",
        "icloud",
    ],

    "DEVICE_HARDWARE": [
        "screen",
        "display",
        "speaker",
        "microphone",
        "keyboard",
        "key",
        "camera",
        "charger port",
        "broken",
        "crack",
        "cracked",
        "damaged",
        "hardware",
        "won't turn on",
        "doesn't turn on",
    ],

    "FEATURE_HOW_TO": [
        "how do i",
        "how can i",
        "how to",
        "where can i",
        "where is",
        "how does",
        "can i use",
        "is there a way",
    ],

    "ACCOUNT_SECURITY": [
        "apple id",
        "password",
        "account",
        "verification",
        "verify",
        "locked",
        "disabled",
        "sign in",
        "login",
        "security",
        "icloud password",
    ],

    "PURCHASE_REPAIR_WARRANTY": [
        "order",
        "ordered",
        "purchase",
        "purchased",
        "buy",
        "bought",
        "refund",
        "warranty",
        "repair",
        "appointment",
        "reservation",
        "store",
        "replacement",
        "service",
    ],

    "OTHER_UNCLEAR": [
        "help",
        "fix this",
        "what is this",
        "wtf",
        "please help",
        "do something",
    ],
}


print("Loading support interactions...")

items = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))

print(f"Loaded {len(items):,} interactions.")


# ---------------------------------------------------------
# Put each interaction into one or more candidate buckets.
# This is ONLY for sampling coverage.
# It is NOT a label.
# ---------------------------------------------------------

buckets = {
    intent: []
    for intent in KEYWORDS
}

for item in items:

    text = item["customer_message"].lower()

    matched = False

    for intent, keywords in KEYWORDS.items():

        if any(keyword in text for keyword in keywords):
            buckets[intent].append(item)
            matched = True

    # Messages with no obvious topical signal.
    if not matched:
        buckets["OTHER_UNCLEAR"].append(item)


# ---------------------------------------------------------
# Sample 25 from each bucket.
# ---------------------------------------------------------

random.seed(SEED)

selected = {}
used_ids = set()

for intent, bucket in buckets.items():

    available = [
        item
        for item in bucket
        if item["interaction_id"] not in used_ids
    ]

    sample_size = min(
        TARGET_PER_BUCKET,
        len(available),
    )

    chosen = random.sample(
        available,
        sample_size,
    )

    selected[intent] = chosen

    for item in chosen:
        used_ids.add(item["interaction_id"])


# ---------------------------------------------------------
# Collect and shuffle.
# ---------------------------------------------------------

all_selected = []

for group in selected.values():
    all_selected.extend(group)

random.shuffle(all_selected)


# ---------------------------------------------------------
# We need exactly 200 candidates.
# ---------------------------------------------------------

if len(all_selected) < 200:

    remaining = [
        item
        for item in items
        if item["interaction_id"] not in used_ids
    ]

    random.shuffle(remaining)

    all_selected.extend(
        remaining[:200 - len(all_selected)]
    )

all_selected = all_selected[:200]


# ---------------------------------------------------------
# IMPORTANT:
# Do NOT store the sampling bucket in the output.
# The annotation sheet must be blind.
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for item in all_selected:

        clean_item = {
            "interaction_id": item["interaction_id"],
            "customer_tweet_id": item["customer_tweet_id"],
            "agent_tweet_id": item["agent_tweet_id"],
            "customer_message": item["customer_message"],
            "context": item["context"],
            "agent_response": item["agent_response"],
        }

        f.write(
            json.dumps(
                clean_item,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("=" * 70)
print("BLIND GOLDEN CANDIDATES")
print("=" * 70)

print(
    f"Candidates created: {len(all_selected)}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)