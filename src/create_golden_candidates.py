import json
import random
from pathlib import Path

INPUT_PATH = Path("data/apple_support_interactions.jsonl")
OUTPUT_PATH = Path("evaluation/golden_candidates.jsonl")

SEED = 42
CANDIDATE_COUNT = 400


KEYWORDS = {
    "IOS_UPDATE": [
        "update",
        "ios",
        "upgrade",
        "install",
        "updating",
        "software update",
        "macos",
    ],
    "BATTERY_POWER": [
        "battery",
        "charging",
        "charger",
        "charge",
        "power",
        "drain",
        "draining",
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
        "unresponsive",
    ],
    "CONNECTIVITY": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "connection",
        "connect",
        "cellular",
    ],
    "APPS_MEDIA": [
        "music",
        "itunes",
        "imovie",
        "facetime",
        "message",
        "messages",
        "photo",
        "photos",
        "app",
        "youtube",
    ],
    "DEVICE_HARDWARE": [
        "screen",
        "speaker",
        "keyboard",
        "key",
        "broken",
        "damage",
        "hardware",
        "turns on",
        "won't turn on",
    ],
    "FEATURE_HOW_TO": [
        "how do i",
        "how can i",
        "how to",
        "where is",
        "can i use",
        "how does",
    ],
    "ACCOUNT_SECURITY": [
        "apple id",
        "password",
        "account",
        "security",
        "verify",
        "verification",
        "login",
        "sign in",
    ],
    "PURCHASE_REPAIR_WARRANTY": [
        "order",
        "purchase",
        "buy",
        "reservation",
        "warranty",
        "repair",
        "store",
        "appointment",
        "refund",
    ],
}


def heuristic_intents(text: str):
    text = text.lower()

    scores = {}

    for intent, words in KEYWORDS.items():

        score = 0

        for word in words:
            if word in text:
                score += 1

        if score:
            scores[intent] = score

    if not scores:
        return ["OTHER_UNCLEAR"]

    max_score = max(scores.values())

    return [
        intent
        for intent, score in scores.items()
        if score == max_score
    ]


print("Loading support interactions...")

items = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))

print(f"Loaded {len(items):,} interactions.")


# ---------------------------------------------------------
# Group examples by provisional intent
# ---------------------------------------------------------

groups = {}

for item in items:

    intent_candidates = heuristic_intents(
        item["customer_message"]
    )

    for intent in intent_candidates:

        groups.setdefault(intent, []).append(item)


# ---------------------------------------------------------
# Sample from each group
# ---------------------------------------------------------

random.seed(SEED)

selected = []
seen = set()

per_group = max(
    CANDIDATE_COUNT // len(groups),
    1
)

for intent, group in groups.items():

    sample_size = min(
        per_group,
        len(group)
    )

    for item in random.sample(
        group,
        sample_size
    ):
        interaction_id = item["interaction_id"]

        if interaction_id in seen:
            continue

        item["provisional_intent"] = intent

        selected.append(item)
        seen.add(interaction_id)


# Fill remaining slots randomly.

remaining = [
    item
    for item in items
    if item["interaction_id"] not in seen
]

random.shuffle(remaining)

for item in remaining:

    if len(selected) >= CANDIDATE_COUNT:
        break

    item["provisional_intent"] = "OTHER_UNCLEAR"
    selected.append(item)


random.shuffle(selected)

selected = selected[:CANDIDATE_COUNT]


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

    for item in selected:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            ) + "\n"
        )


print()
print("=" * 70)
print("GOLDEN SET CANDIDATES")
print("=" * 70)

print(f"Candidates generated: {len(selected)}")
print(f"Saved to: {OUTPUT_PATH}")