from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "apple_support_interactions.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "weak_training.jsonl"
)


# High-confidence patterns only.
# These are training labels, NOT evaluation ground truth.

PATTERNS: dict[str, tuple[str, ...]] = {
    "IOS_UPDATE": (
        r"\bios\s*(?:10|11|12|13|14|15|16|17|18|19|20)\b",
        r"\bios\s+update\b",
        r"\bsoftware\s+update\b",
        r"\bupdate\s+(?:my|the)\s+(?:iphone|ipad|mac|macbook)\b",
        r"\b(?:can't|cannot|won't|doesn't)\s+(?:install|download|complete)\s+(?:the\s+)?update\b",
        r"\bupdate\s+(?:is\s+)?stuck\b",
        r"\bupdate\s+error\b",
        r"\bupgrad(?:e|ing)\s+to\s+ios\b",
        r"\bmacos\s+(?:update|upgrade)\b",
    ),

    "BATTERY_POWER": (
        r"\bbattery\s+(?:drain|draining|life|dies|dead)\b",
        r"\bbattery\s+.*\bdrain(?:s|ing)?\b",
        r"\b(?:charge|charging)\s+.*\bbattery\b",
        r"\b(?:won't|doesn't)\s+charge\b",
        r"\bphone\s+(?:dies|shuts\s+down)\s+.*\bbattery\b",
        r"\bcharger\b",
    ),

    "PERFORMANCE_STABILITY": (
        r"\b(?:slow|slower|lag|lagging)\b",
        r"\b(?:freeze|freezes|freezing|frozen)\b",
        r"\b(?:crash|crashes|crashing)\b",
        r"\b(?:restart|restarts|restarting|reboot|reboots)\b",
        r"\b(?:stuck|hanging|hangs|unresponsive)\b",
    ),

    "CONNECTIVITY": (
        r"\bwi[\s-]?fi\b",
        r"\bbluetooth\b",
        r"\b(?:no\s+service|cellular\s+(?:data|network)|mobile\s+data)\b",
        r"\b(?:can't|cannot|won't)\s+connect\b",
        r"\bconnection\s+(?:problem|issue)\b",
        r"\bhotspot\b",
    ),

    "APPS_MEDIA": (
        r"\bapple\s+music\b",
        r"\bitunes\b",
        r"\bimovie\b",
        r"\bfacetime\b",
        r"\bmessages?\s+(?:app|issue|problem|not\s+working)\b",
        r"\bphotos?\s+(?:app|issue|problem|not\s+working)\b",
        r"\byoutube\b",
        r"\bapp\s+store\s+(?:app|issue|problem|not\s+working)\b",
        r"\b(?:music|photo|video)\s+(?:app|issue|problem)\b",
    ),

    "DEVICE_HARDWARE": (
        r"\bcracked?\s+(?:screen|display)\b",
        r"\bbroken\s+(?:screen|display|keyboard|speaker|camera)\b",
        r"\b(?:screen|display|speaker|camera|microphone)\s+(?:isn't|is\s+not|won't|doesn't)\s+working\b",
        r"\b(?:key|keys)\s+(?:fell|fall|broken)\b",
        r"\bphysical\s+damage\b",
        r"\bhardware\s+(?:problem|issue|failure)\b",
        r"\bwon't\s+turn\s+on\b",
        r"\bdoesn't\s+turn\s+on\b",
    ),

    "FEATURE_HOW_TO": (
        r"\bhow\s+do\s+i\b",
        r"\bhow\s+can\s+i\b",
        r"\bhow\s+to\b",
        r"\bwhere\s+(?:is|can\s+i|do\s+i)\b",
        r"\bis\s+there\s+(?:a\s+way|a\s+setting)\b",
        r"\bhow\s+does\s+(?:this|it|the)\b",
    ),

    "ACCOUNT_SECURITY": (
        r"\bapple\s+id\b",
        r"\b(?:apple\s+id|icloud)\s+(?:password|account)\b",
        r"\bpassword\s+(?:is\s+)?(?:locked|disabled|reset)\b",
        r"\baccount\s+(?:locked|disabled|verification|verify)\b",
        r"\b(?:sign\s+in|log\s+in|login)\s+(?:problem|issue|error)\b",
        r"\bverification\s+(?:code|problem|issue)\b",
    ),

    "PURCHASE_REPAIR_WARRANTY": (
        r"\border\b",
        r"\brefund\b",
        r"\bwarranty\b",
        r"\brepair\b",
        r"\bappointment\b",
        r"\breservation\b",
        r"\breplacement\b",
        r"\bpurchase(?:d)?\b",
        r"\bbuy(?:ing)?\b",
        r"\baccessor(?:y|ies)\b",
    ),
}


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def matching_intents(text: str) -> list[str]:
    matches: list[str] = []

    for intent, patterns in PATTERNS.items():
        if any(
            re.search(pattern, text)
            for pattern in patterns
        ):
            matches.append(intent)

    return matches


def main() -> None:
    print("=" * 72)
    print("HIVER — HIGH-PRECISION WEAK LABELING")
    print("=" * 72)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    total = 0
    accepted = 0
    rejected = 0

    class_counts: dict[str, int] = {}

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as source, OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as destination:

        for line in source:

            if not line.strip():
                continue

            total += 1

            item = json.loads(line)

            text = normalize(
                item["customer_message"]
            )

            matches = matching_intents(text)

            # Reject ambiguous examples.
            if len(matches) != 1:
                rejected += 1
                continue

            intent = matches[0]

            output = {
                "interaction_id": item[
                    "interaction_id"
                ],
                "text": item[
                    "customer_message"
                ],
                "intent": intent,
            }

            destination.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

            accepted += 1

            class_counts[intent] = (
                class_counts.get(intent, 0)
                + 1
            )

    print()
    print(f"Interactions processed: {total:,}")
    print(f"High-confidence labels: {accepted:,}")
    print(f"Rejected/ambiguous: {rejected:,}")

    print("\nDistribution:")

    for intent, count in sorted(
        class_counts.items(),
    ):
        print(
            f"  {intent:<28} {count:>8,}"
        )

    print()
    print(f"Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()