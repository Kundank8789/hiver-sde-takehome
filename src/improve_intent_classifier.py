from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = PROJECT_ROOT / "evaluation" / "golden_set.csv"
WEAK_TRAINING_PATH = PROJECT_ROOT / "data" / "weak_training.jsonl"

MODEL = "openai/gpt-oss-20b"

INTENTS = [
    "IOS_UPDATE",
    "BATTERY_POWER",
    "PERFORMANCE_STABILITY",
    "CONNECTIVITY",
    "APPS_MEDIA",
    "DEVICE_HARDWARE",
    "FEATURE_HOW_TO",
    "ACCOUNT_SECURITY",
    "PURCHASE_REPAIR_WARRANTY",
    "OTHER_UNCLEAR",
]

# ------------------------------------------------------------
# Explicit examples for important boundary cases.
# These are generic examples, not evaluation examples.
# ------------------------------------------------------------

CURATED_EXAMPLES = {
    "IOS_UPDATE": [
        "I cannot install the latest iOS update.",
        "My iPhone update is stuck on verifying.",
    ],
    "BATTERY_POWER": [
        "My iPhone battery drains very quickly.",
        "My phone will not charge properly.",
    ],
    "PERFORMANCE_STABILITY": [
        "My iPhone keeps freezing and restarting.",
        "My phone has become extremely slow.",
    ],
    "CONNECTIVITY": [
        "Bluetooth will not connect to my car.",
        "My iPhone cannot connect to Wi-Fi.",
    ],
    "APPS_MEDIA": [
        "Apple Music keeps crashing on my iPhone.",
        "Photos is not opening correctly.",
    ],
    "DEVICE_HARDWARE": [
        "My iPhone screen is cracked.",
        "My iPad will not turn on.",
    ],
    "FEATURE_HOW_TO": [
        "How do I turn off notifications on my iPhone?",
        "How can I enable this feature?",
    ],
    "ACCOUNT_SECURITY": [
        "My Apple ID is locked and I cannot sign in.",
        "I forgot my Apple ID password.",
    ],
    "PURCHASE_REPAIR_WARRANTY": [
        "How do I arrange a repair under warranty?",
        "I need a replacement for my defective iPhone.",
    ],
    "OTHER_UNCLEAR": [
        "Please fix this.",
        "This is broken. Help.",
    ],
}


INTENT_GUIDELINES = {
    "IOS_UPDATE": (
        "The update itself is the primary problem: downloading, installing, "
        "verifying, or failing to complete an iOS/macOS update."
    ),
    "BATTERY_POWER": (
        "The primary problem is battery drain, charging, battery life, "
        "charger behavior, or power."
    ),
    "PERFORMANCE_STABILITY": (
        "The primary problem is freezing, crashing, restarting, lag, "
        "slowness, hanging, or general device instability."
    ),
    "CONNECTIVITY": (
        "The primary problem is Wi-Fi, Bluetooth, cellular, mobile data, "
        "hotspot, network, or connection failure."
    ),
    "APPS_MEDIA": (
        "The primary problem is a specific app or media service such as "
        "Music, Photos, Messages, FaceTime, iMovie, YouTube, or App Store."
    ),
    "DEVICE_HARDWARE": (
        "There is evidence of physical damage or hardware failure such as "
        "a cracked screen, broken component, or device not powering on."
    ),
    "FEATURE_HOW_TO": (
        "The customer is primarily asking how to use, configure, enable, "
        "disable, or understand an Apple feature or setting."
    ),
    "ACCOUNT_SECURITY": (
        "The primary problem concerns Apple ID, account access, password, "
        "authentication, verification, or security."
    ),
    "PURCHASE_REPAIR_WARRANTY": (
        "The primary problem concerns an order, purchase, refund, repair, "
        "warranty, replacement, appointment, or service."
    ),
    "OTHER_UNCLEAR": (
        "The message is too vague, incomplete, or ambiguous to identify "
        "a reliable primary support intent."
    ),
}


SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": INTENTS,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "intent",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


# ============================================================
# Text utilities
# ============================================================

def normalize(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def useful_example(text: str) -> bool:
    """
    Keep only demonstrations with enough semantic content.
    """

    text = " ".join(text.split())

    if len(text) < 55:
        return False

    if len(text) > 260:
        return False

    alpha_words = [
        word
        for word in text.split()
        if any(ch.isalpha() for ch in word)
    ]

    if len(alpha_words) < 9:
        return False

    # Reject messages dominated by mentions/emoji/noise.
    mention_count = text.count("@")

    if mention_count > 4:
        return False

    return True


# ============================================================
# Golden-set leakage protection
# ============================================================

def load_golden_ids() -> set[str]:

    golden = pd.read_csv(
        GOLDEN_PATH,
        usecols=["interaction_id"],
    )

    return {
        str(value).strip()
        for value in golden["interaction_id"]
        if str(value).strip()
        and str(value).strip() != "nan"
    }


# ============================================================
# Weak examples
# ============================================================

def load_weak_examples(
    golden_ids: set[str],
) -> dict[str, list[str]]:

    buckets = {
        intent: []
        for intent in INTENTS
        if intent != "OTHER_UNCLEAR"
    }

    with WEAK_TRAINING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            interaction_id = str(
                item.get("interaction_id", "")
            ).strip()

            intent = str(
                item.get("intent", "")
            ).strip()

            text = str(
                item.get("text", "")
            ).strip()

            if interaction_id in golden_ids:
                continue

            if intent not in buckets:
                continue

            if not useful_example(text):
                continue

            buckets[intent].append(text)

    return buckets


# ============================================================
# Build few-shot examples
# ============================================================

def build_examples(
    golden_ids: set[str],
) -> dict[str, list[str]]:

    weak = load_weak_examples(
        golden_ids
    )

    examples: dict[str, list[str]] = {}

    for intent in INTENTS:

        selected = list(
            CURATED_EXAMPLES.get(
                intent,
                [],
            )
        )

        # Add at most one historical weak example.
        for candidate in weak.get(
            intent,
            [],
        ):

            if candidate not in selected:

                selected.append(candidate)

                break

        examples[intent] = selected[:3]

    return examples


# ============================================================
# Prompt
# ============================================================

def build_system_prompt(
    examples: dict[str, list[str]],
) -> str:

    parts = [
        "You classify AppleSupport customer messages.",
        "",
        "Return exactly one PRIMARY support intent.",
        "",
        "HARD RULES:",
        "1. If the customer explicitly asks HOW to do something, "
        "enable something, disable something, or configure a feature, "
        "use FEATURE_HOW_TO unless the primary request is actually "
        "account access, purchase/repair, or another more specific "
        "operational problem.",
        "2. If the primary symptom is battery drain or charging, "
        "use BATTERY_POWER even when the customer blames an iOS update.",
        "3. If the primary symptom is Bluetooth/Wi-Fi/cellular/network "
        "connection failure, use CONNECTIVITY even when an update is "
        "mentioned as the cause.",
        "4. Use IOS_UPDATE only when the update process itself is failing.",
        "5. Use DEVICE_HARDWARE only when there is evidence of physical "
        "damage or actual hardware failure.",
        "6. Use PURCHASE_REPAIR_WARRANTY for purchase, refund, repair, "
        "replacement, warranty, or appointment requests.",
        "7. Use ACCOUNT_SECURITY for Apple ID, passwords, verification, "
        "authentication, or account access.",
        "8. Use OTHER_UNCLEAR only when the primary issue genuinely "
        "cannot be determined.",
        "",
        "INTENT DEFINITIONS:",
    ]

    for intent in INTENTS:

        parts.append(
            f"\n{intent}: "
            f"{INTENT_GUIDELINES[intent]}"
        )

        examples_for_intent = examples.get(
            intent,
            [],
        )

        if examples_for_intent:

            parts.append(
                "Examples:"
            )

            for example in examples_for_intent:

                parts.append(
                    f"- {example}"
                )

    parts.extend(
        [
            "",
            "MULTI-ISSUE RULE:",
            "Choose the customer's primary support need, not every symptom.",
            "",
            "When uncertain, use OTHER_UNCLEAR with lower confidence.",
            "",
            "Return only the requested structured output.",
        ]
    )

    return "\n".join(parts)


# ============================================================
# Classifier
# ============================================================

class ImprovedIntentClassifier:

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.8,
    ) -> None:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = MODEL
        self.max_retries = max_retries
        self.base_delay = base_delay

        golden_ids = load_golden_ids()

        self.examples = build_examples(
            golden_ids
        )

        self.system_prompt = build_system_prompt(
            self.examples
        )

    def classify(
        self,
        customer_message: str,
        context: str = "",
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        user_content = (
            "Customer message:\n"
            f"{customer_message.strip()}\n\n"
        )

        if context.strip():
            user_content += (
                "Conversation context:\n"
                f"{context.strip()}\n\n"
            )

        user_content += (
            "Classify the PRIMARY customer support need."
        )

        last_error: Exception | None = None

        for attempt in range(
            self.max_retries + 1
        ):

            try:

                response = (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": self.system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_content,
                            },
                        ],
                        temperature=0.0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "apple_support_intent",
                                "strict": True,
                                "schema": SCHEMA,
                            },
                        },
                    )
                )

                if not response.choices:
                    raise RuntimeError(
                        "Groq returned no choices."
                    )

                content = (
                    response.choices[0]
                    .message
                    .content
                )

                if not content:
                    raise RuntimeError(
                        "Groq returned empty content."
                    )

                result = json.loads(
                    content
                )

                intent = result.get(
                    "intent"
                )

                confidence = float(
                    result.get(
                        "confidence"
                    )
                )

                reason = str(
                    result.get(
                        "reason",
                        ""
                    )
                ).strip()

                if intent not in INTENTS:
                    raise ValueError(
                        f"Invalid intent: {intent}"
                    )

                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"Invalid confidence: {confidence}"
                    )

                if not reason:
                    raise ValueError(
                        "Empty classification reason."
                    )

                return {
                    "intent": intent,
                    "confidence": confidence,
                    "reason": reason,
                    "attempts": attempt + 1,
                }

            except Exception as exc:

                last_error = exc

                if attempt >= self.max_retries:
                    break

                delay = (
                    self.base_delay
                    * (2 ** attempt)
                )

                time.sleep(
                    delay
                )

        return {
            "intent": "OTHER_UNCLEAR",
            "confidence": 0.0,
            "reason": (
                "Classifier failed after retries; "
                "safe fallback applied."
            ),
            "attempts": self.max_retries + 1,
            "error": str(last_error),
        }


# ============================================================
# Smoke test
# ============================================================

def main() -> None:

    classifier = ImprovedIntentClassifier()

    print("=" * 72)
    print("IMPROVED GROQ INTENT CLASSIFIER")
    print("=" * 72)

    for intent, examples in classifier.examples.items():

        print(f"\n{intent}:")

        for example in examples:
            print(
                f"  - {example[:220]}"
            )

    print(
        "\nGolden-set IDs excluded from historical demonstrations."
    )

    tests = [
        "My iPhone battery is draining extremely fast.",
        "I can't install the latest iOS update.",
        "Bluetooth won't connect to my car.",
        "My Apple ID password is locked.",
        "How do I turn off notifications on my iPhone?",
        "How do I enable Dark Mode?",
        "My iPhone keeps freezing and restarting.",
        "My screen is cracked.",
        "I need a repair under warranty.",
    ]

    print("\nSmoke tests:")

    for text in tests:

        result = classifier.classify(
            customer_message=text
        )

        print(
            f"\n{text}\n"
            f" -> {result['intent']} "
            f"({result['confidence']:.2f})\n"
            f"    {result['reason']}"
        )

        if result.get("error"):
            print(
                f"    ERROR: {result['error']}"
            )

            print(
                f"    Attempts: {result.get('attempts', 1)}"
            )


if __name__ == "__main__":
    main()