from __future__ import annotations

import json
from typing import Any

from llm_client import GroqClient


INTENTS = (
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
)


INTENT_DESCRIPTIONS = {
    "IOS_UPDATE": (
        "Problems installing, updating, upgrading, or recovering "
        "from an iOS/macOS software update."
    ),
    "BATTERY_POWER": (
        "Battery drain, charging, charger, battery life, or power problems."
    ),
    "PERFORMANCE_STABILITY": (
        "Slowness, lag, freezing, crashing, restarting, hanging, "
        "or general device instability."
    ),
    "CONNECTIVITY": (
        "Wi-Fi, Bluetooth, cellular, network, hotspot, or connection problems."
    ),
    "APPS_MEDIA": (
        "Problems involving Apple apps, media, messaging, music, "
        "photos, video, or software services."
    ),
    "DEVICE_HARDWARE": (
        "Physical device faults, damaged components, broken parts, "
        "or hardware failures."
    ),
    "FEATURE_HOW_TO": (
        "Questions about how to use, configure, enable, disable, "
        "or understand an Apple feature."
    ),
    "ACCOUNT_SECURITY": (
        "Apple ID, password, account access, authentication, "
        "verification, or security problems."
    ),
    "PURCHASE_REPAIR_WARRANTY": (
        "Purchases, orders, refunds, reservations, repairs, "
        "service appointments, accessories, or warranty questions."
    ),
    "OTHER_UNCLEAR": (
        "The message is too ambiguous or does not reliably fit "
        "one of the defined support intents."
    ),
}


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": list(INTENTS),
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


SYSTEM_PROMPT = """You are an AppleSupport customer-intent classifier.

Classify the customer's PRIMARY support need into exactly one
of the provided intents.

Rules:
1. Focus on what the customer currently needs help with.
2. Use conversation context when it clarifies the current message.
3. Do not classify based only on keywords.
4. If multiple unrelated problems are present and no primary problem
   is clear, choose OTHER_UNCLEAR.
5. Do not invent information that is not present.
6. Confidence must reflect genuine uncertainty.
7. Return only the required structured output.
"""


class IntentClassifier:
    """LLM-based AppleSupport intent classifier."""

    def __init__(
        self,
        client: GroqClient | None = None,
    ) -> None:
        self.client = client or GroqClient()

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
                "Relevant previous context:\n"
                f"{context.strip()}\n"
            )

        raw = self.client.structured(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        user_content
                        + "\nChoose exactly one intent."
                    ),
                },
            ],
            schema_name="apple_support_intent",
            schema=INTENT_SCHEMA,
            temperature=0.0,
        )

        result = json.loads(raw)

        # Defensive application-level validation.
        intent = result.get("intent")
        confidence = result.get("confidence")
        reason = result.get("reason")

        if intent not in INTENTS:
            raise ValueError(
                f"Invalid intent returned: {intent}"
            )

        if not isinstance(confidence, (int, float)):
            raise ValueError(
                "Classifier confidence must be numeric."
            )

        confidence = float(confidence)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Confidence outside [0, 1]: {confidence}"
            )

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "Classifier reason cannot be empty."
            )

        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason.strip(),
        }