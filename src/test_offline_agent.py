from __future__ import annotations

import json

from agent import OfflineSupportAgent


TEST_CASES = [
    {
        "message": (
            "My iPhone battery is draining very quickly "
            "after the latest update."
        ),
        "intent": "BATTERY_POWER",
        "confidence": 0.95,
    },
    {
        "message": (
            "Bluetooth won't connect to my car."
        ),
        "intent": "CONNECTIVITY",
        "confidence": 0.90,
    },
    {
        "message": (
            "My Apple ID password is locked "
            "and I cannot sign in."
        ),
        "intent": "ACCOUNT_SECURITY",
        "confidence": 0.95,
    },
    {
        "message": (
            "How do I turn off notifications on my iPhone?"
        ),
        "intent": "FEATURE_HOW_TO",
        "confidence": 0.95,
    },
    {
        "message": (
            "I need a repair under warranty."
        ),
        "intent": "PURCHASE_REPAIR_WARRANTY",
        "confidence": 0.95,
    },
]


def main() -> None:

    print("=" * 80)
    print("HIVER — OFFLINE END-TO-END AGENT TEST")
    print("=" * 80)

    agent = OfflineSupportAgent()

    for case in TEST_CASES:

        print()
        print("=" * 80)
        print("CUSTOMER:")
        print(case["message"])
        print("=" * 80)

        result = agent.handle(
            customer_message=case["message"],
            intent=case["intent"],
            confidence=case["confidence"],
        )

        print(
            json.dumps(
                {
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "should_escalate": result[
                        "should_escalate"
                    ],
                    "escalation_reason": result[
                        "escalation_reason"
                    ],
                    "reply": result["reply"],
                    "grounded": result["grounded"],
                    "evidence_count": result[
                        "evidence_count"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()