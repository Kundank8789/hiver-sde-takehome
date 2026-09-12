from __future__ import annotations

import json

from escalation import decide_escalation
from reply_generator import ReplyGenerator


def main() -> None:

    print("=" * 80)
    print("HIVER — FINAL OFFLINE PIPELINE SANITY TEST")
    print("=" * 80)

    tests = [
        {
            "message": (
                "My iPhone battery is draining very quickly."
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
                "My Apple ID password is locked."
            ),
            "intent": "ACCOUNT_SECURITY",
            "confidence": 0.95,
        },
        {
            "message": (
                "I need a repair under warranty."
            ),
            "intent": "PURCHASE_REPAIR_WARRANTY",
            "confidence": 0.95,
        },
        {
            "message": (
                "My battery is smoking and getting very hot."
            ),
            "intent": "BATTERY_POWER",
            "confidence": 0.95,
        },
    ]

    for test in tests:

        decision = decide_escalation(
            intent=test["intent"],
            confidence=test["confidence"],
            customer_message=test["message"],
        )

        print()
        print(
            json.dumps(
                {
                    "customer": test["message"],
                    "intent": test["intent"],
                    "confidence": test["confidence"],
                    "should_escalate": (
                        decision.should_escalate
                    ),
                    "reason": decision.reason,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()