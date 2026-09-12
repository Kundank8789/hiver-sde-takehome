from __future__ import annotations

import json

from support_agent import SupportAgent


def main() -> None:
    print("=" * 80)
    print("HIVER — FULL SUPPORT AGENT SMOKE TEST")
    print("=" * 80)

    message = (
        "My iPhone battery is draining very quickly "
        "after the latest update."
    )

    print("\nCustomer:")
    print(message)

    agent = SupportAgent()

    result = agent.handle(
        customer_message=message,
        top_k=5,
    )

    output = {
        "intent": result["intent"],
        "confidence": result["confidence"],
        "classification_reason": result[
            "classification_reason"
        ],
        "should_escalate": result[
            "should_escalate"
        ],
        "escalation_reason": result[
            "escalation_reason"
        ],
        "reply": result["reply"],
        "grounded": result["grounded"],
        "evidence_used": result[
            "evidence_used"
        ],
        "evidence_score": result[
            "evidence_score"
        ],
    }

    print("\nAgent output:")
    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()