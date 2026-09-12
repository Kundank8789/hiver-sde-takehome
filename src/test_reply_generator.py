from __future__ import annotations

from groq_reply_generator import GroqReplyGenerator


def main() -> None:

    generator = GroqReplyGenerator()

    historical_cases = [
        {
            "customer_message": (
                "My battery drains very quickly after the update."
            ),
            "agent_response": (
                "We know how important battery life is. "
                "Let's work together to investigate where your charge is going."
            ),
        },
        {
            "customer_message": (
                "My iPhone battery is draining very quickly."
            ),
            "agent_response": (
                "We'd like to help investigate the battery issue. "
                "Let's continue support through a private channel."
            ),
        },
    ]

    result = generator.generate(
        customer_message=(
            "My iPhone battery is draining very quickly "
            "after the latest update."
        ),
        intent="BATTERY_POWER",
        historical_cases=historical_cases,
        should_escalate=False,
        escalation_reason=(
            "Suitable for standard automated support."
        ),
    )

    print("=" * 80)
    print("GROQ REPLY GENERATOR TEST")
    print("=" * 80)

    print(
        f"\nReply:\n{result['reply']}"
    )

    print(
        f"\nGrounded: "
        f"{result['grounded']}"
    )

    print(
        f"Evidence used: "
        f"{result['evidence_used']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )


if __name__ == "__main__":
    main()