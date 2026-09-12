from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str


ACCOUNT_INTENTS = {
    "ACCOUNT_SECURITY",
}

CASE_SPECIFIC_INTENTS = {
    "PURCHASE_REPAIR_WARRANTY",
}

HIGH_RISK_PATTERNS = (
    "battery is smoking",
    "battery smoking",
    "battery is on fire",
    "battery caught fire",
    "battery caught on fire",
    "phone is smoking",
    "phone smoking",
    "device is smoking",
    "device smoking",
    "phone is overheating",
    "phone overheating",
    "device is overheating",
    "device overheating",
    "battery is overheating",
    "battery overheating",
    "swollen battery",
    "swollen iphone",
    "swollen phone",
    "exploded",
    "explosion",
    "fire",
    "smoke",
)


def normalize(text: str) -> str:
    return " ".join(
        str(text).lower().split()
    )


def decide_escalation(
    *,
    intent: str,
    confidence: float,
    customer_message: str,
) -> EscalationDecision:

    message = normalize(
        customer_message
    )

    # --------------------------------------------------
    # Rule 1: Safety-critical conditions.
    # Highest priority.
    # --------------------------------------------------

    if any(
        pattern in message
        for pattern in HIGH_RISK_PATTERNS
    ):
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "The message indicates a potentially unsafe "
                "battery or hardware condition requiring immediate "
                "human handling."
            ),
        )

    # --------------------------------------------------
    # Rule 2: Account security.
    # --------------------------------------------------

    if intent in ACCOUNT_INTENTS:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "Account-specific access or security investigation "
                "is required."
            ),
        )

    # --------------------------------------------------
    # Rule 3: Purchase/repair/warranty.
    # --------------------------------------------------

    if intent in CASE_SPECIFIC_INTENTS:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "Purchase, repair, warranty, refund, replacement, "
                "or appointment support requires case-specific handling."
            ),
        )

    # --------------------------------------------------
    # Rule 4: Low-confidence classification.
    # --------------------------------------------------

    if confidence < 0.70:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "Intent confidence is below the safe "
                "auto-handling threshold."
            ),
        )

    # --------------------------------------------------
    # Rule 5: Very short / insufficient messages.
    # --------------------------------------------------

    words = message.split()

    if len(words) < 5:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "The customer message does not contain enough "
                "information for safe automated resolution."
            ),
        )

    vague_signals = {
        "fix this",
        "fix it",
        "do something",
        "please help",
        "help me",
        "help",
    }

    if message in vague_signals:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "The customer has not provided enough information "
                "to determine a safe automated resolution."
            ),
        )

    # --------------------------------------------------
    # Default: suitable for automated handling.
    # --------------------------------------------------

    return EscalationDecision(
        should_escalate=False,
        reason=(
            "The issue appears suitable for standard "
            "automated support based on the available information."
        ),
    )