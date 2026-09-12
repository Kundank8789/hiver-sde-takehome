from __future__ import annotations

import re
from typing import Any


INTENT_OPENERS = {
    "IOS_UPDATE": (
        "We can help with the iOS update issue."
    ),
    "BATTERY_POWER": (
        "We can help with the battery or charging issue."
    ),
    "PERFORMANCE_STABILITY": (
        "We can help troubleshoot the performance issue."
    ),
    "CONNECTIVITY": (
        "We can help troubleshoot the connection issue."
    ),
    "APPS_MEDIA": (
        "We can help troubleshoot the app or media issue."
    ),
    "DEVICE_HARDWARE": (
        "We can help with the device issue."
    ),
    "FEATURE_HOW_TO": (
        "We can help you with that feature."
    ),
    "ACCOUNT_SECURITY": (
        "We can help with your Apple ID or account issue."
    ),
    "PURCHASE_REPAIR_WARRANTY": (
        "We can help with the purchase, repair, or warranty issue."
    ),
    "OTHER_UNCLEAR": (
        "We'd like to help with this issue."
    ),
}


def clean_historical_response(
    text: str,
) -> str:
    """
    Convert a historical public Twitter response into
    reusable evidence rather than copying Twitter-specific artifacts.
    """

    text = str(text).strip()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    # Remove Twitter mentions.
    text = re.sub(
        r"@\w+",
        "",
        text,
    )

    # Remove common Twitter-specific private-channel wording.
    text = re.sub(
        r"\bplease\s+send\s+us\s+a\s+dm\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\blet'?s\s+meet\s+(?:in|via)\s+dm\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip(" -:")


def choose_evidence(
    retrieved_cases: list[dict[str, Any]],
) -> tuple[str, float] | None:
    """
    Choose the highest-ranked clean historical response.
    """

    for case in retrieved_cases:

        raw_response = case.get(
            "agent_response",
            "",
        )

        cleaned = clean_historical_response(
            raw_response
        )

        score = float(
            case.get(
                "rerank_score",
                case.get(
                    "retrieval_score",
                    0.0,
                ),
            )
        )

        # Don't use useless one-word or ultra-short responses.
        if len(cleaned.split()) < 6:
            continue

        return cleaned, score

    return None


class ReplyGenerator:
    """
    Conservative offline reply generator.

    It uses historical AppleSupport responses as evidence but
    does not directly expose raw Twitter artifacts.
    """

    def generate(
        self,
        *,
        customer_message: str,
        intent: str,
        retrieved_cases: list[dict[str, Any]],
        should_escalate: bool,
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        evidence = choose_evidence(
            retrieved_cases
        )

        # --------------------------------------------------
        # No usable historical evidence.
        # --------------------------------------------------

        if evidence is None:

            return {
                "reply": (
                    "Thanks for reaching out. "
                    "We'd like to help. Please share a few "
                    "more details about what you're experiencing."
                ),
                "grounded": False,
                "evidence_count": 0,
                "evidence_score": 0.0,
            }

        historical_response, evidence_score = evidence

        # --------------------------------------------------
        # Escalated cases.
        # --------------------------------------------------

        if should_escalate:

            reply = (
                f"{INTENT_OPENERS.get(intent, 'We’d like to help')} "
                "Please provide the relevant details through "
                "a private support channel so the case can be "
                "reviewed securely."
            )

            return {
                "reply": reply,
                "grounded": True,
                "evidence_count": 1,
                "evidence_score": evidence_score,
            }

        # --------------------------------------------------
        # Automated cases.
        # --------------------------------------------------

        reply = (
            f"{INTENT_OPENERS.get(intent, 'We’d like to help')} "
            f"{historical_response}"
        )

        # Avoid excessively long replies.
        if len(reply) > 500:
            reply = reply[:497].rstrip() + "..."

        return {
            "reply": reply,
            "grounded": True,
            "evidence_count": 1,
            "evidence_score": evidence_score,
        }