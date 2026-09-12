from __future__ import annotations

from typing import Any

from escalation import decide_escalation
from reply_generator import ReplyGenerator
from retriever import HistoricalRetriever


class OfflineSupportAgent:
    """
    Offline end-to-end AppleSupport support agent.

    Intended for deterministic pipeline testing before
    enabling LLM calls.
    """

    def __init__(
        self,
        retriever: HistoricalRetriever | None = None,
        reply_generator: ReplyGenerator | None = None,
    ) -> None:

        self.retriever = (
            retriever
            or HistoricalRetriever()
        )

        self.reply_generator = (
            reply_generator
            or ReplyGenerator()
        )

    def handle(
        self,
        *,
        customer_message: str,
        intent: str,
        confidence: float,
        top_k: int = 5,
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        escalation = decide_escalation(
            intent=intent,
            confidence=confidence,
            customer_message=customer_message,
        )

        retrieved = self.retriever.retrieve(
            query=customer_message,
            intent=intent,
            candidate_k=20,
            top_k=top_k,
        )

        reply = self.reply_generator.generate(
            customer_message=customer_message,
            intent=intent,
            retrieved_cases=retrieved,
            should_escalate=escalation.should_escalate,
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "should_escalate": (
                escalation.should_escalate
            ),
            "escalation_reason": (
                escalation.reason
            ),
            "reply": reply["reply"],
            "grounded": reply["grounded"],
            "evidence_count": reply["evidence_count"],
            "retrieved_cases": retrieved,
        }