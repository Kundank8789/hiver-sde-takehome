from __future__ import annotations

from typing import Any

from escalation import decide_escalation
from groq_reply_generator import GroqReplyGenerator
from improve_intent_classifier import ImprovedIntentClassifier
from retriever import HistoricalRetriever


class SupportAgent:
    """
    End-to-end AppleSupport AI support agent.

    Pipeline:
        customer message
            -> intent classification
            -> escalation decision
            -> historical retrieval
            -> grounded reply generation
    """

    def __init__(
        self,
        classifier: ImprovedIntentClassifier | None = None,
        retriever: HistoricalRetriever | None = None,
        reply_generator: GroqReplyGenerator | None = None,
    ) -> None:

        self.classifier = (
            classifier
            or ImprovedIntentClassifier()
        )

        self.retriever = (
            retriever
            or HistoricalRetriever()
        )

        self.reply_generator = (
            reply_generator
            or GroqReplyGenerator()
        )

    def handle(
        self,
        *,
        customer_message: str,
        context: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        # --------------------------------------------------
        # 1. Intent classification
        # --------------------------------------------------

        classification = self.classifier.classify(
            customer_message=customer_message,
            context=context,
        )

        intent = str(
            classification["intent"]
        )

        confidence = float(
            classification["confidence"]
        )

        classification_reason = str(
            classification["reason"]
        )

        # --------------------------------------------------
        # 2. Deterministic escalation
        # --------------------------------------------------

        escalation = decide_escalation(
            intent=intent,
            confidence=confidence,
            customer_message=customer_message,
        )

        # --------------------------------------------------
        # 3. Historical retrieval
        # --------------------------------------------------

        retrieved_cases = self.retriever.retrieve(
            query=customer_message,
            intent=intent,
            candidate_k=20,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 4. Reply generation
        # --------------------------------------------------

        if escalation.should_escalate:

            # Don't waste an additional LLM call for cases
            # that must go to a human.
            reply = (
                "We'd like to help with your issue. "
                "Please provide the relevant details through "
                "a private support channel so the case can be "
                "reviewed securely."
            )

            grounded = bool(
                len(retrieved_cases) > 0
            )

            evidence_used = min(
                len(retrieved_cases),
                1,
            )

            evidence_score = (
                float(
                    retrieved_cases[0].get(
                        "rerank_score",
                        retrieved_cases[0].get(
                            "retrieval_score",
                            0.0,
                        ),
                    )
                )
                if retrieved_cases
                else 0.0
            )

            generation_reason = (
                "Deterministic escalation response used; "
                "no generative reply call was required."
            )

        else:

            reply_result = (
                self.reply_generator.generate(
                    customer_message=customer_message,
                    intent=intent,
                    historical_cases=retrieved_cases,
                    should_escalate=False,
                    escalation_reason=escalation.reason,
                )
            )

            reply = str(
                reply_result["reply"]
            )

            grounded = bool(
                reply_result["grounded"]
            )

            evidence_used = int(
                reply_result["evidence_used"]
            )

            evidence_score = (
                float(
                    retrieved_cases[0].get(
                        "rerank_score",
                        retrieved_cases[0].get(
                            "retrieval_score",
                            0.0,
                        ),
                    )
                )
                if retrieved_cases
                else 0.0
            )

            generation_reason = str(
                reply_result.get(
                    "reason",
                    "",
                )
            )

        # --------------------------------------------------
        # 5. Stable public output contract
        # --------------------------------------------------

        return {
            "intent": intent,
            "confidence": confidence,
            "classification_reason": classification_reason,
            "should_escalate": (
                escalation.should_escalate
            ),
            "escalation_reason": (
                escalation.reason
            ),
            "reply": reply,
            "grounded": grounded,
            "evidence_used": evidence_used,
            "evidence_score": evidence_score,
            "generation_reason": generation_reason,
            "retrieved_cases": retrieved_cases,
        }