from __future__ import annotations

import re
from collections import Counter
from typing import Any

from retriever import HistoricalRetriever


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


HIGH_PRECISION_PATTERNS = {
    "ACCOUNT_SECURITY": [
        r"\bapple id\b.*\b(password|locked|disabled|verify|sign in|login)\b",
        r"\b(forgot|forgotten)\b.*\bpassword\b",
        r"\baccount\b.*\b(locked|disabled|verification)\b",
    ],

    "BATTERY_POWER": [
        r"\bbattery\b.*\b(drain|draining|charge|charging|life)\b",
        r"\b(drain|draining)\b.*\bbattery\b",
        r"\bwon'?t charge\b",
        r"\bnot charging\b",
        r"\bhold a charge\b",
        r"\bbattery\b.*\b(overheat|smoke|fire)\b",
    ],

    "CONNECTIVITY": [
        r"\bbluetooth\b.*\b(connect|connecting|pair)\b",
        r"\b(connect|connecting|connection)\b.*\b(bluetooth|wifi|wi-fi|cellular|network)\b",
        r"\b(wifi|wi-fi)\b.*\b(won'?t|cannot|can'?t|unable)\b",
        r"\bno service\b",
        r"\bcellular\b.*\b(won'?t|cannot|can'?t|unable)\b",
        r"\bnetwork\b.*\b(won'?t|cannot|can'?t|unable)\b",
    ],

    "PERFORMANCE_STABILITY": [
        r"\bfreez(es|ing)?\b",
        r"\bcrash(es|ing)?\b",
        r"\brestart(s|ing)?\b",
        r"\breboot(s|ing)?\b",
        r"\blag(ging)?\b",
        r"\bslow\b",
        r"\bslowness\b",
        r"\bstuck\b",
        r"\bunresponsive\b",
    ],

    "IOS_UPDATE": [
        r"\binstall\b.*\b(ios|update)\b",
        r"\b(ios|software)\b.*\b(update|upgrade)\b.*\b(fail|failed|stuck|error|install)\b",
        r"\bcan'?t\b.*\binstall\b.*\bupdate\b",
        r"\bupdate\b.*\bverification\b",
        r"\bsoftware update\b.*\b(error|failed|stuck)\b",
        r"\bupdate\b.*\bwon'?t install\b",
    ],

    "DEVICE_HARDWARE": [
        r"\b(cracked|broken|damaged)\b.*\b(screen|display)\b",
        r"\b(screen|display)\b.*\b(cracked|broken|damaged)\b",
        r"\bwon'?t turn on\b",
        r"\bdoesn'?t turn on\b",
        r"\bphysical damage\b",
        r"\b(camera|speaker|button)\b.*\b(broken|damaged|not working)\b",
    ],

    "PURCHASE_REPAIR_WARRANTY": [
        r"\b(warranty|repair|replacement|refund|order|purchase)\b",
        r"\bservice appointment\b",
        r"\breplace\b.*\biphone\b",
    ],

    "FEATURE_HOW_TO": [
        r"^\s*how do i\b",
        r"^\s*how can i\b",
        r"^\s*how to\b",
        r"\bhow do i\b.*\b(turn off|enable|disable|use|change|set)\b",
        r"\bhow can i\b.*\b(turn off|enable|disable|use|change|set)\b",
        r"\bwhere can i\b.*\b(setting|feature)\b",
        r"\bhow does\b.*\bwork\b",
    ],

    "APPS_MEDIA": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bfacetime\b",
        r"\bimessage\b",
        r"\bphotos app\b",
        r"\bapp store\b",
        r"\byoutube\b",
        r"\bimovie\b",
    ],
}


def normalize(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
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


def rule_scores(
    text: str,
) -> dict[str, int]:

    text = normalize(text)

    scores = {
        intent: 0
        for intent in INTENTS
    }

    for intent, patterns in HIGH_PRECISION_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text,
            ):
                scores[intent] += 1

    return scores


def infer_retrieval_intent(
    results: list[dict[str, Any]],
) -> tuple[str, float] | None:

    intent_scores = Counter()

    usable_results = 0

    for result in results:

        metadata_intent = result.get(
            "intent"
        )

        if metadata_intent in INTENTS:

            score = float(
                result.get(
                    "rerank_score",
                    result.get(
                        "retrieval_score",
                        0.0,
                    ),
                )
            )

            if score <= 0:
                continue

            intent_scores[
                metadata_intent
            ] += score

            usable_results += 1

    if not intent_scores:
        return None

    intent, total_score = (
        intent_scores.most_common(1)[0]
    )

    if usable_results < 2:
        return None

    normalized_score = min(
        0.85,
        0.45 + (
            total_score / usable_results
        ),
    )

    return intent, normalized_score


class HybridIntentClassifier:
    """
    Hybrid classifier using:
      1. high-precision deterministic rules
      2. historical retrieval
      3. OTHER_UNCLEAR only when both are insufficient

    The LLM fallback is optional.
    """

    def __init__(
        self,
        llm_classifier: Any | None = None,
        retriever: HistoricalRetriever | None = None,
        rule_threshold: int = 2,
        retrieval_threshold: float = 0.25,
    ) -> None:

        self.llm = llm_classifier

        self.retriever = (
            retriever
            or HistoricalRetriever()
        )

        self.rule_threshold = (
            rule_threshold
        )

        self.retrieval_threshold = (
            retrieval_threshold
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

        message = normalize(
            customer_message
        )

        scores = rule_scores(
            message
        )

        best_intent = max(
            scores,
            key=scores.get,
        )

        best_rule_score = scores[
            best_intent
        ]

        # --------------------------------------------------
        # 1. High-confidence rules
        # --------------------------------------------------

        if best_rule_score >= self.rule_threshold:

            return {
                "intent": best_intent,
                "confidence": min(
                    0.95,
                    0.82
                    + (
                        0.05
                        * best_rule_score
                    ),
                ),
                "reason": (
                    "High-precision support rules "
                    "identified the primary intent."
                ),
                "method": "rules",
            }

        # --------------------------------------------------
        # 2. Strong single-signal rules
        # --------------------------------------------------

        active_intents = [
            intent
            for intent, score
            in scores.items()
            if score > 0
        ]

        if len(active_intents) == 1:

            intent = active_intents[0]

            if intent in {
                "ACCOUNT_SECURITY",
                "BATTERY_POWER",
                "CONNECTIVITY",
                "IOS_UPDATE",
                "PURCHASE_REPAIR_WARRANTY",
                "FEATURE_HOW_TO",
            }:

                return {
                    "intent": intent,
                    "confidence": 0.82,
                    "reason": (
                        "A high-information support signal "
                        "identified the primary intent."
                    ),
                    "method": "rules",
                }

        # --------------------------------------------------
        # 3. Retrieval-based classification
        # --------------------------------------------------

        results = self.retriever.retrieve(
            query=customer_message,
            intent=(
                best_intent
                if best_rule_score > 0
                else None
            ),
            candidate_k=30,
            top_k=8,
            min_score=self.retrieval_threshold,
        )

        retrieval_prediction = (
            infer_retrieval_intent(
                results
            )
        )

        if retrieval_prediction is not None:

            intent, confidence = (
                retrieval_prediction
            )

            return {
                "intent": intent,
                "confidence": confidence,
                "reason": (
                    "Intent inferred from multiple "
                    "historically similar AppleSupport cases."
                ),
                "method": "historical_retrieval",
            }

        # --------------------------------------------------
        # 4. Optional LLM fallback
        # --------------------------------------------------

        if self.llm is not None:

            result = self.llm.classify(
                customer_message=customer_message,
                context=context,
            )

            return {
                **result,
                "method": "llm_fallback",
            }

        # --------------------------------------------------
        # 5. Conservative fallback
        # --------------------------------------------------

        return {
            "intent": (
                best_intent
                if best_rule_score > 0
                else "OTHER_UNCLEAR"
            ),
            "confidence": (
                0.50
                if best_rule_score > 0
                else 0.0
            ),
            "reason": (
                "No sufficiently reliable rule or "
                "historical match was available."
            ),
            "method": "fallback",
        }