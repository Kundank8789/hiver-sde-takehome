from __future__ import annotations

import json
from typing import Any

from llm_client import GroqClient


REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string"
        },
        "grounded": {
            "type": "boolean"
        },
        "evidence_used": {
            "type": "integer",
            "minimum": 0
        },
        "reason": {
            "type": "string"
        },
    },
    "required": [
        "reply",
        "grounded",
        "evidence_used",
        "reason",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
You are an AppleSupport customer-support reply writer.

Your task is to draft a concise customer-facing reply grounded ONLY
in the supplied historical AppleSupport evidence.

STRICT GROUNDING POLICY:

1. Every troubleshooting instruction must be explicitly supported by
   the supplied historical evidence.
2. Do NOT invent steps, settings paths, procedures, diagnostics,
   policies, refunds, warranties, guarantees, timelines, or outcomes.
3. Do NOT add generic troubleshooting advice that is not present in
   the evidence.
4. Do NOT expose Twitter usernames or raw URLs from historical examples.
5. Do NOT mention that you are an AI.
6. Do NOT claim the issue is fixed.
7. Do NOT claim that Apple will perform an action unless the evidence
   explicitly supports that action.
8. If the evidence does not contain enough information to provide a
   useful solution, ask a concise clarifying question instead.
9. If the case is marked for escalation, clearly direct the customer
   toward human/private support without inventing a process.
10. Keep the response concise, professional, empathetic, and natural.

GROUNDING PRIORITY:

- Historical evidence is the only source of factual troubleshooting
  guidance.
- The customer's message tells you what problem to address.
- The predicted intent tells you the support category.
- Do not use outside knowledge.

Return only the required structured output.
"""


class GroqReplyGenerator:

    def __init__(
        self,
        client: GroqClient | None = None,
    ) -> None:

        self.client = (
            client
            or GroqClient()
        )

    def generate(
        self,
        *,
        customer_message: str,
        intent: str,
        historical_cases: list[dict[str, Any]],
        should_escalate: bool,
        escalation_reason: str,
    ) -> dict[str, Any]:

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        evidence_blocks = []

        for index, case in enumerate(
            historical_cases[:5],
            start=1,
        ):

            customer = str(
                case.get(
                    "customer_message",
                    "",
                )
            ).strip()

            response = str(
                case.get(
                    "agent_response",
                    "",
                )
            ).strip()

            if not customer or not response:
                continue

            evidence_blocks.append(
                f"""
Historical case {index}

Customer:
{customer}

AppleSupport response:
{response}
""".strip()
            )

        evidence = (
            "\n\n".join(
                evidence_blocks
            )
            if evidence_blocks
            else "No usable historical evidence was retrieved."
        )

        user_prompt = f"""
Customer message:
{customer_message}

Predicted intent:
{intent}
Your response MUST be supported by the historical evidence below.

Historical AppleSupport evidence:

Should escalate:
{should_escalate}

Escalation reason:
{escalation_reason}

Historical AppleSupport evidence:
{evidence}

Write the shortest useful customer-facing reply that is supported
by that evidence. Do not add troubleshooting steps that are absent
from the evidence.
"""

        raw = self.client.structured(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            schema_name="apple_support_reply",
            schema=REPLY_SCHEMA,
            temperature=0.0,
        )

        result = json.loads(raw)

        reply = str(
            result.get(
                "reply",
                "",
            )
        ).strip()

        grounded = bool(
            result.get(
                "grounded",
                False,
            )
        )

        evidence_used = int(
            result.get(
                "evidence_used",
                0,
            )
        )

        reason = str(
            result.get(
                "reason",
                "",
            )
        ).strip()

        if not reply:
            raise ValueError(
                "Generated reply is empty."
            )

        if evidence_used < 0:
            raise ValueError(
                "evidence_used cannot be negative."
            )

        return {
            "reply": reply,
            "grounded": grounded,
            "evidence_used": evidence_used,
            "reason": reason,
        }