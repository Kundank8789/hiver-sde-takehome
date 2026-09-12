from __future__ import annotations

import json
from typing import Any

from llm_client import GroqClient


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "groundedness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "relevance": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "helpfulness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "style": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "unsupported_claim": {
            "type": "boolean",
        },
        "overall": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "correctness",
        "groundedness",
        "relevance",
        "helpfulness",
        "style",
        "unsupported_claim",
        "overall",
        "reason",
    ],
    "additionalProperties": False,
}


JUDGE_PROMPT = """You are evaluating an AppleSupport AI reply.

Evaluate the drafted reply against:
1. The customer's request.
2. The historical AppleSupport evidence provided.
3. The escalation decision, if applicable.

Scoring:
5 = excellent
4 = good
3 = acceptable
2 = weak
1 = poor

Correctness:
Does the response address the customer's actual problem?

Groundedness:
Is the response supported by the historical evidence?
Do not reward invented troubleshooting claims.

Relevance:
Does it directly address the customer's request?

Helpfulness:
Would the reply move the customer toward resolution?

Style:
Is it concise, professional, empathetic, and appropriate for
customer support?

unsupported_claim:
True when the reply introduces a material fact, policy,
troubleshooting step, promise, or instruction that is not supported
by the supplied evidence.

Overall:
Overall customer-support quality.

Return only the required structured output.
"""


class ResponseJudge:

    def __init__(
        self,
        client: GroqClient | None = None,
    ) -> None:

        self.client = (
            client
            or GroqClient()
        )

    def judge(
        self,
        *,
        customer_message: str,
        historical_evidence: str,
        draft_reply: str,
        should_escalate: bool,
    ) -> dict[str, Any]:

        user_content = f"""
Customer message:
{customer_message}

Historical AppleSupport evidence:
{historical_evidence}

Should escalate:
{should_escalate}

Draft reply:
{draft_reply}
"""

        raw = self.client.structured(
            messages=[
                {
                    "role": "system",
                    "content": JUDGE_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            schema_name="apple_support_reply_judgment",
            schema=JUDGE_SCHEMA,
            temperature=0.0,
        )

        return json.loads(raw)