from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqClient:
    """
    Groq wrapper with bounded retries.

    Retry policy:
    - Temporary 429/network failures: retry.
    - Daily/token quota errors: fail immediately.
    - Other API/schema errors: fail immediately.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
        base_delay: float = 1.0,
    ) -> None:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    @staticmethod
    def _is_daily_quota_error(
        error: Exception,
    ) -> bool:

        message = str(
            error
        ).lower()

        return (
            "tokens per day" in message
            or "tpd" in message
            or "tokens per day" in message
        )

    @staticmethod
    def _is_retryable_error(
        error: Exception,
    ) -> bool:

        message = str(
            error
        ).lower()

        return (
            "rate limit" in message
            and not GroqClient._is_daily_quota_error(
                error
            )
        ) or any(
            phrase in message
            for phrase in (
                "timed out",
                "timeout",
                "connection",
                "temporarily unavailable",
            )
        )

    def structured(
        self,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> str:

        last_error: Exception | None = None

        for attempt in range(
            self.max_retries + 1
        ):

            try:

                response = (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                )

                if not response.choices:
                    raise RuntimeError(
                        "Groq returned no choices."
                    )

                content = (
                    response.choices[0]
                    .message
                    .content
                )

                if not content:
                    raise RuntimeError(
                        "Groq returned empty content."
                    )

                return content

            except Exception as exc:

                last_error = exc

                # Never waste retries on daily quota exhaustion.
                if self._is_daily_quota_error(
                    exc
                ):
                    raise

                if (
                    attempt >= self.max_retries
                    or not self._is_retryable_error(
                        exc
                    )
                ):
                    raise

                delay = (
                    self.base_delay
                    * (2 ** attempt)
                )

                time.sleep(
                    delay
                )

        raise RuntimeError(
            "Groq request failed."
        ) from last_error