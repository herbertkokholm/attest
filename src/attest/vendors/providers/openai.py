"""OpenAI adapter for the `Rater` protocol.

Requires the `openai` package, gated behind the `attest[openai]` extra. The
SDK is imported lazily inside `_client`, so this module itself imports
cleanly without the extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import DEFAULT_SCREENING_PROMPT, parse_ordinal_response


@dataclass
class OpenAIRater:
    """Rates records with an OpenAI model via the Chat Completions API.

    Attributes:
        model: OpenAI model identifier (e.g. "gpt-4o").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``OPENAI_API_KEY``) when None.
        prompt: System prompt instructing the model how to rate a record.
        max_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str = DEFAULT_SCREENING_PROMPT
    max_tokens: int = 8
    vendor: str = field(default="openai", init=False)

    def _client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "the 'openai' package is required to use OpenAIRater; "
                "install it with: pip install 'attest[openai]'"
            ) from exc
        return openai.OpenAI(api_key=self.api_key)

    def rate(self, record: Record) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the OpenAI Chat Completions API.

        Args:
            record: The record to rate.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                },
            ],
        )
        text = response.choices[0].message.content or ""
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {"text": text, "id": getattr(response, "id", None)}
        return ordinal, raw_response
