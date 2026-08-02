"""Anthropic Claude adapter for the `Rater` protocol.

Requires the `anthropic` package, gated behind the `attest[anthropic]`
extra. The SDK is imported lazily inside `_client`, so this module itself
imports cleanly without the extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import DEFAULT_SCREENING_PROMPT, parse_ordinal_response


@dataclass
class AnthropicRater:
    """Rates records with an Anthropic Claude model via the Messages API.

    Attributes:
        model: Anthropic model identifier (e.g. "claude-sonnet-5").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``ANTHROPIC_API_KEY``) when None.
        prompt: System prompt instructing the model how to rate a record.
        max_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str = DEFAULT_SCREENING_PROMPT
    max_tokens: int = 8
    vendor: str = field(default="anthropic", init=False)

    def _client(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "the 'anthropic' package is required to use AnthropicRater; "
                "install it with: pip install 'attest[anthropic]'"
            ) from exc
        return anthropic.Anthropic(api_key=self.api_key)

    def rate(self, record: Record) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Anthropic Messages API.

        Args:
            record: The record to rate.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                }
            ],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {"text": text, "id": getattr(response, "id", None)}
        return ordinal, raw_response
