"""Google Gemini adapter for the `Rater` protocol.

Requires the `google-generativeai` package, gated behind the
`attest[google]` extra. The SDK is imported lazily inside `_client`, so this
module itself imports cleanly without the extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import DEFAULT_SCREENING_PROMPT, parse_ordinal_response


@dataclass
class GoogleRater:
    """Rates records with a Google Gemini model via `google-generativeai`.

    Attributes:
        model: Gemini model identifier (e.g. "gemini-1.5-pro").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``GOOGLE_API_KEY``) when None.
        prompt: System instruction telling the model how to rate a record.
        max_output_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str = DEFAULT_SCREENING_PROMPT
    max_output_tokens: int = 8
    vendor: str = field(default="google", init=False)

    def _client(self) -> Any:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "the 'google-generativeai' package is required to use GoogleRater; "
                "install it with: pip install 'attest[google]'"
            ) from exc
        if self.api_key is not None:
            genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(model_name=self.model, system_instruction=self.prompt)

    def rate(self, record: Record) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Gemini `generate_content` API.

        Args:
            record: The record to rate.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().generate_content(
            f"Title: {record.title}\nAbstract: {record.abstract}",
            generation_config={"max_output_tokens": self.max_output_tokens},
        )
        text = response.text
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {"text": text}
        return ordinal, raw_response
