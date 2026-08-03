"""Adapter for self-hosted, open-weight models behind an OpenAI-compatible HTTP API.

Servers such as vLLM, llama.cpp, or Ollama's OpenAI-compatible endpoint speak
plain HTTP, so -- unlike the other provider adapters -- this one needs no
vendor SDK and no optional extra: it uses only the standard library.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import compose_system_prompt, parse_ordinal_response

DEFAULT_BASE_URL = "http://localhost:8000/v1"


@dataclass
class OpenModelRater:
    """Rates records via an OpenAI-compatible `/chat/completions` endpoint.

    Attributes:
        model: Model identifier as known to the serving endpoint.
        base_url: Base URL of the OpenAI-compatible server, without the
            trailing `/chat/completions` path.
        api_key: Optional bearer token for the endpoint, if it requires one.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
        timeout: Request timeout in seconds.
    """

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    timeout: float = 30.0
    vendor: str = field(default="openmodel", init=False)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by POSTing a chat completion request to `base_url`.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw decoded JSON response.
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": compose_system_prompt(prompt if prompt is not None else self.prompt),
                },
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        ordinal = parse_ordinal_response(text)
        return ordinal, body
