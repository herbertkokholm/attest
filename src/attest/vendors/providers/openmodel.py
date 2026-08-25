"""Adapter for self-hosted, open-weight models behind an OpenAI-compatible HTTP API.

Servers such as vLLM, llama.cpp, or Ollama's OpenAI-compatible endpoint speak
plain HTTP, so -- unlike the other provider adapters -- this one needs no
vendor SDK and no optional extra: it uses only the standard library.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import (
    check_model_version,
    compose_batch_system_prompt,
    compose_batch_user_message,
    compose_system_prompt,
    parse_batch_response,
    parse_ordinal_response,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"


@dataclass
class OpenModelRater:
    """Rates records via an OpenAI-compatible `/chat/completions` endpoint.

    Attributes:
        model: Model identifier as known to the serving endpoint.
        model_version: Expected resolved model version. The OpenAI-compatible
            schema echoes the serving model back in the response's `model`
            field, so this is checked against it after every call, the same
            way `attest.vendors.providers.openai.OpenAIRater` checks a live
            OpenAI response (see `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature included in the request payload,
            unless `send_temperature` is `False`.
        send_temperature: When `False`, `temperature` is omitted from the
            request body entirely instead of being sent. Some reasoning
            models behind an OpenAI-compatible endpoint reject an explicit
            non-default `temperature` outright, the same way current-generation
            Claude and some OpenAI models do -- see
            `attest.vendors.providers.anthropic.AnthropicRater.send_temperature`
            and `attest.provenance.config.VendorSpec.send_temperature`.
        base_url: Base URL of the OpenAI-compatible server, without the
            trailing `/chat/completions` path.
        api_key: Optional bearer token for the endpoint, if it requires one.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply. Reasoning models
            behind an OpenAI-compatible endpoint spend part of this budget on
            a hidden reasoning phase (echoed back, if at all, in a
            `reasoning_content` field alongside `content`) before ever
            emitting the actual answer, so a low value tuned for a
            non-reasoning model can silently starve `content` to empty --
            confirmed against Alexandra Institute's `qwen3.5-397b` endpoint,
            2026-08-25, where `content` came back `""` at the previous
            default of 8.
        reasoning_effort: Forwarded as the request body's `reasoning_effort`
            field when not `None`; omitted entirely otherwise, reproducing
            prior behavior exactly. On the one OpenAI-compatible reasoning
            model this has been checked against (`qwen3.5-397b` via
            Alexandra Institute, 2026-08-25), `"none"` disables the hidden
            reasoning phase outright (cutting completion tokens for a
            single-letter answer from roughly 950 to 2), while `"low"`,
            `"medium"`, and `"high"` were all indistinguishable from each
            other and from omitting the field -- i.e. this endpoint exposes
            no graduated effort control, only on/off. Whether any given
            OpenAI-compatible server honors this key at all, and whether it
            offers real graduation, is a per-deployment empirical question --
            see `attest.provenance.config.VendorSpec.reasoning_effort`.
        timeout: Request timeout in seconds.
    """

    model: str
    model_version: str
    temperature: float
    send_temperature: bool = True
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    reasoning_effort: str | None = None
    timeout: float = 30.0
    vendor: str = field(default="openmodel", init=False)

    def _temperature_kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature} if self.send_temperature else {}

    def _reasoning_effort_kwargs(self) -> dict[str, Any]:
        return {} if self.reasoning_effort is None else {"reasoning_effort": self.reasoning_effort}

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by POSTing a chat completion request to `base_url`.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw decoded JSON response.

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
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
            **self._temperature_kwargs(),
            **self._reasoning_effort_kwargs(),
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
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=body.get("model"),
        )
        text = body["choices"][0]["message"]["content"]
        ordinal = parse_ordinal_response(text)
        return ordinal, body

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Rate every record in `records` together, in one POST to `base_url`.

        Packs all of `records` into a single request: the system message
        uses `compose_batch_system_prompt` (a per-id JSON output contract)
        instead of `compose_system_prompt`'s single letter, and the user
        message lists every record by id (`compose_batch_user_message`).
        Parsed by id via `parse_batch_response`, order-independent.

        Args:
            records: The records to rate together.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            One `(ordinal, raw_response)` pair per record in `records`, in order.

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
            VendorResponseError: If some record has no parseable rating in
                the response.
        """
        record_ids = [record.id for record in records]
        payload = {
            "model": self.model,
            "max_tokens": max(self.max_tokens, 16 * len(records)),
            "messages": [
                {
                    "role": "system",
                    "content": compose_batch_system_prompt(
                        prompt if prompt is not None else self.prompt
                    ),
                },
                {"role": "user", "content": compose_batch_user_message(records)},
            ],
            **self._temperature_kwargs(),
            **self._reasoning_effort_kwargs(),
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
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=body.get("model"),
        )
        text = body["choices"][0]["message"]["content"]
        ratings = parse_batch_response(text, record_ids)
        return [(ratings[record_id], {**body, "record_id": record_id}) for record_id in record_ids]
