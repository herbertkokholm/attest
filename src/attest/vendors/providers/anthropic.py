"""Anthropic Claude adapter for the `Rater` and `BatchRater` protocols.

Requires the `anthropic` package, gated behind the `attest[anthropic]`
extra. The SDK is imported lazily inside `_client`, so this module itself
imports cleanly without the extra installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import (
    DEFAULT_SCREENING_PROMPT,
    VendorResponseError,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


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

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Anthropic Messages API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompt if prompt is not None else self.prompt,
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


@dataclass
class AnthropicBatchRater:
    """Rates records with an Anthropic Claude model via the Message Batches API.

    Issues the same system prompt and user message, and parses results with
    the same `parse_ordinal_response`, as `AnthropicRater`, so sync and batch
    execution yield the same ratings for the same inputs.

    Attributes:
        model: Anthropic model identifier (e.g. "claude-sonnet-5").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``ANTHROPIC_API_KEY``) when None.
        prompt: System prompt instructing the model how to rate a record.
        max_tokens: Maximum tokens to request in each reply.
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
                "the 'anthropic' package is required to use AnthropicBatchRater; "
                "install it with: pip install 'attest[anthropic]'"
            ) from exc
        return anthropic.Anthropic(api_key=self.api_key)

    def _request(self, record: Record, custom_id: str, prompt: str) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                    }
                ],
            },
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Submit `records` as one Anthropic Message Batch.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.

        Returns:
            A `BatchHandle` identifying the submitted batch.
        """
        prompts = prompts or {}
        id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
        requests = [
            self._request(record, id_map[record.id], prompts.get(record.id, self.prompt))
            for record in records
        ]
        batch = self._client().messages.batches.create(requests=requests)
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=batch.id,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the Message Batch's `processing_status`."""
        batch = self._client().messages.batches.retrieve(handle.provider_batch_id)
        if batch.processing_status == "ended":
            return "completed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Retrieve and parse this batch's per-record results.

        A record whose result did not succeed, or whose text does not parse
        as an ordinal rating, is simply absent from the returned mapping.
        """
        reverse = {custom_id: record_id for record_id, custom_id in handle.id_map.items()}
        results: dict[str, tuple[int, Any]] = {}
        for entry in self._client().messages.batches.results(handle.provider_batch_id):
            record_id = reverse.get(entry.custom_id)
            if record_id is None or entry.result.type != "succeeded":
                continue
            text = "".join(
                block.text
                for block in entry.result.message.content
                if getattr(block, "type", "") == "text"
            )
            try:
                ordinal = parse_ordinal_response(text)
            except VendorResponseError:
                continue
            results[record_id] = (ordinal, {"text": text, "custom_id": entry.custom_id})
        return results
