"""OpenAI adapter for the `Rater` and `BatchRater` protocols.

Requires the `openai` package, gated behind the `attest[openai]` extra. The
SDK is imported lazily inside `_client`, so this module itself imports
cleanly without the extra installed.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from attest.contracts.input import Record
from attest.vendors.base import (
    VendorResponseError,
    compose_system_prompt,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


@dataclass
class OpenAIRater:
    """Rates records with an OpenAI model via the Chat Completions API.

    Attributes:
        model: OpenAI model identifier (e.g. "gpt-4o").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``OPENAI_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str | None = None
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

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the OpenAI Chat Completions API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": compose_system_prompt(prompt if prompt is not None else self.prompt),
                },
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


@dataclass
class OpenAIBatchRater:
    """Rates records with an OpenAI model via the Batch API.

    Issues the same messages, and parses results with the same
    `parse_ordinal_response`, as `OpenAIRater`, so sync and batch execution
    yield the same ratings for the same inputs.

    Attributes:
        model: OpenAI model identifier (e.g. "gpt-4o").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``OPENAI_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.
        completion_window: Vendor-side completion SLA for the batch job.
    """

    model: str
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    completion_window: str = "24h"
    vendor: str = field(default="openai", init=False)

    def _client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "the 'openai' package is required to use OpenAIBatchRater; "
                "install it with: pip install 'attest[openai]'"
            ) from exc
        return openai.OpenAI(api_key=self.api_key)

    def _request_line(self, record: Record, custom_id: str, prompt: str | None) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": compose_system_prompt(prompt)},
                    {
                        "role": "user",
                        "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                    },
                ],
            },
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Upload a JSONL request file and submit it as one OpenAI Batch job.

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
        lines = "\n".join(
            json.dumps(
                self._request_line(record, id_map[record.id], prompts.get(record.id, self.prompt))
            )
            for record in records
        )
        client = self._client()
        upload = client.files.create(file=io.BytesIO(lines.encode("utf-8")), purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window,
        )
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=batch.id,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the Batch job's `status`."""
        batch = self._client().batches.retrieve(handle.provider_batch_id)
        if batch.status == "completed":
            return "completed"
        if batch.status in ("failed", "expired", "cancelled"):
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Download and parse the Batch job's output JSONL.

        A line whose request errored, or whose text does not parse as an
        ordinal rating, is simply absent from the returned mapping.
        """
        reverse = {custom_id: record_id for record_id, custom_id in handle.id_map.items()}
        client = self._client()
        batch = client.batches.retrieve(handle.provider_batch_id)
        if batch.output_file_id is None:
            return {}
        content = client.files.content(batch.output_file_id).text
        results: dict[str, tuple[int, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            record_id = reverse.get(entry.get("custom_id"))
            if record_id is None or entry.get("error"):
                continue
            text = entry["response"]["body"]["choices"][0]["message"]["content"] or ""
            try:
                ordinal = parse_ordinal_response(text)
            except VendorResponseError:
                continue
            results[record_id] = (ordinal, entry)
        return results
