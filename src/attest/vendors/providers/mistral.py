"""Mistral adapter for the `Rater` and `BatchRater` protocols.

Requires the `mistralai` package, gated behind the `attest[mistral]` extra.
The SDK is imported lazily inside `_client`, so this module itself imports
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
    DEFAULT_SCREENING_PROMPT,
    VendorResponseError,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


@dataclass
class MistralRater:
    """Rates records with a Mistral model via the (la Plateforme) chat API.

    Attributes:
        model: Mistral model identifier (e.g. "mistral-small-latest").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``MISTRAL_API_KEY``) when None.
        prompt: System prompt instructing the model how to rate a record.
        max_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str = DEFAULT_SCREENING_PROMPT
    max_tokens: int = 8
    vendor: str = field(default="mistral", init=False)

    def _client(self) -> Any:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ImportError(
                "the 'mistralai' package is required to use MistralRater; "
                "install it with: pip install 'attest[mistral]'"
            ) from exc
        return Mistral(api_key=self.api_key)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Mistral chat completion API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        response = self._client().chat.complete(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": prompt if prompt is not None else self.prompt},
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
class MistralBatchRater:
    """Rates records with a Mistral model via the la Plateforme Batch API.

    Issues the same messages, and parses results with the same
    `parse_ordinal_response`, as `MistralRater`, so sync and batch execution
    yield the same ratings for the same inputs.

    Mistral's batch surface mirrors OpenAI's (upload a JSONL request file,
    create a batch job against it, poll `status`, download a JSONL output
    file) and -- like the other batch adapters -- is not exercised against a
    live API in this codebase's tests, only
    `attest.vendors.batch.DeterministicBatchRater` is. Adjust
    `_client`/`submit`/`poll`/`fetch` if an installed SDK version has renamed
    these methods.

    Attributes:
        model: Mistral model identifier (e.g. "mistral-small-latest").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``MISTRAL_API_KEY``) when None.
        prompt: System prompt instructing the model how to rate a record.
        max_tokens: Maximum tokens to request in each reply.
    """

    model: str
    api_key: str | None = None
    prompt: str = DEFAULT_SCREENING_PROMPT
    max_tokens: int = 8
    vendor: str = field(default="mistral", init=False)

    def _client(self) -> Any:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ImportError(
                "the 'mistralai' package is required to use MistralBatchRater; "
                "install it with: pip install 'attest[mistral]'"
            ) from exc
        return Mistral(api_key=self.api_key)

    def _request_line(self, record: Record, custom_id: str, prompt: str) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "body": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": prompt},
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
        """Upload a JSONL request file and submit it as one Mistral batch job.

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
        upload = client.files.upload(
            file={"file_name": "batch.jsonl", "content": io.BytesIO(lines.encode("utf-8"))},
            purpose="batch",
        )
        job = client.batch.jobs.create(
            input_files=[upload.id],
            model=self.model,
            endpoint="/v1/chat/completions",
        )
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=job.id,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the batch job's `status`."""
        job = self._client().batch.jobs.get(job_id=handle.provider_batch_id)
        if job.status == "SUCCESS":
            return "completed"
        if job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Download and parse the batch job's output JSONL.

        A line whose request errored, or whose text does not parse as an
        ordinal rating, is simply absent from the returned mapping.
        """
        reverse = {custom_id: record_id for record_id, custom_id in handle.id_map.items()}
        client = self._client()
        job = client.batch.jobs.get(job_id=handle.provider_batch_id)
        if job.output_file is None:
            return {}
        content = client.files.download(file_id=job.output_file).read().decode("utf-8")
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
