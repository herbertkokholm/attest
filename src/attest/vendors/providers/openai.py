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
    check_model_version,
    chunk_records,
    compose_batch_system_prompt,
    compose_batch_user_message,
    compose_system_prompt,
    parse_batch_response,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


@dataclass
class OpenAIRater:
    """Rates records with an OpenAI model via the Chat Completions API.

    Attributes:
        model: OpenAI model identifier (e.g. "gpt-4o").
        model_version: Expected resolved model version. The Chat Completions
            API has no separate version parameter -- `model` is already what
            is sent -- so this is checked against the version the response
            itself reports (`response.model`) after every call, catching a
            floating alias silently resolving to a snapshot other than the
            one this configuration was hashed under (see
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Chat Completions API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``OPENAI_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
        request_logprobs: If True, request per-token log probabilities
            (`logprobs=True, top_logprobs=top_logprobs`) and retain the
            vendor's own, un-normalized logprob structure in the raw
            response under `"logprobs"`. This does not change what the
            model samples -- only what side-channel metadata comes back --
            so unlike `temperature`/`model_version` it is deliberately not a
            hash-versioned `VendorSpec` field; toggling it never opens a new
            ensemble-config epoch. Actual support is unverified without
            running `tools/vendor_logprob_probe.py` against the configured
            model.
        top_logprobs: Number of top alternative tokens to request logprobs
            for at each position, when `request_logprobs` is True.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    request_logprobs: bool = False
    top_logprobs: int = 5
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

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
        """
        logprobs_kwargs: dict[str, Any] = (
            {"logprobs": True, "top_logprobs": self.top_logprobs} if self.request_logprobs else {}
        )
        response = self._client().chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
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
            **logprobs_kwargs,
        )
        reported_version = getattr(response, "model", None)
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=reported_version,
        )
        text = response.choices[0].message.content or ""
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {
            "text": text,
            "id": getattr(response, "id", None),
            "model": reported_version,
        }
        response_logprobs = getattr(response.choices[0], "logprobs", None)
        if self.request_logprobs and response_logprobs is not None:
            raw_response["logprobs"] = response_logprobs.model_dump()
        return ordinal, raw_response

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Rate every record in `records` together, in one Chat Completions call.

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
        response = self._client().chat.completions.create(
            model=self.model,
            max_completion_tokens=max(self.max_tokens, 16 * len(records)),
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": compose_batch_system_prompt(
                        prompt if prompt is not None else self.prompt
                    ),
                },
                {"role": "user", "content": compose_batch_user_message(records)},
            ],
        )
        reported_version = getattr(response, "model", None)
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=reported_version,
        )
        text = response.choices[0].message.content or ""
        ratings = parse_batch_response(text, record_ids)
        raw_response: dict[str, Any] = {
            "text": text,
            "id": getattr(response, "id", None),
            "model": reported_version,
        }
        return [
            (ratings[record_id], {**raw_response, "record_id": record_id})
            for record_id in record_ids
        ]


@dataclass
class OpenAIBatchRater:
    """Rates records with an OpenAI model via the Batch API.

    Issues the same messages, and parses results with the same
    `parse_ordinal_response`, as `OpenAIRater`, so sync and batch execution
    yield the same ratings for the same inputs.

    Attributes:
        model: OpenAI model identifier (e.g. "gpt-4o").
        model_version: Expected resolved model version, checked against each
            batch result's own `model` field in `fetch` (see
            `OpenAIRater.model_version` and
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Batch API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``OPENAI_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.
        completion_window: Vendor-side completion SLA for the batch job.
        request_logprobs: If True, request per-token log probabilities on
            every submitted request, mirroring `OpenAIRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field. Batch
            support may lag sync support; verify with
            `tools/vendor_logprob_probe.py` rather than assuming parity.
        top_logprobs: Number of top alternative tokens to request logprobs
            for at each position, when `request_logprobs` is True.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    completion_window: str = "24h"
    request_logprobs: bool = False
    top_logprobs: int = 5
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
        body: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": compose_system_prompt(prompt)},
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                },
            ],
        }
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def _batch_request_line(
        self, records: Sequence[Record], custom_id: str, prompt: str | None
    ) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "max_completion_tokens": max(self.max_tokens, 16 * len(records)),
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": compose_batch_system_prompt(prompt)},
                    {"role": "user", "content": compose_batch_user_message(records)},
                ],
            },
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
        *,
        batch_size: int = 1,
    ) -> BatchHandle:
        """Upload a JSONL request file and submit it as one OpenAI Batch job.

        At `batch_size == 1`, one JSONL row per record -- unchanged from
        before `batch_size` packing existed. At `batch_size > 1`, `records`
        is grouped by resolved prompt (via `prompts`, preserving order) and
        split into chunks of at most `batch_size` records; one JSONL row is
        submitted per chunk (a singleton chunk still uses the single-record
        row, byte-identical to the `batch_size == 1` case), and
        `id_map` maps every record id in a chunk to that chunk's shared
        `custom_id`, so `fetch` can split the chunk's one response back into
        per-record votes.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.
            batch_size: Maximum number of records packed into one request.

        Returns:
            A `BatchHandle` identifying the submitted batch.
        """
        prompts = prompts or {}
        if batch_size <= 1:
            id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
            lines = "\n".join(
                json.dumps(
                    self._request_line(
                        record, id_map[record.id], prompts.get(record.id, self.prompt)
                    )
                )
                for record in records
            )
        else:
            chunks = chunk_records(records, lambda r: prompts.get(r.id), batch_size)
            id_map = {}
            rows: list[str] = []
            for i, chunk in enumerate(chunks):
                custom_id = f"item-{i}"
                for record in chunk:
                    id_map[record.id] = custom_id
                chunk_prompt = prompts.get(chunk[0].id, self.prompt)
                if len(chunk) == 1:
                    rows.append(json.dumps(self._request_line(chunk[0], custom_id, chunk_prompt)))
                else:
                    rows.append(
                        json.dumps(self._batch_request_line(chunk, custom_id, chunk_prompt))
                    )
            lines = "\n".join(rows)
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

        A line whose request errored is simply absent from the returned
        mapping, as is every record in a chunk (see `submit`) whose row's
        text does not parse: for a singleton chunk that is one record's
        unparseable single-letter reply (mirroring the pre-`batch_size`
        per-record behavior); for a multi-record chunk, `parse_batch_response`
        requires a rating for every record the row covers, so if any of them
        is unparseable the whole chunk's records are omitted together --
        the vendor gave back one response for the whole chunk, so a failure
        to make sense of it cannot be attributed to a single record within it.

        Raises:
            ModelVersionDriftError: If a succeeded result's `model` field
                differs from `self.model_version` -- not caught and skipped
                like a parse failure, since it invalidates every result in
                the batch, not just this record.
        """
        chunks: dict[str, list[str]] = {}
        for record_id, custom_id in handle.id_map.items():
            chunks.setdefault(custom_id, []).append(record_id)
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
            record_ids = chunks.get(entry.get("custom_id"))
            if not record_ids or entry.get("error"):
                continue
            reported_version = entry["response"]["body"].get("model")
            check_model_version(
                vendor=self.vendor,
                model=self.model,
                expected_version=self.model_version,
                reported_version=reported_version,
            )
            choice = entry["response"]["body"]["choices"][0]
            text = choice["message"]["content"] or ""
            if len(record_ids) == 1:
                try:
                    ordinal = parse_ordinal_response(text)
                except VendorResponseError:
                    continue
                record_entry = entry
                if self.request_logprobs and choice.get("logprobs") is not None:
                    record_entry = {**entry, "logprobs": choice["logprobs"]}
                results[record_ids[0]] = (ordinal, record_entry)
            else:
                try:
                    ratings = parse_batch_response(text, record_ids)
                except VendorResponseError:
                    continue
                for record_id in record_ids:
                    results[record_id] = (ratings[record_id], {**entry, "record_id": record_id})
        return results
