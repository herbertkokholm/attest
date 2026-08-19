"""Together AI adapter for the `Rater` and `BatchRater` protocols.

Requires the `together` package, gated behind the `attest[together]` extra.
The SDK is imported lazily inside `_client`, so this module itself imports
cleanly without the extra installed.

Together's chat and batch surfaces were confirmed against the installed
`together` SDK (v2.28.0): `chat.completions.create` returns a response with
the same `.choices[0].message.content` / `.model` shape as OpenAI's, and
`files`/`batches` mirror OpenAI's Batch API closely enough (`purpose="batch-api"`
file upload, `batches.create(input_file_id, endpoint, completion_window)`,
a `status` enum of VALIDATING/IN_PROGRESS/COMPLETED/FAILED/EXPIRED/CANCELLED,
`output_file_id`/`error_file_id`) that `TogetherBatchRater` below submits the
same `custom_id`/`method`/`url`/`body` JSONL convention as
`attest.vendors.providers.openai.OpenAIBatchRater`. That per-line convention
itself is not directly documented in the installed package -- there is no
bundled worked example of a batch input file -- so treat it as a
high-confidence inference from the API shape rather than a confirmed fact,
and verify against a real batch run before relying on it in production.
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
    SingleRecordOnlyRateMany,
    VendorResponseError,
    check_model_version,
    compose_system_prompt,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


@dataclass
class TogetherRater(SingleRecordOnlyRateMany):
    """Rates records with a Together AI model via its Chat Completions API.

    Attributes:
        model: Together model identifier (e.g. "meta-llama/Llama-3.3-70B-Instruct-Turbo").
        model_version: Expected resolved model version. The Chat Completions
            API has no separate version parameter -- `model` is already what
            is sent -- so this is checked against the version the response
            itself reports (`response.model`) after every call, catching a
            floating alias silently resolving to a snapshot other than the
            one this configuration was hashed under (see
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Chat Completions API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``TOGETHER_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
        request_logprobs: If True, request per-token log probabilities
            (`logprobs=True, top_logprobs=top_logprobs`) and retain the
            vendor's own, un-normalized logprob structure in the raw
            response under `"logprobs"`. Mirrors
            `attest.vendors.providers.openai.OpenAIRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field. Together's
            Chat Completions endpoint is OpenAI-compatible, so this is a
            high-confidence but still unverified-in-practice add; check with
            `tools/vendor_logprob_probe.py` before relying on it.
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
    vendor: str = field(default="together", init=False)

    def _client(self) -> Any:
        try:
            from together import Together
        except ImportError as exc:
            raise ImportError(
                "the 'together' package is required to use TogetherRater; "
                "install it with: pip install 'attest[together]'"
            ) from exc
        return Together(api_key=self.api_key)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Together Chat Completions API.

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
            max_tokens=self.max_tokens,
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
            dump = getattr(response_logprobs, "model_dump", None)
            raw_response["logprobs"] = dump() if callable(dump) else response_logprobs
        return ordinal, raw_response


@dataclass
class TogetherBatchRater:
    """Rates records with a Together AI model via its Batch API.

    Issues the same messages, and parses results with the same
    `parse_ordinal_response`, as `TogetherRater`, so sync and batch execution
    yield the same ratings for the same inputs.

    Together's batch surface mirrors OpenAI's (upload a JSONL request file,
    create a batch job against it, poll `status`, download a JSONL output
    file) and -- like the other batch adapters -- is not exercised against a
    live API in this codebase's tests, only
    `attest.vendors.batch.DeterministicBatchRater` is. The per-line
    `custom_id`/`method`/`url`/`body` request convention is an inference from
    the SDK's shape (see module docstring), not a confirmed fact; adjust
    `_client`/`submit`/`poll`/`fetch` if a live batch run shows otherwise.

    Attributes:
        model: Together model identifier (e.g. "meta-llama/Llama-3.3-70B-Instruct-Turbo").
        model_version: Expected resolved model version, checked against each
            batch result's own `model` field in `fetch` (see
            `TogetherRater.model_version` and
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Batch API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``TOGETHER_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.
        completion_window: Vendor-side completion SLA for the batch job.
        request_logprobs: If True, request per-token log probabilities on
            every submitted request, mirroring
            `TogetherRater.request_logprobs` -- see there for why this is
            not a `VendorSpec` field, and why support is a high-confidence
            but still unverified-in-practice inference from the
            OpenAI-compatible shape. Batch support may lag sync support even
            if sync does support it; verify with
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
    vendor: str = field(default="together", init=False)

    def _client(self) -> Any:
        try:
            from together import Together
        except ImportError as exc:
            raise ImportError(
                "the 'together' package is required to use TogetherBatchRater; "
                "install it with: pip install 'attest[together]'"
            ) from exc
        return Together(api_key=self.api_key)

    def _request_line(self, record: Record, custom_id: str, prompt: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
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

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
        *,
        batch_size: int = 1,
    ) -> BatchHandle:
        """Upload a JSONL request file and submit it as one Together batch job.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.
            batch_size: Maximum number of records packed into one request.
                This provider has not yet been converted to true
                multi-record packing (see `Config.batch_size`).

        Returns:
            A `BatchHandle` identifying the submitted batch.

        Raises:
            NotImplementedError: If `batch_size > 1`. Never falls back to a
                silent one-request-per-record loop: that would send one
                record per request while the configuration's hashed
                `batch_size` claims more, misrepresenting the instrument
                that actually ran.
        """
        if batch_size > 1:
            raise NotImplementedError(
                f"{type(self).__name__}.submit does not support packing more than one record "
                f"into a request (batch_size={batch_size}); this provider has not yet been "
                "converted to true multi-record packing -- see Config.batch_size"
            )
        prompts = prompts or {}
        id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
        lines = "\n".join(
            json.dumps(
                self._request_line(record, id_map[record.id], prompts.get(record.id, self.prompt))
            )
            for record in records
        )
        client = self._client()
        upload = client.files.upload(file=io.BytesIO(lines.encode("utf-8")), purpose="batch-api")
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
        """Check the batch job's `status`."""
        batch = self._client().batches.retrieve(handle.provider_batch_id)
        if batch.status == "COMPLETED":
            return "completed"
        if batch.status in ("FAILED", "EXPIRED", "CANCELLED"):
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Download and parse the batch job's output JSONL.

        A line whose request errored, or whose text does not parse as an
        ordinal rating, is simply absent from the returned mapping.

        Raises:
            ModelVersionDriftError: If a succeeded result's `model` field
                differs from `self.model_version` -- not caught and skipped
                like a parse failure, since it invalidates every result in
                the batch, not just this record.
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
            reported_version = entry["response"]["body"].get("model")
            check_model_version(
                vendor=self.vendor,
                model=self.model,
                expected_version=self.model_version,
                reported_version=reported_version,
            )
            choice = entry["response"]["body"]["choices"][0]
            text = choice["message"]["content"] or ""
            try:
                ordinal = parse_ordinal_response(text)
            except VendorResponseError:
                continue
            if self.request_logprobs and choice.get("logprobs") is not None:
                entry = {**entry, "logprobs": choice["logprobs"]}
            results[record_id] = (ordinal, entry)
        return results
