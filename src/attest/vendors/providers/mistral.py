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
    VendorResponseError,
    check_model_version,
    compose_system_prompt,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus


@dataclass
class MistralRater:
    """Rates records with a Mistral model via the (la Plateforme) chat API.

    Attributes:
        model: Mistral model identifier (e.g. "mistral-small-latest").
        model_version: Expected resolved model version. The chat API has no
            separate version parameter -- `model` is already what is sent --
            so this is checked against the version the response itself
            reports (`response.model`) after every call, catching a floating
            alias silently resolving to a snapshot other than the one this
            configuration was hashed under (see
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the chat completion API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``MISTRAL_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
        request_logprobs: If True, request per-token log probabilities and
            retain the vendor's own, un-normalized logprob structure in the
            raw response under `"logprobs"`. Mirrors
            `attest.vendors.providers.openai.OpenAIRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field. Whether
            Mistral's chat completion API actually honors `logprobs`/
            `top_logprobs` is unverified; check with
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

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
        """
        logprobs_kwargs: dict[str, Any] = (
            {"logprobs": True, "top_logprobs": self.top_logprobs} if self.request_logprobs else {}
        )
        response = self._client().chat.complete(
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
        model_version: Expected resolved model version, checked against each
            batch result's own `model` field in `fetch` (see
            `MistralRater.model_version` and
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Batch API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``MISTRAL_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.
        request_logprobs: If True, request per-token log probabilities on
            every submitted request, mirroring `MistralRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field, and why
            support is unverified without `tools/vendor_logprob_probe.py`.
            Batch support may lag sync support even if sync does support it.
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
            "body": body,
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

        Raises:
            ModelVersionDriftError: If a succeeded result's `model` field
                differs from `self.model_version` -- not caught and skipped
                like a parse failure, since it invalidates every result in
                the batch, not just this record.
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
