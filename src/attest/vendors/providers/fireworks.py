"""Fireworks AI adapter for the `Rater` and `BatchRater` protocols.

Requires the `fireworks-ai` package, gated behind the `attest[fireworks]`
extra. The SDK is imported lazily inside `_client`, so this module itself
imports cleanly without the extra installed.

`FireworksRater` calls the OpenAI-compatible `chat.completions.create`
endpoint, confirmed against the installed `fireworks-ai` SDK (v1.2.6): the
response has the same `.choices[0].message.content` / `.model` shape as
OpenAI's.

`FireworksBatchRater` is a best-effort implementation of a genuinely
different batch model than every other vendor here: Fireworks' Batch
Inference API is dataset/job-resource based (create a dataset, upload a
JSONL file to it, create a `batchInferenceJobs` job referencing an input and
output dataset id, poll the job's `state`, then download the output dataset
via a signed URL) rather than OpenAI's file-upload-plus-`/v1/batches` shape.
The job-resource request/response schema below (`datasets.create/upload/get`,
`batch_inference_jobs.create/get`, the `JOB_STATE_*` enum, the job-level
`systemPrompt` field) is confirmed against Fireworks' own OpenAPI spec and
the installed SDK. Two things are *not* confirmed and are inferred instead:

1. The per-row JSONL schema for a batch-inference (as opposed to
   fine-tuning) CHAT dataset. Fireworks' bundled docs only show the
   fine-tuning row shape (`{"messages": [...]}` including the target
   assistant turn). This adapter assumes the same `{"messages": [...]}`
   shape for input rows (system + user turn, no assistant turn -- that is
   what the job is meant to generate), and assumes each output row is the
   same conversation with a trailing assistant turn appended, whose content
   is the parsed rating.
2. Row-to-row correlation between input and output. Fireworks' batch API
   exposes no `custom_id`/row-id concept anywhere in its schema, so this
   assumes output rows preserve input row order, concatenated across output
   files in filename-sorted order if the output dataset is sharded into more
   than one file.

Both assumptions should be verified against a real batch run (or updated
Fireworks documentation) before relying on `FireworksBatchRater` in
production; see the request/response inspection this module was built from
in the branch history for how far the confirmed part of this contract goes.

Fireworks' system prompt could alternatively be set once, job-level, via the
job's own `systemPrompt` field (injected into every row lacking its own
leading system message) -- but this adapter instead puts the composed system
prompt directly in each row's `messages`, exactly like every other batch
adapter in `attest.vendors.providers`, so records rated under different
`Config.track_prompts` criteria within the same batch stay correct; the
job-level field is never used.
"""

from __future__ import annotations

import json
import re
import urllib.request
import uuid
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

_ID_SANITIZE_RE = re.compile(r"[^a-z0-9-]")


def _dataset_id(ensemble_config_id: str, suffix: str) -> str:
    """Derive a Fireworks-legal dataset id from an ensemble config id.

    Fireworks dataset ids must be lowercase alphanumeric-and-hyphen; a random
    suffix is appended (rather than relying on `ensemble_config_id` alone) so
    resubmitting under the same configuration never collides with a dataset
    left over from a previous run.
    """
    sanitized = _ID_SANITIZE_RE.sub("-", ensemble_config_id.lower())[:40].strip("-") or "batch"
    return f"attest-{sanitized}-{uuid.uuid4().hex[:8]}-{suffix}"


@dataclass
class FireworksRater:
    """Rates records with a Fireworks AI model via its Chat Completions API.

    Attributes:
        model: Fireworks model identifier (e.g.
            "accounts/fireworks/models/llama-v3p1-70b-instruct").
        model_version: Expected resolved model version. The Chat Completions
            API has no separate version parameter -- `model` is already what
            is sent -- so this is checked against the version the response
            itself reports (`response.model`) after every call, catching a
            floating alias silently resolving to a snapshot other than the
            one this configuration was hashed under (see
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Chat Completions API.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``FIREWORKS_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    vendor: str = field(default="fireworks", init=False)

    def _client(self) -> Any:
        try:
            from fireworks import Fireworks
        except ImportError as exc:
            raise ImportError(
                "the 'fireworks-ai' package is required to use FireworksRater; "
                "install it with: pip install 'attest[fireworks]'"
            ) from exc
        return Fireworks(api_key=self.api_key)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Fireworks Chat Completions API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
        """
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
        return ordinal, raw_response


@dataclass
class FireworksBatchRater:
    """Rates records with a Fireworks AI model via its Batch Inference API.

    See the module docstring for the full account of what is confirmed
    (the dataset/job resource lifecycle, field names, and state enum) versus
    inferred (the per-row JSONL schema and output-row correlation) in this
    adapter -- verify both before relying on this in production.

    Attributes:
        model: Fireworks model identifier (e.g.
            "accounts/fireworks/models/llama-v3p1-70b-instruct").
        model_version: Expected resolved model version, checked in `fetch`
            against the completed job's own `model` field (see
            `FireworksRater.model_version` and
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed via the job's
            `inferenceParameters`.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``FIREWORKS_API_KEY``) when None.
        account_id: Fireworks account id that owns the datasets and batch
            job this rater creates; defaults to the SDK's own environment
            lookup (``FIREWORKS_ACCOUNT_ID``) when None. Required (by the
            Fireworks API itself, not this adapter) for every dataset and
            batch-job call, unlike `FireworksRater`'s plain chat completions.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.
        timeout: Timeout in seconds for downloading the output dataset's
            signed-URL content in `fetch`.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    account_id: str | None = None
    prompt: str | None = None
    max_tokens: int = 8
    timeout: float = 30.0
    vendor: str = field(default="fireworks", init=False)

    def _client(self) -> Any:
        try:
            from fireworks import Fireworks
        except ImportError as exc:
            raise ImportError(
                "the 'fireworks-ai' package is required to use FireworksBatchRater; "
                "install it with: pip install 'attest[fireworks]'"
            ) from exc
        return Fireworks(api_key=self.api_key, account_id=self.account_id)

    def _input_row(self, record: Record, prompt: str | None) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": compose_system_prompt(prompt)},
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                },
            ]
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Create an input dataset, upload it, and submit one Fireworks batch inference job.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.

        Returns:
            A `BatchHandle` identifying the submitted job.

        Raises:
            VendorResponseError: If the create-job response has no `name`,
                which would leave the returned handle unable to poll or
                fetch this job.
        """
        prompts = prompts or {}
        id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
        lines = "\n".join(
            json.dumps(self._input_row(record, prompts.get(record.id, self.prompt)))
            for record in records
        )
        client = self._client()
        input_dataset_id = _dataset_id(ensemble_config_id, "in")
        output_dataset_id = _dataset_id(ensemble_config_id, "out")
        client.datasets.create(
            dataset_id=input_dataset_id,
            dataset={"exampleCount": str(len(records)), "format": "CHAT"},
        )
        client.datasets.upload(dataset_id=input_dataset_id, file=lines.encode("utf-8"))
        client.datasets.create(dataset_id=output_dataset_id, dataset={"format": "CHAT"})
        job = client.batch_inference_jobs.create(
            model=self.model,
            input_dataset_id=input_dataset_id,
            output_dataset_id=output_dataset_id,
            inference_parameters={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
        if job.name is None:
            raise VendorResponseError(
                "fireworks batch_inference_jobs.create response did not include a job name"
            )
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=job.name,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the batch inference job's `state`."""
        job = self._client().batch_inference_jobs.get(
            batch_inference_job_id=handle.provider_batch_id
        )
        if job.state == "JOB_STATE_COMPLETED":
            return "completed"
        if job.state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Download and parse the output dataset's JSONL content.

        Assumes output rows are in the same order as the submitted input
        rows (see module docstring); a row whose content does not parse as
        an ordinal rating is simply absent from the returned mapping.

        Raises:
            ModelVersionDriftError: If the completed job's own `model` field
                differs from `self.model_version` -- checked once for the
                whole job, since Fireworks batch output rows carry no
                per-row model field to check individually.
        """
        client = self._client()
        job = client.batch_inference_jobs.get(batch_inference_job_id=handle.provider_batch_id)
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=job.model,
        )
        output_dataset_id = job.output_dataset_id
        if output_dataset_id is None:
            return {}
        download = client.datasets.get_download_endpoint(dataset_id=output_dataset_id)
        reverse = {custom_id: record_id for record_id, custom_id in handle.id_map.items()}
        results: dict[str, tuple[int, Any]] = {}
        index = 0
        for filename in sorted(download.filename_to_signed_urls):
            url = download.filename_to_signed_urls[filename]
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                record_id = reverse.get(f"item-{index}")
                index += 1
                if record_id is None:
                    continue
                entry = json.loads(line)
                messages = entry.get("messages") or []
                assistant_texts = [
                    m.get("content") for m in messages if m.get("role") == "assistant"
                ]
                if not assistant_texts or not assistant_texts[-1]:
                    continue
                try:
                    ordinal = parse_ordinal_response(assistant_texts[-1])
                except VendorResponseError:
                    continue
                results[record_id] = (ordinal, entry)
        return results
