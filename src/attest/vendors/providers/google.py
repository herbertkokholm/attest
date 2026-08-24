"""Google Gemini adapter for the `Rater` and `BatchRater` protocols.

Requires the `google-genai` package, gated behind the `attest[google]`
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
    VendorResponseError,
    chunk_records,
    compose_batch_system_prompt,
    compose_batch_user_message,
    compose_system_prompt,
    parse_batch_response,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle, BatchStatus

_BATCH_TERMINAL_STATES = frozenset({"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"})
_BATCH_FAILED_STATES = frozenset({"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"})


def _serialize_logprobs(logprobs_result: Any) -> Any:
    """Best-effort plain-data conversion of a Gemini `logprobs_result`.

    `google-genai`'s response objects are uniformly pydantic, so
    `model_dump` should always succeed; `to_dict` is tried first only for
    forward compatibility with SDK versions that might expose it, and
    `str()` is the last-resort fallback rather than raising, since the raw
    response is retained for audit/debugging only (see
    `attest.vendors.base.Rater.rate`) and should never block a run.
    """
    for method_name in ("to_dict", "model_dump"):
        method = getattr(logprobs_result, method_name, None)
        if callable(method):
            return method()
    return str(logprobs_result)


@dataclass
class GoogleRater:
    """Rates records with a Google Gemini model via `google-genai`.

    Attributes:
        model: Gemini model identifier (e.g. "gemini-1.5-pro").
        model_version: Expected resolved model version. Retained for the
            ensemble configuration's hash/audit trail only -- unlike
            Anthropic/OpenAI/Mistral, `generate_content`'s response does not
            expose which resolved snapshot served the request, so there is
            no vendor-reported value to check this against here (see
            `attest.vendors.base.check_model_version`, which every other
            provider's `rate` calls but this one cannot).
        temperature: Sampling temperature passed to `generate_content` via
            `config`.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``GOOGLE_API_KEY``, falling back to ``GEMINI_API_KEY``)
            when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_output_tokens: Maximum tokens to request in the reply.
        request_logprobs: If True, request per-token log probabilities
            (`response_logprobs`/`logprobs` in `config`) and retain the
            vendor's own, un-normalized logprob structure in the raw
            response under `"logprobs"`. Mirrors
            `attest.vendors.providers.openai.OpenAIRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field. Support
            is model-family-dependent on Gemini and unverified here; check
            with `tools/vendor_logprob_probe.py` before relying on it.
        top_logprobs: Number of top alternative tokens to request logprobs
            for at each position, when `request_logprobs` is True.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    prompt: str | None = None
    max_output_tokens: int = 8
    request_logprobs: bool = False
    top_logprobs: int = 5
    vendor: str = field(default="google", init=False)

    def _client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "the 'google-genai' package is required to use GoogleRater; "
                "install it with: pip install 'attest[google]'"
            ) from exc
        return genai.Client(api_key=self.api_key)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Gemini `generate_content` API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        system_prompt = compose_system_prompt(prompt if prompt is not None else self.prompt)
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.request_logprobs:
            config["response_logprobs"] = True
            config["logprobs"] = self.top_logprobs
        response = self._client().models.generate_content(
            model=self.model,
            contents=f"Title: {record.title}\nAbstract: {record.abstract}",
            config=config,
        )
        text = response.text
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {"text": text}
        if self.request_logprobs:
            logprobs_result = getattr(response.candidates[0], "logprobs_result", None)
            if logprobs_result is not None:
                raw_response["logprobs"] = _serialize_logprobs(logprobs_result)
        return ordinal, raw_response

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Rate every record in `records` together, in one `generate_content` call.

        Packs all of `records` into a single request: the system
        instruction uses `compose_batch_system_prompt` (a per-id JSON
        output contract) instead of `compose_system_prompt`'s single
        letter, and the user content lists every record by id
        (`compose_batch_user_message`). Parsed by id via
        `parse_batch_response`, order-independent.

        Args:
            records: The records to rate together.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            One `(ordinal, raw_response)` pair per record in `records`, in order.

        Raises:
            VendorResponseError: If some record has no parseable rating in
                the response.
        """
        record_ids = [record.id for record in records]
        system_prompt = compose_batch_system_prompt(prompt if prompt is not None else self.prompt)
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max(self.max_output_tokens, 16 * len(records)),
            "temperature": self.temperature,
        }
        response = self._client().models.generate_content(
            model=self.model,
            contents=compose_batch_user_message(records),
            config=config,
        )
        text = response.text
        ratings = parse_batch_response(text, record_ids)
        raw_response: dict[str, Any] = {"text": text}
        return [
            (ratings[record_id], {**raw_response, "record_id": record_id})
            for record_id in record_ids
        ]


@dataclass
class GoogleBatchRater:
    """Rates records with a Google Gemini model via its batch prediction API.

    Issues the same system instruction and prompt, and parses results with
    the same `parse_ordinal_response`, as `GoogleRater`, so sync and batch
    execution yield the same ratings for the same inputs.

    Gemini's batch surface is newer and less stable than the synchronous
    `generate_content` API `GoogleRater` uses, and -- like the other batch
    adapters -- this class is not exercised against a live API in this
    codebase's tests, only `attest.vendors.batch.DeterministicBatchRater` is.
    The method names below match `google-genai`'s batch job surface as of
    this writing; adjust `_client`/`submit`/`poll`/`fetch` if an installed
    SDK version has renamed them.

    `fetch` correlates each inlined response back to the request that
    produced it purely by position -- `client.batches.get`'s
    `dest.inlined_responses` is documented to come back in the same order
    as the submitted `inlined_requests` -- rather than by any per-request
    identifier, since inlined batch requests/responses carry only a free-form
    `metadata` dict with no guaranteed round-trip semantics.

    Attributes:
        model: Gemini model identifier (e.g. "gemini-1.5-pro").
        model_version: Expected resolved model version. Retained for the
            ensemble configuration's hash/audit trail only -- see
            `GoogleRater.model_version` for why this batch surface cannot be
            checked against a vendor-reported value either.
        temperature: Sampling temperature passed to each request's `config`.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``GOOGLE_API_KEY``, falling back to ``GEMINI_API_KEY``)
            when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_output_tokens: Maximum tokens to request in each reply.
        request_logprobs: If True, request per-token log probabilities on
            every submitted request, mirroring `GoogleRater.request_logprobs`
            -- see there for why this is not a `VendorSpec` field, and why
            support is unverified without `tools/vendor_logprob_probe.py`.
            Gemini's batch surface is newer than the sync surface, so batch
            support should not be assumed even where sync support exists.
        top_logprobs: Number of top alternative tokens to request logprobs
            for at each position, when `request_logprobs` is True.
    """

    model: str
    model_version: str
    temperature: float
    api_key: str | None = None
    prompt: str | None = None
    max_output_tokens: int = 8
    request_logprobs: bool = False
    top_logprobs: int = 5
    vendor: str = field(default="google", init=False)

    def _client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "the 'google-genai' package is required to use GoogleBatchRater; "
                "install it with: pip install 'attest[google]'"
            ) from exc
        return genai.Client(api_key=self.api_key)

    def _request(self, record: Record, prompt: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": compose_system_prompt(prompt),
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.request_logprobs:
            config["response_logprobs"] = True
            config["logprobs"] = self.top_logprobs
        return {
            "contents": [f"Title: {record.title}\nAbstract: {record.abstract}"],
            "config": config,
        }

    def _batch_request(self, records: Sequence[Record], prompt: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": compose_batch_system_prompt(prompt),
            "max_output_tokens": max(self.max_output_tokens, 16 * len(records)),
            "temperature": self.temperature,
        }
        return {
            "contents": [compose_batch_user_message(records)],
            "config": config,
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
        *,
        batch_size: int = 1,
    ) -> BatchHandle:
        """Submit `records` as one Gemini batch job.

        At `batch_size == 1`, one inlined request per record -- unchanged
        from before `batch_size` packing existed. At `batch_size > 1`,
        `records` is grouped by resolved prompt (via `prompts`, preserving
        order) and split into chunks of at most `batch_size` records; one
        inlined request is submitted per chunk (a singleton chunk still uses
        the single-record request, byte-identical to the `batch_size == 1`
        case), and `id_map` maps every record id in a chunk to that chunk's
        shared `key` (in submission order), so `fetch` can zip the job's
        positionally-ordered responses back against the right records.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.
            batch_size: Maximum number of records packed into one request.

        Returns:
            A `BatchHandle` identifying the submitted batch job.
        """
        prompts = prompts or {}
        client = self._client()
        if batch_size <= 1:
            id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
            requests = [
                self._request(record, prompts.get(record.id, self.prompt)) for record in records
            ]
        else:
            chunks = chunk_records(records, lambda r: prompts.get(r.id), batch_size)
            id_map = {}
            requests = []
            for i, chunk in enumerate(chunks):
                custom_id = f"item-{i}"
                for record in chunk:
                    id_map[record.id] = custom_id
                chunk_prompt = prompts.get(chunk[0].id, self.prompt)
                if len(chunk) == 1:
                    requests.append(self._request(chunk[0], chunk_prompt))
                else:
                    requests.append(self._batch_request(chunk, chunk_prompt))
        job = client.batches.create(model=self.model, src=requests)
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=job.name,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the batch job's `state`."""
        job = self._client().batches.get(name=handle.provider_batch_id)
        if job.state in _BATCH_TERMINAL_STATES:
            return "completed"
        if job.state in _BATCH_FAILED_STATES:
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Retrieve and parse this batch job's per-chunk inlined responses.

        A record whose response is missing is simply absent from the
        returned mapping, as is every record in a chunk (see `submit`) whose
        response text does not parse: for a singleton chunk that is one
        record's unparseable single-letter reply (mirroring the
        pre-`batch_size` per-record behavior); for a multi-record chunk,
        `parse_batch_response` requires a rating for every record the
        response covers, so if any of them is unparseable the whole chunk's
        records are omitted together.
        """
        job = self._client().batches.get(name=handle.provider_batch_id)
        chunks: dict[str, list[str]] = {}
        for record_id, custom_id in handle.id_map.items():
            chunks.setdefault(custom_id, []).append(record_id)
        inlined_responses = job.dest.inlined_responses if job.dest is not None else None
        results: dict[str, tuple[int, Any]] = {}
        for custom_id, entry in zip(chunks.keys(), inlined_responses or (), strict=False):
            record_ids = chunks[custom_id]
            if entry.response is None:
                continue
            text = entry.response.text
            if len(record_ids) == 1:
                try:
                    ordinal = parse_ordinal_response(text)
                except VendorResponseError:
                    continue
                raw: dict[str, Any] = {"text": text, "key": custom_id}
                if self.request_logprobs:
                    logprobs_result = getattr(entry.response.candidates[0], "logprobs_result", None)
                    if logprobs_result is not None:
                        raw["logprobs"] = _serialize_logprobs(logprobs_result)
                results[record_ids[0]] = (ordinal, raw)
            else:
                try:
                    ratings = parse_batch_response(text, record_ids)
                except VendorResponseError:
                    continue
                for record_id in record_ids:
                    results[record_id] = (
                        ratings[record_id],
                        {"text": text, "key": custom_id, "record_id": record_id},
                    )
        return results
