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
class AnthropicRater:
    """Rates records with an Anthropic Claude model via the Messages API.

    Attributes:
        model: Anthropic model identifier (e.g. "claude-sonnet-5").
        model_version: Expected resolved model version. The Messages API has
            no separate version parameter -- `model` is already what is sent
            -- so this is checked against the version the response itself
            reports (`response.model`) after every call, catching a floating
            alias silently resolving to a snapshot other than the one this
            configuration was hashed under (see
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Messages API, unless
            `send_temperature` is `False`.
        send_temperature: When `False`, `temperature` is omitted from the
            Messages API request entirely instead of being sent. Claude
            Sonnet 5 (and the rest of the Claude 4.6+ generation) rejects
            any explicit `temperature`/`top_p`/`top_k` value with HTTP 400 --
            confirmed against Anthropic's own current model documentation,
            2026-08-24 -- with no parameter analogous to OpenAI's
            `reasoning_effort="none"` that re-enables it, so omission is the
            only way to avoid the error. See
            `attest.provenance.config.VendorSpec.send_temperature`.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``ANTHROPIC_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in the reply.

    No `request_logprobs` field: unlike
    `attest.vendors.providers.openai.OpenAIRater`,
    `attest.vendors.providers.mistral.MistralRater`, and
    `attest.vendors.providers.google.GoogleRater`, the Messages API exposes
    no per-token log-probability equivalent as of this writing, so there is
    nothing for such a field to apply -- adding one anyway would be a
    decorative, never-applied config field. If Anthropic's API adds logprobs
    support, add `request_logprobs`/`top_logprobs` here following the same
    pattern the other three providers use, and wire it through
    `attest.vendors.registry._build_anthropic`.
    """

    model: str
    model_version: str
    temperature: float
    send_temperature: bool = True
    api_key: str | None = None
    prompt: str | None = None
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

    def _temperature_kwargs(self) -> dict[str, Any]:
        return {"temperature": self.temperature} if self.send_temperature else {}

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Anthropic Messages API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.

        Raises:
            ModelVersionDriftError: If the response's `model` differs from
                `self.model_version`.
        """
        response = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=compose_system_prompt(prompt if prompt is not None else self.prompt),
            messages=[
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                }
            ],
            **self._temperature_kwargs(),
        )
        reported_version = getattr(response, "model", None)
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=reported_version,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {
            "text": text,
            "id": getattr(response, "id", None),
            "model": reported_version,
            "temperature_applied": self.send_temperature,
        }
        return ordinal, raw_response

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Rate every record in `records` together, in one Messages API call.

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
        response = self._client().messages.create(
            model=self.model,
            max_tokens=max(self.max_tokens, 16 * len(records)),
            system=compose_batch_system_prompt(prompt if prompt is not None else self.prompt),
            messages=[{"role": "user", "content": compose_batch_user_message(records)}],
            **self._temperature_kwargs(),
        )
        reported_version = getattr(response, "model", None)
        check_model_version(
            vendor=self.vendor,
            model=self.model,
            expected_version=self.model_version,
            reported_version=reported_version,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        ratings = parse_batch_response(text, record_ids)
        raw_response: dict[str, Any] = {
            "text": text,
            "id": getattr(response, "id", None),
            "model": reported_version,
            "temperature_applied": self.send_temperature,
        }
        return [
            (ratings[record_id], {**raw_response, "record_id": record_id})
            for record_id in record_ids
        ]


@dataclass
class AnthropicBatchRater:
    """Rates records with an Anthropic Claude model via the Message Batches API.

    Issues the same system prompt and user message, and parses results with
    the same `parse_ordinal_response`, as `AnthropicRater`, so sync and batch
    execution yield the same ratings for the same inputs.

    Attributes:
        model: Anthropic model identifier (e.g. "claude-sonnet-5").
        model_version: Expected resolved model version, checked against each
            batch result's own `message.model` in `fetch` (see
            `AnthropicRater.model_version` and
            `attest.vendors.base.check_model_version`).
        temperature: Sampling temperature passed to the Message Batches API,
            unless `send_temperature` is `False`.
        send_temperature: When `False`, `temperature` is omitted from every
            submitted request's params instead of being sent -- see
            `AnthropicRater.send_temperature` for why.
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``ANTHROPIC_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_tokens: Maximum tokens to request in each reply.

    No `request_logprobs` field, for the same reason as `AnthropicRater` --
    see its docstring.
    """

    model: str
    model_version: str
    temperature: float
    send_temperature: bool = True
    api_key: str | None = None
    prompt: str | None = None
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

    def _request(self, record: Record, custom_id: str, prompt: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": compose_system_prompt(prompt),
            "messages": [
                {
                    "role": "user",
                    "content": f"Title: {record.title}\nAbstract: {record.abstract}",
                }
            ],
        }
        if self.send_temperature:
            params["temperature"] = self.temperature
        return {"custom_id": custom_id, "params": params}

    def _batch_request(
        self, records: Sequence[Record], custom_id: str, prompt: str | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(self.max_tokens, 16 * len(records)),
            "system": compose_batch_system_prompt(prompt),
            "messages": [{"role": "user", "content": compose_batch_user_message(records)}],
        }
        if self.send_temperature:
            params["temperature"] = self.temperature
        return {"custom_id": custom_id, "params": params}

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
        *,
        batch_size: int = 1,
    ) -> BatchHandle:
        """Submit `records` as one Anthropic Message Batch.

        At `batch_size == 1`, one batch request per record -- unchanged from
        before `batch_size` packing existed. At `batch_size > 1`, `records`
        is grouped by resolved prompt (via `prompts`, preserving order) and
        split into chunks of at most `batch_size` records; one batch request
        is submitted per chunk (a singleton chunk still uses the
        single-record request, byte-identical to the `batch_size == 1`
        case), and `id_map` maps every record id in a chunk to that chunk's
        shared `custom_id`, so `fetch` can split the chunk's one result back
        into per-record votes.

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
            requests = [
                self._request(record, id_map[record.id], prompts.get(record.id, self.prompt))
                for record in records
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
                    requests.append(self._request(chunk[0], custom_id, chunk_prompt))
                else:
                    requests.append(self._batch_request(chunk, custom_id, chunk_prompt))
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
        """Retrieve and parse this batch's per-chunk results.

        A record whose result did not succeed is simply absent from the
        returned mapping, as is every record in a chunk (see `submit`)
        whose result's text does not parse: for a singleton chunk that is
        one record's unparseable single-letter reply (mirroring the
        pre-`batch_size` per-record behavior); for a multi-record chunk,
        `parse_batch_response` requires a rating for every record the
        result covers, so if any of them is unparseable the whole chunk's
        records are omitted together -- the vendor gave back one result for
        the whole chunk, so a failure to make sense of it cannot be
        attributed to a single record within it.

        Raises:
            ModelVersionDriftError: If a succeeded result's `message.model`
                differs from `self.model_version` -- not caught and skipped
                like a parse failure, since it invalidates every result in
                the batch, not just this record.
        """
        chunks: dict[str, list[str]] = {}
        for record_id, custom_id in handle.id_map.items():
            chunks.setdefault(custom_id, []).append(record_id)
        results: dict[str, tuple[int, Any]] = {}
        for entry in self._client().messages.batches.results(handle.provider_batch_id):
            record_ids = chunks.get(entry.custom_id)
            if not record_ids or entry.result.type != "succeeded":
                continue
            reported_version = getattr(entry.result.message, "model", None)
            check_model_version(
                vendor=self.vendor,
                model=self.model,
                expected_version=self.model_version,
                reported_version=reported_version,
            )
            text = "".join(
                block.text
                for block in entry.result.message.content
                if getattr(block, "type", "") == "text"
            )
            if len(record_ids) == 1:
                try:
                    ordinal = parse_ordinal_response(text)
                except VendorResponseError:
                    continue
                results[record_ids[0]] = (
                    ordinal,
                    {
                        "text": text,
                        "custom_id": entry.custom_id,
                        "model": reported_version,
                        "temperature_applied": self.send_temperature,
                    },
                )
            else:
                try:
                    ratings = parse_batch_response(text, record_ids)
                except VendorResponseError:
                    continue
                for record_id in record_ids:
                    results[record_id] = (
                        ratings[record_id],
                        {
                            "text": text,
                            "custom_id": entry.custom_id,
                            "model": reported_version,
                            "record_id": record_id,
                            "temperature_applied": self.send_temperature,
                        },
                    )
        return results
