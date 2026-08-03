"""Google Gemini adapter for the `Rater` and `BatchRater` protocols.

Requires the `google-generativeai` package, gated behind the
`attest[google]` extra. The SDK is imported lazily inside `_client`, so this
module itself imports cleanly without the extra installed.
"""

from __future__ import annotations

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
class GoogleRater:
    """Rates records with a Google Gemini model via `google-generativeai`.

    Attributes:
        model: Gemini model identifier (e.g. "gemini-1.5-pro").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``GOOGLE_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_output_tokens: Maximum tokens to request in the reply.
    """

    model: str
    api_key: str | None = None
    prompt: str | None = None
    max_output_tokens: int = 8
    vendor: str = field(default="google", init=False)

    def _client(self, prompt: str) -> Any:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "the 'google-generativeai' package is required to use GoogleRater; "
                "install it with: pip install 'attest[google]'"
            ) from exc
        if self.api_key is not None:
            genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(model_name=self.model, system_instruction=prompt)

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Rate `record` by calling the Gemini `generate_content` API.

        Args:
            record: The record to rate.
            prompt: Screening prompt to use, overriding `self.prompt`.

        Returns:
            The parsed ordinal rating and the raw response payload.
        """
        system_prompt = compose_system_prompt(prompt if prompt is not None else self.prompt)
        response = self._client(system_prompt).generate_content(
            f"Title: {record.title}\nAbstract: {record.abstract}",
            generation_config={"max_output_tokens": self.max_output_tokens},
        )
        text = response.text
        ordinal = parse_ordinal_response(text)
        raw_response: dict[str, Any] = {"text": text}
        return ordinal, raw_response


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
    The method names below match `google-generativeai`'s batch job surface
    as of this writing; adjust `_client`/`submit`/`poll`/`fetch` if an
    installed SDK version has renamed them.

    Attributes:
        model: Gemini model identifier (e.g. "gemini-1.5-pro").
        api_key: API key to use; defaults to the SDK's own environment
            lookup (``GOOGLE_API_KEY``) when None.
        prompt: Screening criteria text, or None to use the kernel's generic
            fallback (`attest.vendors.base.SCREENING_TASK_PREAMBLE`). Criteria
            only -- `compose_system_prompt` appends the output contract, so
            this must never itself already contain a copy of it.
        max_output_tokens: Maximum tokens to request in each reply.
    """

    model: str
    api_key: str | None = None
    prompt: str | None = None
    max_output_tokens: int = 8
    vendor: str = field(default="google", init=False)

    def _client(self) -> Any:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "the 'google-generativeai' package is required to use GoogleBatchRater; "
                "install it with: pip install 'attest[google]'"
            ) from exc
        if self.api_key is not None:
            genai.configure(api_key=self.api_key)
        return genai

    def _request(self, record: Record, custom_id: str, prompt: str | None) -> dict[str, Any]:
        return {
            "key": custom_id,
            "request": {
                "contents": [
                    {"parts": [{"text": f"Title: {record.title}\nAbstract: {record.abstract}"}]}
                ],
                "system_instruction": {"parts": [{"text": compose_system_prompt(prompt)}]},
                "generation_config": {"max_output_tokens": self.max_output_tokens},
            },
        }

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Submit `records` as one Gemini batch job.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt to use for
                it, overriding `self.prompt`.

        Returns:
            A `BatchHandle` identifying the submitted batch job.
        """
        prompts = prompts or {}
        genai = self._client()
        id_map = {record.id: f"item-{i}" for i, record in enumerate(records)}
        requests = [
            self._request(record, id_map[record.id], prompts.get(record.id, self.prompt))
            for record in records
        ]
        job = genai.GenerativeModel(model_name=self.model).batches.create(requests=requests)
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
        job = self._client().get_batch(handle.provider_batch_id)
        if job.state == "BATCH_STATE_SUCCEEDED":
            return "completed"
        if job.state in ("BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"):
            return "failed"
        return "pending"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Retrieve and parse this batch job's per-record inlined responses.

        A record whose response is missing, or whose text does not parse as
        an ordinal rating, is simply absent from the returned mapping.
        """
        job = self._client().get_batch(handle.provider_batch_id)
        reverse = {custom_id: record_id for record_id, custom_id in handle.id_map.items()}
        results: dict[str, tuple[int, Any]] = {}
        for entry in job.dest.inlined_responses:
            record_id = reverse.get(entry.key)
            if record_id is None or entry.response is None:
                continue
            text = entry.response.text
            try:
                ordinal = parse_ordinal_response(text)
            except VendorResponseError:
                continue
            results[record_id] = (ordinal, {"text": text, "key": entry.key})
        return results
