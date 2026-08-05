"""Asynchronous vendor Batch API execution: a parallel path to `run_ensemble`.

Vendor Batch APIs are asynchronous -- submit a set of requests, receive a
batch id, poll until it completes, retrieve results keyed by a per-item id --
which does not fit the synchronous `attest.vendors.base.Rater.rate(record)`
protocol. `BatchRater` is the parallel abstraction that does fit it, and
`run_ensemble_batch` drives a set of `BatchRater`s to the exact same
`attest.vendors.base.EnsembleRun` shape `run_ensemble` produces: the same raw
vote vectors, stamped with the same `ensemble_config_id`. Everything after
`screen` -- planes, stats, ablation, the validation record -- is unaffected
and unaware of which strategy produced the votes.

A `BatchHandle` round-trips to/from JSON via `attest.io.store.RunStore`, so a
batch job can be submitted in one process invocation and polled/fetched in a
later one. Submission is idempotent: `submit_batch` skips a vendor that
already has a persisted handle stamped with the current `ensemble_config_id`,
so retrying a partially submitted run never double-submits (and never
double-bills) a vendor.

Like `attest.vendors.base`, this module itself never makes a network call;
network access, if any, is confined to a concrete `BatchRater`'s own
`submit`/`poll`/`fetch` implementation (see `attest.vendors.providers`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from attest.contracts.input import Record
from attest.ensemble.votes import Vote, VoteVector
from attest.io.store import RunStore
from attest.provenance.config import Config, compute_ensemble_config_id
from attest.vendors.base import DeterministicRater, EnsembleRun

BatchStatus = Literal["pending", "completed", "failed"]


class BatchError(RuntimeError):
    """Raised when a batch job cannot be submitted, resumed, or assembled."""


@dataclass(frozen=True)
class BatchHandle:
    """A serializable reference to one vendor's in-flight or completed batch job.

    Round-trips to/from JSON via `to_dict`/`from_dict`, so a batch can be
    submitted in one process invocation and polled/fetched in a later one.

    Attributes:
        vendor: Vendor this batch belongs to.
        model: Model identifier used for this batch.
        provider_batch_id: The vendor's own identifier for this batch job.
        submitted_at: Timestamp the batch was submitted.
        ensemble_config_id: Ensemble configuration id this batch's eventual
            votes will be stamped with.
        id_map: Mapping of attest record id to the vendor's per-item custom
            id, used to recover the record id a fetched result belongs to.
        prompts: Mapping of attest record id to the screening prompt text
            resolved for it at submission time. Real vendor batch adapters
            leave this empty -- the vendor already baked each record's
            prompt into its submitted request, so there is nothing to
            replay at fetch time. `DeterministicBatchRater` is the
            exception: it recomputes ratings on `fetch`, in a possibly
            different process, so it needs this to reproduce the same
            per-track prompt sensitivity `DeterministicRater.rate` has.
    """

    vendor: str
    model: str
    provider_batch_id: str
    submitted_at: datetime
    ensemble_config_id: str
    id_map: Mapping[str, str]
    prompts: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this handle as a plain, JSON-serializable dict.

        `prompts` is omitted when empty, so a handle that does not use it
        serializes identically to one built before this field existed.
        """
        payload: dict[str, Any] = {
            "vendor": self.vendor,
            "model": self.model,
            "provider_batch_id": self.provider_batch_id,
            "submitted_at": self.submitted_at.isoformat(),
            "ensemble_config_id": self.ensemble_config_id,
            "id_map": dict(self.id_map),
        }
        if self.prompts:
            payload["prompts"] = dict(self.prompts)
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> BatchHandle:
        """Reconstruct a `BatchHandle` from `to_dict`'s output."""
        return BatchHandle(
            vendor=payload["vendor"],
            model=payload["model"],
            provider_batch_id=payload["provider_batch_id"],
            submitted_at=datetime.fromisoformat(payload["submitted_at"]),
            ensemble_config_id=payload["ensemble_config_id"],
            id_map=dict(payload["id_map"]),
            prompts=dict(payload.get("prompts", {})),
        )


@runtime_checkable
class BatchRater(Protocol):
    """One ensemble member's asynchronous vendor Batch API execution strategy.

    Mirrors `attest.vendors.base.Rater` but for a vendor's asynchronous Batch
    API: submit a set of requests, receive an opaque batch id, poll until the
    vendor reports completion, then retrieve results keyed by record id.
    Implementations must issue the same prompt and parse results with the
    same `attest.vendors.base.parse_ordinal_response` as their synchronous
    `Rater` twin, so sync and batch execution are interchangeable and yield
    the same ratings for the same inputs.

    Attributes:
        vendor: Name of the vendor this rater belongs to (e.g. "anthropic").
        model: Identifier of the model this rater uses.
    """

    vendor: str
    model: str

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Submit `records` as one vendor batch job.

        Args:
            records: The records to rate in this batch.
            ensemble_config_id: The ensemble configuration id this batch's
                eventual votes will be stamped with.
            prompts: Mapping of record id to the screening prompt text
                resolved for it, overriding this rater's own configured
                prompt for that record. A record with no entry (or `None`)
                uses this rater's own prompt, unchanged -- so existing
                callers that never pass `prompts` see identical behavior to
                before this parameter existed.

        Returns:
            A `BatchHandle` identifying the submitted job, ready to persist.
        """
        ...

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Check the vendor's current status for the batch `handle` refers to."""
        ...

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Retrieve completed results for the batch `handle` refers to.

        Args:
            handle: A handle for a batch this rater previously submitted,
                already reported "completed" by `poll`.

        Returns:
            Mapping of record id to `(ordinal, raw_response)`, one entry per
            record the vendor successfully rated. A record that failed
            within the vendor's batch is simply absent from the mapping --
            not an error -- so the caller records a missing vote for it
            rather than aborting the run.
        """
        ...


@dataclass
class DeterministicBatchRater:
    """A seedable, network-free `BatchRater` mirroring `DeterministicRater`.

    Simulates a vendor Batch API entirely in memory and completes
    immediately: `poll` always reports "completed", and `fetch` recomputes
    each record's rating on demand from `handle.id_map`'s own record ids and
    this rater's `seed`/`vendor`/`model` -- exactly as `DeterministicRater`
    would -- rather than from any state private to this instance. That makes
    it safe to `submit` with one instance and `fetch` with a freshly
    constructed one (e.g. across separate CLI invocations), mirroring how a
    real vendor batch is resumed across process boundaries, and guarantees
    identical ratings to its synchronous twin for the same seed.

    Attributes:
        vendor: Name to report as this rater's vendor.
        model: Name to report as this rater's model.
        seed: Seed distinguishing this rater's ratings from another
            `DeterministicBatchRater` with the same vendor and model.
        fail_record_ids: Record ids that `fetch` should omit, simulating a
            per-item failure within this vendor's batch.
        request_logprobs: Mirrors `DeterministicRater.request_logprobs` --
            forwarded to the internal `DeterministicRater` `fetch` builds,
            so a batch submitted/fetched with this set also attaches a
            deterministic fake `raw_response["logprobs"]`.
    """

    vendor: str
    model: str = "deterministic-v1"
    seed: int = 0
    fail_record_ids: frozenset[str] = field(default_factory=frozenset)
    request_logprobs: bool = False

    def submit(
        self,
        records: Sequence[Record],
        ensemble_config_id: str,
        prompts: Mapping[str, str] | None = None,
    ) -> BatchHandle:
        """Build a handle covering `records`; the simulated batch is already done."""
        id_map = {record.id: record.id for record in records}
        provider_batch_id = f"deterministic:{self.vendor}:{self.seed}:{ensemble_config_id}"
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id=provider_batch_id,
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map=id_map,
            prompts=dict(prompts) if prompts else {},
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        """Always report completion: the simulated batch finishes on submit."""
        return "completed"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        """Recompute each record's rating from `handle`'s record ids, this rater's
        seed, and (if present) the prompt resolved for that record at submit time --
        reproducing exactly what `DeterministicRater.rate` would have returned
        synchronously, so batch/sync parity holds under per-track prompts too.
        """
        rater = DeterministicRater(
            vendor=self.vendor,
            model=self.model,
            seed=self.seed,
            request_logprobs=self.request_logprobs,
        )
        results: dict[str, tuple[int, Any]] = {}
        for record_id in handle.id_map:
            if record_id in self.fail_record_ids:
                continue
            placeholder = Record(id=record_id, title="", abstract="", track="batch")
            results[record_id] = rater.rate(placeholder, prompt=handle.prompts.get(record_id))
        return results


def submit_batch(
    records: Sequence[Record],
    batch_raters: Sequence[BatchRater],
    config: Config,
    store: RunStore,
) -> str:
    """Submit one batch per rater in `batch_raters`, idempotently, persisting each handle.

    A rater is skipped -- not resubmitted -- when `store` already holds a
    handle for its vendor stamped with the current `ensemble_config_id`, so
    retrying a partially submitted run never double-submits a vendor. Each
    handle is persisted immediately after its own submission, so a crash
    partway through still leaves the earlier vendors' handles resumable.

    Args:
        records: The records to submit for rating.
        batch_raters: The ensemble members to submit a batch to.
        config: The ensemble configuration in force, used to compute the
            `ensemble_config_id` every submitted batch is stamped with, and
            -- via `config.prompt_for_track` -- to resolve each record's
            screening prompt from its `track`.
        store: The run directory to persist handles into.

    Returns:
        The `ensemble_config_id` every submitted (or already-resumed) batch
        is stamped with.

    Raises:
        BatchError: If a persisted handle exists for a vendor but is stamped
            with a different `ensemble_config_id` than `config`.
    """
    ensemble_config_id = compute_ensemble_config_id(config)
    prompts = {
        record.id: resolved
        for record in records
        if (resolved := config.prompt_for_track(record.track)) is not None
    }
    existing = {
        vendor: BatchHandle.from_dict(payload)
        for vendor, payload in store.read_batch_handles().items()
    }
    for rater in batch_raters:
        handle = existing.get(rater.vendor)
        if handle is not None:
            if handle.ensemble_config_id != ensemble_config_id:
                raise BatchError(
                    f"vendor '{rater.vendor}' already has a batch handle stamped with "
                    f"ensemble_config_id '{handle.ensemble_config_id}', which does not match "
                    f"the current configuration's id '{ensemble_config_id}'"
                )
            continue
        handle = rater.submit(records, ensemble_config_id, prompts=prompts)
        store.write_batch_handle(handle.vendor, handle.to_dict())
    return ensemble_config_id


def poll_and_fetch_batch(
    batch_raters: Sequence[BatchRater],
    store: RunStore,
    *,
    poll_interval: float = 1.0,
    max_poll_interval: float = 30.0,
    backoff_factor: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> EnsembleRun:
    """Poll every rater's persisted batch handle to completion, fetch, and assemble votes.

    Backs off geometrically between polls (`poll_interval`, multiplied by
    `backoff_factor` each round, capped at `max_poll_interval`) so a slow
    vendor batch is not polled tightly. Resumable: reads its handles from
    `store` rather than requiring the caller to hold onto them, so submit and
    fetch can be separate invocations of the process.

    Args:
        batch_raters: The ensemble members whose persisted handles to poll
            and fetch. Every rater's vendor must have a handle in `store`.
        store: The run directory holding the persisted batch handles.
        poll_interval: Seconds to wait before the first re-poll of a still-pending batch.
        max_poll_interval: Upper bound on the backed-off poll interval.
        backoff_factor: Multiplier applied to the poll interval after each pending poll.
        sleep: Sleep function, overridable in tests to avoid real waiting.

    Returns:
        An `EnsembleRun` assembled from every vendor's fetched results, with
        one `VoteVector` per record id that received at least one vote,
        ordered by record id, stamped with the batches' shared
        `ensemble_config_id`. A record id that a vendor's batch failed on is
        simply missing that vendor's `Vote` in the record's `VoteVector`.

    Raises:
        BatchError: If a rater's vendor has no persisted handle, if the
            persisted handles are stamped with more than one
            `ensemble_config_id`, or if a vendor's batch is reported failed.
    """
    raw_handles = store.read_batch_handles()
    handles: dict[str, BatchHandle] = {}
    for rater in batch_raters:
        payload = raw_handles.get(rater.vendor)
        if payload is None:
            raise BatchError(f"no persisted batch handle for vendor '{rater.vendor}'")
        handles[rater.vendor] = BatchHandle.from_dict(payload)

    ensemble_config_ids = {handle.ensemble_config_id for handle in handles.values()}
    if len(ensemble_config_ids) > 1:
        raise BatchError(
            f"persisted batch handles stamp more than one ensemble_config_id: "
            f"{sorted(ensemble_config_ids)}"
        )
    ensemble_config_id = next(iter(ensemble_config_ids))

    fetched: dict[str, dict[str, tuple[int, Any]]] = {}
    for rater in batch_raters:
        handle = handles[rater.vendor]
        interval = poll_interval
        status = rater.poll(handle)
        while status == "pending":
            sleep(interval)
            interval = min(interval * backoff_factor, max_poll_interval)
            status = rater.poll(handle)
        if status == "failed":
            raise BatchError(f"vendor '{rater.vendor}' batch '{handle.provider_batch_id}' failed")
        fetched[rater.vendor] = rater.fetch(handle)

    record_ids = sorted({record_id for handle in handles.values() for record_id in handle.id_map})
    votes: list[VoteVector] = []
    raw_responses: dict[str, dict[str, Any]] = {}
    for record_id in record_ids:
        record_votes: list[Vote] = []
        record_raw: dict[str, Any] = {}
        for rater in batch_raters:
            result = fetched[rater.vendor].get(record_id)
            if result is None:
                continue
            ordinal, raw = result
            record_votes.append(Vote(vendor=rater.vendor, rating=ordinal))
            record_raw[rater.vendor] = raw
        if not record_votes:
            continue
        votes.append(
            VoteVector(
                record_id=record_id,
                ensemble_config_id=ensemble_config_id,
                votes=tuple(record_votes),
            )
        )
        raw_responses[record_id] = record_raw

    return EnsembleRun(votes=votes, raw_responses=raw_responses)


def run_ensemble_batch(
    records: Sequence[Record],
    batch_raters: Sequence[BatchRater],
    config: Config,
    store: RunStore,
    *,
    poll_interval: float = 1.0,
    max_poll_interval: float = 30.0,
    backoff_factor: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> EnsembleRun:
    """Submit, poll to completion, fetch, and assemble one ensemble run via vendor batch APIs.

    Produces the exact same `EnsembleRun` shape as
    `attest.vendors.base.run_ensemble` -- the same raw vote vectors, stamped
    with the same `ensemble_config_id` -- but driven by each rater's
    asynchronous Batch API instead of a synchronous `Rater.rate` call per
    record. Resumable: if `store` already holds persisted handles for these
    vendors under the current configuration, submission is skipped and this
    call only polls, fetches, and assembles.

    Args:
        records: The records to rate, in the order the resulting vote
            vectors should be returned in.
        batch_raters: The ensemble members to run, in the order their votes
            are recorded into each record's vote vector.
        config: The ensemble configuration in force for this run.
        store: The run directory to persist batch handles into and resume from.
        poll_interval: Seconds to wait before the first re-poll of a still-pending batch.
        max_poll_interval: Upper bound on the backed-off poll interval.
        backoff_factor: Multiplier applied to the poll interval after each pending poll.
        sleep: Sleep function, overridable in tests to avoid real waiting.

    Returns:
        An `EnsembleRun` holding one `VoteVector` per input record that
        received at least one vote, in `records` order, and the raw,
        per-vendor responses that produced them.
    """
    submit_batch(records, batch_raters, config, store)
    assembled = poll_and_fetch_batch(
        batch_raters,
        store,
        poll_interval=poll_interval,
        max_poll_interval=max_poll_interval,
        backoff_factor=backoff_factor,
        sleep=sleep,
    )
    by_id = {vv.record_id: vv for vv in assembled.votes}
    ordered_votes = [by_id[record.id] for record in records if record.id in by_id]
    return EnsembleRun(votes=ordered_votes, raw_responses=assembled.raw_responses)


__all__ = [
    "BatchError",
    "BatchHandle",
    "BatchRater",
    "BatchStatus",
    "DeterministicBatchRater",
    "poll_and_fetch_batch",
    "run_ensemble_batch",
    "submit_batch",
]
