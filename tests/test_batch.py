"""Tests for attest.vendors.batch: BatchRater, DeterministicBatchRater, run_ensemble_batch."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attest.contracts.input import Record
from attest.io.store import RunStore
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.vendors.base import DeterministicRater, run_ensemble
from attest.vendors.batch import (
    BatchError,
    BatchHandle,
    BatchRater,
    BatchStatus,
    DeterministicBatchRater,
    poll_and_fetch_batch,
    run_ensemble_batch,
    submit_batch,
)


def _records(*ids: str) -> list[Record]:
    return [Record(id=record_id, title="title", abstract="abstract", track=1) for record_id in ids]


def _config() -> Config:
    return Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "vendor-b": VendorSpec(
                model="model-b", model_version="v1", prompt_version="p1", temperature=0.0
            ),
        },
        aggregation="boundary_dispersion",
        tau=1.0,
    )


def _sync_raters(seed_a: int = 1, seed_b: int = 2) -> list[DeterministicRater]:
    return [
        DeterministicRater(vendor="vendor-a", model="model-a", seed=seed_a),
        DeterministicRater(vendor="vendor-b", model="model-b", seed=seed_b),
    ]


def _batch_raters(seed_a: int = 1, seed_b: int = 2) -> list[DeterministicBatchRater]:
    return [
        DeterministicBatchRater(vendor="vendor-a", model="model-a", seed=seed_a),
        DeterministicBatchRater(vendor="vendor-b", model="model-b", seed=seed_b),
    ]


class _CountingBatchRater:
    """Wraps a `BatchRater`, counting `submit` calls, to test submission idempotency."""

    def __init__(self, inner: BatchRater) -> None:
        self.inner = inner
        self.submit_calls = 0

    @property
    def vendor(self) -> str:
        return self.inner.vendor

    @property
    def model(self) -> str:
        return self.inner.model

    def submit(
        self, records: Any, ensemble_config_id: str, prompts: Mapping[str, str] | None = None
    ) -> BatchHandle:
        self.submit_calls += 1
        return self.inner.submit(records, ensemble_config_id, prompts=prompts)

    def poll(self, handle: BatchHandle) -> BatchStatus:
        return self.inner.poll(handle)

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        return self.inner.fetch(handle)


class _ScriptedPollRater:
    """A `BatchRater` whose `poll` reports pending a fixed number of times before completing."""

    def __init__(self, inner: BatchRater, pending_polls: int) -> None:
        self.inner = inner
        self.pending_polls = pending_polls
        self.poll_calls = 0

    @property
    def vendor(self) -> str:
        return self.inner.vendor

    @property
    def model(self) -> str:
        return self.inner.model

    def submit(
        self, records: Any, ensemble_config_id: str, prompts: Mapping[str, str] | None = None
    ) -> BatchHandle:
        return self.inner.submit(records, ensemble_config_id, prompts=prompts)

    def poll(self, handle: BatchHandle) -> BatchStatus:
        self.poll_calls += 1
        if self.poll_calls <= self.pending_polls:
            return "pending"
        return self.inner.poll(handle)

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        return self.inner.fetch(handle)


class _AlwaysFailingRater:
    """A `BatchRater` whose batch is always reported failed."""

    vendor = "vendor-a"
    model = "model-a"

    def submit(
        self, records: Any, ensemble_config_id: str, prompts: Mapping[str, str] | None = None
    ) -> BatchHandle:
        return BatchHandle(
            vendor=self.vendor,
            model=self.model,
            provider_batch_id="doomed",
            submitted_at=datetime.now(UTC),
            ensemble_config_id=ensemble_config_id,
            id_map={},
        )

    def poll(self, handle: BatchHandle) -> BatchStatus:
        return "failed"

    def fetch(self, handle: BatchHandle) -> dict[str, tuple[int, Any]]:
        raise AssertionError("fetch should not be called for a failed batch")


# --- equivalence ---------------------------------------------------------------


def test_run_ensemble_batch_matches_run_ensemble_for_same_seed(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1", "rec-2", "rec-3")

    sync_run = run_ensemble(records, _sync_raters(), config)
    store = RunStore(tmp_path / "run")
    batch_run = run_ensemble_batch(records, _batch_raters(), config, store)

    assert batch_run.votes == sync_run.votes
    assert [v.record_id for v in batch_run.votes] == [v.record_id for v in sync_run.votes]


def test_run_ensemble_batch_matches_run_ensemble_under_per_track_prompts(tmp_path: Path) -> None:
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "vendor-b": VendorSpec(
                model="model-b", model_version="v1", prompt_version="p1", temperature=0.0
            ),
        },
        aggregation="boundary_dispersion",
        tau=1.0,
        track_prompts={"review-a": "criteria A", "review-b": "criteria B"},
    )
    records = [
        Record(id="rec-1", title="title", abstract="abstract", track="review-a"),
        Record(id="rec-2", title="title", abstract="abstract", track="review-b"),
    ]

    sync_run = run_ensemble(records, _sync_raters(), config)
    store = RunStore(tmp_path / "run")
    batch_run = run_ensemble_batch(records, _batch_raters(), config, store)

    assert batch_run.votes == sync_run.votes


def test_deterministic_batch_rater_forwards_request_logprobs_through_fetch(
    tmp_path: Path,
) -> None:
    rater = DeterministicBatchRater(vendor="openai", model="model-a", seed=1, request_logprobs=True)
    handle = rater.submit(_records("rec-1"), ensemble_config_id="cfg")

    fetched = rater.fetch(handle)

    ordinal, raw = fetched["rec-1"]
    assert "logprobs" in raw
    sync_ordinal, sync_raw = DeterministicRater(
        vendor="openai", model="model-a", seed=1, request_logprobs=True
    ).rate(Record(id="rec-1", title="", abstract="", track="batch"))
    assert ordinal == sync_ordinal
    assert raw == sync_raw


def test_deterministic_batch_rater_omits_logprobs_by_default(tmp_path: Path) -> None:
    rater = DeterministicBatchRater(vendor="openai", model="model-a", seed=1)
    handle = rater.submit(_records("rec-1"), ensemble_config_id="cfg")

    _ordinal, raw = rater.fetch(handle)["rec-1"]

    assert "logprobs" not in raw


def test_run_ensemble_batch_stamps_the_same_ensemble_config_id(tmp_path: Path) -> None:
    config = _config()
    expected_config_id = compute_ensemble_config_id(config)
    store = RunStore(tmp_path / "run")

    run = run_ensemble_batch(_records("rec-1"), _batch_raters(), config, store)

    assert all(vv.ensemble_config_id == expected_config_id for vv in run.votes)


# --- resumability ----------------------------------------------------------------


def test_batch_handle_round_trips_through_json() -> None:
    handle = BatchHandle(
        vendor="vendor-a",
        model="model-a",
        provider_batch_id="batch-123",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        ensemble_config_id="cfg-id",
        id_map={"rec-1": "item-0", "rec-2": "item-1"},
    )

    restored = BatchHandle.from_dict(handle.to_dict())

    assert restored == handle


def test_submit_then_fetch_across_two_calls_matches_single_call(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1", "rec-2", "rec-3")

    single_store = RunStore(tmp_path / "single")
    single_run = run_ensemble_batch(records, _batch_raters(), config, single_store)

    two_call_store = RunStore(tmp_path / "two-call")
    submit_batch(records, _batch_raters(), config, two_call_store)
    # A fresh set of rater instances, as a later process invocation would construct.
    resumed_run = poll_and_fetch_batch(_batch_raters(), two_call_store)

    assert resumed_run.votes == single_run.votes


# --- idempotent submit -------------------------------------------------------------


def test_idempotent_submit_does_not_resubmit(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1", "rec-2")
    store = RunStore(tmp_path / "run")
    raters = [_CountingBatchRater(r) for r in _batch_raters()]

    submit_batch(records, raters, config, store)
    submit_batch(records, raters, config, store)

    assert all(r.submit_calls == 1 for r in raters)


def test_submit_batch_raises_on_ensemble_config_id_conflict(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    records = _records("rec-1")
    submit_batch(records, _batch_raters(), _config(), store)

    other_config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v2", prompt_version="p1", temperature=0.0
            ),
            "vendor-b": VendorSpec(
                model="model-b", model_version="v1", prompt_version="p1", temperature=0.0
            ),
        },
        aggregation="boundary_dispersion",
        tau=1.0,
    )

    try:
        submit_batch(records, _batch_raters(), other_config, store)
    except BatchError:
        pass
    else:
        raise AssertionError("expected BatchError for a mismatched ensemble_config_id")


# --- per-item failure ---------------------------------------------------------------


def test_per_item_failure_yields_missing_vote_not_abort(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1", "rec-2", "rec-3")
    store = RunStore(tmp_path / "run")
    raters: list[BatchRater] = [
        DeterministicBatchRater(
            vendor="vendor-a", model="model-a", seed=1, fail_record_ids=frozenset({"rec-2"})
        ),
        DeterministicBatchRater(vendor="vendor-b", model="model-b", seed=2),
    ]

    run = run_ensemble_batch(records, raters, config, store)

    assert [v.record_id for v in run.votes] == ["rec-1", "rec-2", "rec-3"]
    by_id = {v.record_id: v for v in run.votes}
    assert {vote.vendor for vote in by_id["rec-2"].votes} == {"vendor-b"}
    assert {vote.vendor for vote in by_id["rec-1"].votes} == {"vendor-a", "vendor-b"}
    assert {vote.vendor for vote in by_id["rec-3"].votes} == {"vendor-a", "vendor-b"}


def test_poll_and_fetch_batch_raises_on_failed_batch(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1")
    store = RunStore(tmp_path / "run")
    rater = _AlwaysFailingRater()

    submit_batch(records, [rater], config, store)

    try:
        poll_and_fetch_batch([rater], store, sleep=lambda _seconds: None)
    except BatchError:
        pass
    else:
        raise AssertionError("expected BatchError for a failed batch")


def test_poll_and_fetch_batch_backs_off_until_completion(tmp_path: Path) -> None:
    config = _config()
    records = _records("rec-1")
    store = RunStore(tmp_path / "run")
    scripted = _ScriptedPollRater(
        DeterministicBatchRater(vendor="vendor-a", seed=1), pending_polls=3
    )

    submit_batch(records, [scripted], config, store)

    slept: list[float] = []
    run = poll_and_fetch_batch(
        [scripted], store, poll_interval=1.0, backoff_factor=2.0, sleep=slept.append
    )

    assert slept == [1.0, 2.0, 4.0]
    assert [v.record_id for v in run.votes] == ["rec-1"]


# --- import safety ---------------------------------------------------------------


def test_import_attest_batch_with_no_extras_installed_still_succeeds() -> None:
    for module_name in (
        "attest",
        "attest.vendors.batch",
        "attest.vendors.registry",
        "attest.vendors.providers.anthropic",
        "attest.vendors.providers.openai",
        "attest.vendors.providers.google",
        "attest.vendors.providers.mistral",
    ):
        importlib.import_module(module_name)
