"""Tests for attest.provenance: config ids, epochs, changelog, and run records."""

from __future__ import annotations

from datetime import UTC, datetime

from attest.provenance.changelog import ChangeLog
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import maybe_open_epoch, open_epoch
from attest.provenance.runs import RunRecord, start_run


def _config(tau: float = 0.5) -> Config:
    return Config(
        vendors={
            "openai": VendorSpec(model="gpt-4o", model_version="2024-08-06", prompt_version="v1"),
            "anthropic": VendorSpec(
                model="claude-sonnet-5", model_version="2026-01", prompt_version="v1"
            ),
        },
        aggregation="majority",
        tau=tau,
    )


def test_same_descriptor_yields_same_ensemble_config_id() -> None:
    id_a = compute_ensemble_config_id(_config())
    id_b = compute_ensemble_config_id(_config())

    assert id_a == id_b


def test_ensemble_config_id_is_independent_of_dict_construction_order() -> None:
    reordered = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5", model_version="2026-01", prompt_version="v1"
            ),
            "openai": VendorSpec(model="gpt-4o", model_version="2024-08-06", prompt_version="v1"),
        },
        aggregation="majority",
        tau=0.5,
    )

    assert compute_ensemble_config_id(_config()) == compute_ensemble_config_id(reordered)


def test_changed_field_yields_different_id_opens_epoch_and_logs_change() -> None:
    base = _config(tau=0.5)
    changed = _config(tau=0.6)

    id_before = compute_ensemble_config_id(base)
    id_after = compute_ensemble_config_id(changed)
    assert id_before != id_after

    opened_at = datetime(2026, 1, 1, tzinfo=UTC)
    epoch_before = open_epoch(base, opened_at=opened_at)
    epoch_after = maybe_open_epoch(epoch_before, changed, opened_at=opened_at)

    assert epoch_after.id != epoch_before.id
    assert epoch_after.ensemble_config_id == id_after

    log = ChangeLog()
    event = log.record(before=id_before, after=id_after, reason="tau raised to 0.6")

    assert log.history_of(id_after) == [event]
    assert log.previous_config_id(id_after) == id_before


def test_maybe_open_epoch_reuses_current_epoch_when_config_is_unchanged() -> None:
    config = _config()
    epoch = open_epoch(config)

    same = maybe_open_epoch(epoch, config)

    assert same is epoch


def test_run_record_round_trips_to_and_from_dict() -> None:
    run = start_run(
        type="screening",
        track="track-1",
        epoch_id="epoch-1",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    run.counts["screened"] = 42
    run.finish(status="completed", ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC))

    restored = RunRecord.from_dict(run.to_dict())

    assert restored == run
