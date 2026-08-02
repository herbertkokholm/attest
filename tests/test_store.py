"""Tests for attest.io.store: run-directory persistence and validation-record assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from attest.contracts.input import ExternalId, Record
from attest.ensemble.aggregate import g
from attest.ensemble.votes import VoteVector, build_vote_vector
from attest.io.store import RunStore, StoreError, assemble_validation_record, load_input
from attest.planes.recall_audit import AuditRow
from attest.prefilter.framework import Prefilter, Prisma, require_nonempty
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import open_epoch
from attest.provenance.runs import start_run

_VENDORS = ("v1", "v2", "v3")


def _config(tau: float = 1.0) -> Config:
    return Config(
        vendors={
            vendor: VendorSpec(model="m", model_version="1", prompt_version="p")
            for vendor in _VENDORS
        },
        aggregation="boundary_dispersion",
        tau=tau,
    )


def _unanimous_votes(record_id: str, config_id: str, rating: int) -> VoteVector:
    return build_vote_vector(record_id, config_id, {vendor: rating for vendor in _VENDORS})


# --- load_input ---------------------------------------------------------------


def test_load_input_reads_and_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project": "demo",
                "records": [{"id": "r1", "title": "t", "abstract": "a", "track": 1}],
            }
        )
    )

    normalized = load_input(path)

    assert normalized.project == "demo"
    assert [r.id for r in normalized.records] == ["r1"]


def test_load_input_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        load_input(tmp_path / "missing.json")


# --- RunStore: votes and decisions round-trip ----------------------------------


def test_votes_and_decisions_round_trip(tmp_path: Path) -> None:
    config_id = compute_ensemble_config_id(_config())
    votes = [
        _unanimous_votes("r1", config_id, 1),
        _unanimous_votes("r2", config_id, -1),
    ]
    decisions = {vv.record_id: g(vv, tau=1.0) for vv in votes}

    store = RunStore(tmp_path / "run")
    store.write_votes(votes)
    store.write_decisions(config_id, decisions)

    # A fresh RunStore instance over the same directory must see the same data.
    reopened = RunStore(tmp_path / "run")

    assert reopened.read_votes() == sorted(votes, key=lambda vv: vv.record_id)
    assert reopened.read_decisions() == decisions


def test_writes_are_idempotent(tmp_path: Path) -> None:
    config_id = compute_ensemble_config_id(_config())
    votes = [_unanimous_votes("r1", config_id, 1)]

    store = RunStore(tmp_path / "run")
    store.write_votes(votes)
    path = store.root / "votes.json"
    first_write = path.read_text()

    store.write_votes(votes)
    second_write = path.read_text()

    assert first_write == second_write


def test_write_votes_upserts_by_record_id(tmp_path: Path) -> None:
    config_id = compute_ensemble_config_id(_config())
    store = RunStore(tmp_path / "run")

    store.write_votes([_unanimous_votes("r1", config_id, 1)])
    store.write_votes([_unanimous_votes("r2", config_id, -1)])

    assert {vv.record_id for vv in store.read_votes()} == {"r1", "r2"}


def test_votes_reject_mixed_ensemble_config_id(tmp_path: Path) -> None:
    votes = [
        build_vote_vector("r1", "cfg-a", {"v1": 1}),
        build_vote_vector("r2", "cfg-b", {"v1": 1}),
    ]
    store = RunStore(tmp_path / "run")

    with pytest.raises(StoreError):
        store.write_votes(votes)


def test_votes_reject_conflicting_stamp_on_second_write(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_votes([build_vote_vector("r1", "cfg-a", {"v1": 1})])

    with pytest.raises(StoreError):
        store.write_votes([build_vote_vector("r2", "cfg-b", {"v1": 1})])


# --- RunStore: config and epoch -------------------------------------------------


def test_config_round_trips(tmp_path: Path) -> None:
    config = _config()
    store = RunStore(tmp_path / "run")

    ensemble_config_id = store.write_config(config)

    assert ensemble_config_id == compute_ensemble_config_id(config)
    assert store.read_config() == config


def test_config_write_rejects_conflicting_config(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_config(_config(tau=0.1))

    with pytest.raises(StoreError):
        store.write_config(_config(tau=0.9))


def test_read_config_without_a_write_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    with pytest.raises(StoreError):
        store.read_config()


def test_epoch_round_trips(tmp_path: Path) -> None:
    epoch = open_epoch(_config(), opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    store = RunStore(tmp_path / "run")

    store.write_epoch(epoch)

    assert store.read_epoch() == epoch


# --- RunStore: raw responses -----------------------------------------------------


def test_raw_responses_round_trip(tmp_path: Path) -> None:
    responses = {
        "r1": {"v1": {"text": "1", "id": "resp-1"}, "v2": {"text": "-1", "id": "resp-2"}},
        "r2": {"v1": {"text": "0", "id": "resp-3"}},
    }
    store = RunStore(tmp_path / "run")

    store.write_raw_responses("cfg-1", responses)

    assert store.read_raw_responses() == responses


def test_raw_responses_upsert_by_record_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    store.write_raw_responses("cfg-1", {"r1": {"v1": {"text": "1"}}})
    store.write_raw_responses("cfg-1", {"r2": {"v1": {"text": "-1"}}})

    assert set(store.read_raw_responses()) == {"r1", "r2"}


def test_raw_responses_reject_conflicting_stamp(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_raw_responses("cfg-a", {"r1": {"v1": {"text": "1"}}})

    with pytest.raises(StoreError):
        store.write_raw_responses("cfg-b", {"r2": {"v1": {"text": "-1"}}})


# --- RunStore: audit rows and run records ---------------------------------------


def test_audit_rows_round_trip(tmp_path: Path) -> None:
    rows = [
        AuditRow(record_id="r1", stratum="all", human_label=1),
        AuditRow(record_id="r2", stratum="all", human_label=-1),
    ]
    store = RunStore(tmp_path / "run")

    store.write_audit_rows("cfg-1", rows)

    assert store.read_audit_rows() == rows


def test_run_records_round_trip_and_upsert(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    run = start_run(
        type="screening", track=1, epoch_id="epoch-1", started_at=datetime(2026, 1, 1, tzinfo=UTC)
    )

    store.write_run_record(run)
    run.finish(status="completed", ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC))
    store.write_run_record(run)

    restored = store.read_run_records()
    assert restored == [run]


# --- assemble_validation_record: end-to-end -------------------------------------


def test_assemble_validation_record_end_to_end(tmp_path: Path) -> None:
    records = [
        Record(id="dup-a", title="t", abstract="a", track=1, ids=[ExternalId("doi", "x")]),
        Record(id="dup-b", title="t", abstract="a", track=1, ids=[ExternalId("doi", "x")]),
        Record(id="empty", title="t", abstract="", track=1),
        Record(id="rel-1", title="t", abstract="a", track=1, gold_label=1),
        Record(id="rel-2", title="t", abstract="a", track=1, gold_label=1),
        Record(id="exc-1", title="t", abstract="a", track=1, gold_label=-1),
    ]
    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)
    assert {r.id for r in outcome.kept} == {"dup-a", "rel-1", "rel-2", "exc-1"}

    config = _config()
    epoch = open_epoch(config, opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    config_id = epoch.ensemble_config_id

    votes = [
        _unanimous_votes("dup-a", config_id, 1),
        _unanimous_votes("rel-1", config_id, 1),
        _unanimous_votes("rel-2", config_id, 1),
        _unanimous_votes("exc-1", config_id, -1),
    ]
    decisions = {
        vv.record_id: g(vv, aggregation=config.aggregation, tau=config.tau) for vv in votes
    }
    assert not any(d.escalate for d in decisions.values())

    truths = {r.id: r.gold_label for r in records if r.has_gold and r.gold_label is not None}
    audit_rows = [AuditRow(record_id="exc-1", stratum="all", human_label=-1)]

    store = RunStore(tmp_path / "run")
    store.write_config(config)
    store.write_epoch(epoch)
    store.write_votes(votes)
    store.write_decisions(config_id, decisions)
    store.write_audit_rows(config_id, audit_rows)

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes={"all": 1},
    )

    prisma = record.prisma
    assert prisma.identified == 6
    assert prisma.duplicates_removed == 1
    assert prisma.after_dedup == 5
    assert prisma.prefilter_excluded == 1
    assert prisma.screened == 4
    assert prisma.screen_excluded == 1
    assert prisma.included == 3
    assert prisma.screen_excluded + prisma.included == prisma.screened

    assert record.confusion["tp"] == 2
    assert record.escalation_rate == pytest.approx(0.0)

    assert record.recall.point is not None
    assert record.recall.floor is not None
    assert record.recall.floor <= record.recall.point
    assert record.recall.audit_n == 1


def test_assemble_validation_record_requires_stored_votes(tmp_path: Path) -> None:
    config = _config()
    epoch = open_epoch(config, opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    store = RunStore(tmp_path / "run")
    store.write_config(config)
    store.write_epoch(epoch)
    empty_prisma = Prisma(
        identified=0, duplicates_removed=0, after_dedup=0, prefilter_excluded=0, passed=0
    )

    with pytest.raises(StoreError):
        assemble_validation_record(
            store, prefilter_prisma=empty_prisma, truths={}, population_sizes={}
        )
