"""Tests for attest.io.store: run-directory persistence and validation-record assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from attest.contracts.input import ExternalId, Record
from attest.ensemble.aggregate import g
from attest.ensemble.confidence import RecordConfidence
from attest.ensemble.tau import describe_tau
from attest.ensemble.votes import VoteVector, build_vote_vector
from attest.io.store import RunStore, StoreError, assemble_validation_record, load_input
from attest.planes.active_learning import ActiveLearningReview, ActiveLearningSelection
from attest.planes.adjudication import AdjudicationItem
from attest.planes.inclusion_audit import InclusionAuditRow
from attest.planes.recall_audit import AuditRow
from attest.prefilter.framework import Prefilter, Prisma, require_nonempty
from attest.provenance.changelog import ChangeLog, ConfigChangeEvent
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import open_epoch
from attest.provenance.protocol import ValidationProtocol
from attest.provenance.runs import start_run
from attest.provenance.sentinel import (
    SentinelBaseline,
    capture_baseline,
    evaluate_sentinel,
)

_VENDORS = ("v1", "v2", "v3")


def _config(tau: float = 1.0) -> Config:
    return Config(
        vendors={
            vendor: VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0)
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


def test_config_round_trips_with_track_prompts(tmp_path: Path) -> None:
    config = Config(
        vendors={
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0)
        },
        aggregation="boundary_dispersion",
        tau=1.0,
        default_prompt="generic default",
        track_prompts={"review-a": "review A criteria"},
    )
    store = RunStore(tmp_path / "run")

    store.write_config(config)

    assert store.read_config() == config


def test_config_round_trips_with_zero_policy(tmp_path: Path) -> None:
    config = Config(
        vendors={
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0)
        },
        aggregation="boundary_dispersion",
        tau=1.0,
        zero_policy="include",
    )
    store = RunStore(tmp_path / "run")

    store.write_config(config)

    assert store.read_config() == config


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


# --- RunStore: tau report ---------------------------------------------------------


def test_tau_report_round_trips(tmp_path: Path) -> None:
    report = describe_tau(0.3, 4)
    store = RunStore(tmp_path / "run")

    store.write_tau_report(report)

    assert store.read_tau_report() == report


def test_tau_report_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_tau_report() is None


def test_tau_report_write_overwrites_previous_report(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    store.write_tau_report(describe_tau(0.3, 4))
    store.write_tau_report(describe_tau(0.6, 4))

    assert store.read_tau_report() == describe_tau(0.6, 4)


def test_config_round_trips_with_confidence_threshold(tmp_path: Path) -> None:
    config = Config(
        vendors={
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0)
        },
        aggregation="boundary_dispersion",
        tau=1.0,
        confidence_threshold=0.7,
    )
    store = RunStore(tmp_path / "run")

    ensemble_config_id = store.write_config(config)

    # Round-trips through the store despite never appearing in
    # Config.to_dict()/the hash -- see _config_to_dict.
    assert store.read_config() == config
    assert ensemble_config_id == compute_ensemble_config_id(config)


def test_confidence_policy_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    store.write_confidence_policy(low_threshold=0.4, min_supporting_votes=3)

    assert store.read_confidence_policy() == {"low_threshold": 0.4, "min_supporting_votes": 3}


def test_confidence_policy_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_confidence_policy() is None


def test_confidence_policy_write_overwrites_previous_policy(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    store.write_confidence_policy(low_threshold=0.3, min_supporting_votes=3)
    store.write_confidence_policy(low_threshold=0.6, min_supporting_votes=3)

    assert store.read_confidence_policy() == {"low_threshold": 0.6, "min_supporting_votes": 3}


# --- RunStore: audit rows and run records ---------------------------------------


def test_audit_rows_round_trip(tmp_path: Path) -> None:
    rows = [
        AuditRow(record_id="r1", stratum="all", human_label=1),
        AuditRow(record_id="r2", stratum="all", human_label=-1),
    ]
    store = RunStore(tmp_path / "run")

    store.write_audit_rows("cfg-1", rows)

    assert store.read_audit_rows() == rows


def test_inclusion_audit_rows_round_trip(tmp_path: Path) -> None:
    rows = [
        InclusionAuditRow(record_id="r1", stratum="all", human_label=1),
        InclusionAuditRow(record_id="r2", stratum="all", human_label=-1),
    ]
    store = RunStore(tmp_path / "run")

    store.write_inclusion_audit_rows("cfg-1", rows)

    assert store.read_inclusion_audit_rows() == rows


def test_inclusion_audit_rows_are_stored_separately_from_recall_audit_rows(tmp_path: Path) -> None:
    # The two audit planes must never share a file: mixing them would let a
    # TP-side row silently contribute to the FN-side estimate or vice versa.
    exclusion_rows = [AuditRow(record_id="r1", stratum="all", human_label=-1)]
    inclusion_rows = [InclusionAuditRow(record_id="r1", stratum="all", human_label=1)]
    store = RunStore(tmp_path / "run")

    store.write_audit_rows("cfg-1", exclusion_rows)
    store.write_inclusion_audit_rows("cfg-1", inclusion_rows)

    assert store.read_audit_rows() == exclusion_rows
    assert store.read_inclusion_audit_rows() == inclusion_rows


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
    assert record.recall.exact_floor is not None
    assert record.recall.exact_floor <= record.recall.point
    assert record.recall.audit_n == 1
    assert record.unresolved_escalations == 0

    # Every vendor voted unanimously and correctly on both relevant records
    # (rel-1, rel-2), so every pair's error indicators have zero variance --
    # correlation is undefined for all three pairs. Before this was fixed,
    # an undefined pair was silently dropped from the record entirely; now
    # every pair is still present, with its joint counts standing in.
    correlations = record.error_correlation.pairwise_fn_on_relevant
    assert set(correlations) == {"v1|v2", "v1|v3", "v2|v3"}
    for pair in correlations.values():
        assert pair.correlation is None
        assert (pair.n, pair.both, pair.only_a, pair.only_b, pair.neither) == (2, 0, 0, 0, 2)


def test_tp_full_review_used_without_inclusion_audit_rows(
    tmp_path: Path,
) -> None:
    # Sanity check for the branch condition itself: passing
    # inclusion_population_sizes with no stored, labeled inclusion-audit
    # rows must not change anything -- confirms the audited-TP path is
    # opt-in via stored rows, not merely via the parameter being present.
    records = [
        Record(id="rel-1", title="t", abstract="a", track=1, gold_label=1),
        Record(id="rel-2", title="t", abstract="a", track=1, gold_label=1),
        Record(id="exc-1", title="t", abstract="a", track=1, gold_label=-1),
    ]
    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)

    config = _config()
    epoch = open_epoch(config, opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    config_id = epoch.ensemble_config_id

    votes = [
        _unanimous_votes("rel-1", config_id, 1),
        _unanimous_votes("rel-2", config_id, 1),
        _unanimous_votes("exc-1", config_id, -1),
    ]
    decisions = {
        vv.record_id: g(vv, aggregation=config.aggregation, tau=config.tau) for vv in votes
    }
    truths = {r.id: r.gold_label for r in records if r.has_gold and r.gold_label is not None}

    store = RunStore(tmp_path / "run")
    store.write_config(config)
    store.write_epoch(epoch)
    store.write_votes(votes)
    store.write_decisions(config_id, decisions)
    store.write_audit_rows(config_id, [AuditRow(record_id="exc-1", stratum="all", human_label=-1)])
    # Deliberately no write_inclusion_audit_rows call.

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes={"all": 1},
        inclusion_population_sizes={"all": 2},
    )

    assert record.recall.tp_estimation_method == "full_review"
    assert record.recall.estimated_true_positives is None
    assert record.confusion["tp"] == 2


def test_tp_audited_estimate_used_with_inclusion_audit_rows(
    tmp_path: Path,
) -> None:
    records = [
        Record(id="rel-1", title="t", abstract="a", track=1, gold_label=1),
        Record(id="rel-2", title="t", abstract="a", track=1, gold_label=1),
        Record(id="exc-1", title="t", abstract="a", track=1, gold_label=-1),
    ]
    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)

    config = _config()
    epoch = open_epoch(config, opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    config_id = epoch.ensemble_config_id

    votes = [
        _unanimous_votes("rel-1", config_id, 1),
        _unanimous_votes("rel-2", config_id, 1),
        _unanimous_votes("exc-1", config_id, -1),
    ]
    decisions = {
        vv.record_id: g(vv, aggregation=config.aggregation, tau=config.tau) for vv in votes
    }
    truths = {r.id: r.gold_label for r in records if r.has_gold and r.gold_label is not None}

    store = RunStore(tmp_path / "run")
    store.write_config(config)
    store.write_epoch(epoch)
    store.write_votes(votes)
    store.write_decisions(config_id, decisions)
    store.write_audit_rows(config_id, [AuditRow(record_id="exc-1", stratum="all", human_label=-1)])
    # A (trivially exhaustive, n == population) inclusion audit over both
    # included records, both truly relevant -- exercises the audited-TP
    # branch without needing a partial-coverage scaling scenario.
    store.write_inclusion_audit_rows(
        config_id,
        [
            InclusionAuditRow(record_id="rel-1", stratum="all", human_label=1),
            InclusionAuditRow(record_id="rel-2", stratum="all", human_label=1),
        ],
    )

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes={"all": 1},
        inclusion_population_sizes={"all": 2},
    )

    assert record.recall.tp_estimation_method == "inclusion_audit"
    assert record.recall.estimated_true_positives == pytest.approx(2.0)
    # confusion["tp"] is still computed from truths as always (it is not
    # recall's TP source here, but remains available/unaffected).
    assert record.confusion["tp"] == 2
    assert record.recall.exact_floor is not None
    assert record.recall.exact_floor <= record.recall.point


def test_assemble_validation_record_fails_closed_on_unresolved_escalation(tmp_path: Path) -> None:
    records = [
        Record(id="rel-1", title="t", abstract="a", track=1, gold_label=1),
        Record(id="tie-1", title="t", abstract="a", track=1, gold_label=1),
    ]
    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)

    config = _config()
    epoch = open_epoch(config, opened_at=datetime(2026, 1, 1, tzinfo=UTC))
    config_id = epoch.ensemble_config_id

    votes = [
        _unanimous_votes("rel-1", config_id, 1),
        _unanimous_votes("tie-1", config_id, 0),  # mean-zero vote: escalates under
        # the default zero_policy="escalate", regardless of tau.
    ]
    decisions = {
        vv.record_id: g(vv, aggregation=config.aggregation, tau=config.tau) for vv in votes
    }
    assert decisions["tie-1"].escalate

    truths = {r.id: r.gold_label for r in records if r.has_gold and r.gold_label is not None}

    store = RunStore(tmp_path / "run")
    store.write_config(config)
    store.write_epoch(epoch)
    store.write_votes(votes)
    store.write_decisions(config_id, decisions)

    with pytest.raises(StoreError, match="tie-1"):
        assemble_validation_record(
            store, prefilter_prisma=outcome.prisma, truths=truths, population_sizes={}
        )

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes={},
        allow_unresolved_escalations=True,
    )
    assert record.unresolved_escalations == 1
    # tie-1 is excluded from confusion/PRISMA counts, exactly as before this
    # was surfaced -- only now the omission is visible in the record itself.
    # rel-1 (unanimous include, gold-relevant) is still resolved and correct.
    assert record.confusion == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert record.prisma.included == 1
    assert record.prisma.screen_excluded == 0


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


# --- RunStore: changelog (append-only run artifact) -------------------------------


def test_changelog_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    log = ChangeLog()
    log.record(before=None, after="cfg-1", reason="initial")
    log.record(before="cfg-1", after="cfg-2", reason="tau raised")

    store.write_changelog(log)

    assert store.read_changelog() == log


def test_append_change_event_extends_stored_history(tmp_path: Path) -> None:
    from attest.provenance.changelog import CHANGE_TYPE_INITIAL

    store = RunStore(tmp_path / "run")
    first = store.append_change_event(
        ConfigChangeEvent(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            before=None,
            after="cfg-1",
            reason="initial",
            change_type=CHANGE_TYPE_INITIAL,
        )
    )
    assert first.events == store.read_changelog().events

    store.append_change_event(
        ConfigChangeEvent(
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            before="cfg-1",
            after="cfg-2",
            reason="explicit change",
        )
    )

    assert len(store.read_changelog().events) == 2


def test_write_changelog_rejects_shrinking_stored_history(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    full_log = ChangeLog()
    full_log.record(before=None, after="cfg-1", reason="initial")
    full_log.record(before="cfg-1", after="cfg-2", reason="explicit change")
    store.write_changelog(full_log)

    shrunk = ChangeLog(events=full_log.events[:1])
    with pytest.raises(StoreError):
        store.write_changelog(shrunk)


def test_write_changelog_rejects_rewriting_a_stored_event(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    log = ChangeLog()
    log.record(
        before=None, after="cfg-1", reason="initial", timestamp=datetime(2026, 1, 1, tzinfo=UTC)
    )
    store.write_changelog(log)

    rewritten = ChangeLog()
    rewritten.record(
        before=None,
        after="cfg-1",
        reason="a different reason for the same event",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(StoreError):
        store.write_changelog(rewritten)


def test_read_changelog_without_a_write_returns_empty_log(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_changelog() == ChangeLog()


# --- RunStore: validation protocol -------------------------------------------------


def test_protocol_round_trips_and_returns_stable_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    protocol = ValidationProtocol()

    protocol_id = store.write_protocol(protocol)

    assert store.read_protocol() == protocol
    assert store.read_protocol_id() == protocol_id


def test_protocol_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_protocol() is None
    assert store.read_protocol_id() is None


def test_protocol_can_be_overwritten_freely(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_protocol(ValidationProtocol())
    from attest.provenance.protocol import AuditDesign

    updated_id = store.write_protocol(
        ValidationProtocol(audit_design=AuditDesign(stratify_by="track"))
    )

    assert store.read_protocol_id() == updated_id
    assert store.read_protocol().audit_design.stratify_by == "track"  # type: ignore[union-attr]


# --- RunStore: audit draw / audit labels snapshots ---------------------------------


def test_audit_draw_and_labels_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    rows = [
        AuditRow(record_id="r1", stratum="all"),
        AuditRow(record_id="r2", stratum="all"),
    ]

    store.write_audit_draw("cfg-1", rows)
    store.write_audit_labels("cfg-1", {"r1": 1, "r2": -1})

    assert store.read_audit_draw() == {"r1": "all", "r2": "all"}
    assert store.read_audit_labels() == {"r1": 1, "r2": -1}


def test_audit_draw_is_distinct_from_audit_json(tmp_path: Path) -> None:
    # write_audit_rows (the existing combined file validate reads from) does
    # not implicitly populate the audit_draw snapshot -- they are separate
    # artifacts, written by separate calls.
    store = RunStore(tmp_path / "run")
    store.write_audit_rows("cfg-1", [AuditRow(record_id="r1", stratum="all")])

    assert store.read_audit_draw() == {}


def test_audit_draw_persists_sampling_frame_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    rows = [AuditRow(record_id="r1", stratum="all")]

    store.write_audit_draw("cfg-1", rows, sampling_frame_hash="deadbeef")

    assert store.read_audit_sampling_frame_hash() == "deadbeef"


def test_audit_draw_sampling_frame_hash_defaults_to_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_audit_sampling_frame_hash() is None

    store.write_audit_draw("cfg-1", [AuditRow(record_id="r1", stratum="all")])

    assert store.read_audit_sampling_frame_hash() is None


def test_audit_draw_rejects_a_conflicting_sampling_frame_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_audit_draw(
        "cfg-1", [AuditRow(record_id="r1", stratum="all")], sampling_frame_hash="first"
    )

    with pytest.raises(StoreError, match="sampling_frame_hash"):
        store.write_audit_draw(
            "cfg-1", [AuditRow(record_id="r2", stratum="all")], sampling_frame_hash="second"
        )


def test_audit_draw_repeated_calls_with_the_same_hash_are_fine(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_audit_draw(
        "cfg-1", [AuditRow(record_id="r1", stratum="all")], sampling_frame_hash="same"
    )
    store.write_audit_draw(
        "cfg-1", [AuditRow(record_id="r2", stratum="all")], sampling_frame_hash="same"
    )

    assert store.read_audit_sampling_frame_hash() == "same"
    assert store.read_audit_draw() == {"r1": "all", "r2": "all"}


def test_inclusion_audit_draw_persists_sampling_frame_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    rows = [InclusionAuditRow(record_id="r1", stratum="all")]

    store.write_inclusion_audit_draw("cfg-1", rows, sampling_frame_hash="deadbeef")

    assert store.read_inclusion_audit_sampling_frame_hash() == "deadbeef"


def test_inclusion_audit_draw_sampling_frame_hash_defaults_to_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_inclusion_audit_sampling_frame_hash() is None

    store.write_inclusion_audit_draw("cfg-1", [InclusionAuditRow(record_id="r1", stratum="all")])

    assert store.read_inclusion_audit_sampling_frame_hash() is None


def test_inclusion_audit_draw_rejects_a_conflicting_sampling_frame_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_inclusion_audit_draw(
        "cfg-1", [InclusionAuditRow(record_id="r1", stratum="all")], sampling_frame_hash="first"
    )

    with pytest.raises(StoreError, match="sampling_frame_hash"):
        store.write_inclusion_audit_draw(
            "cfg-1",
            [InclusionAuditRow(record_id="r2", stratum="all")],
            sampling_frame_hash="second",
        )


def test_audit_rows_round_trip_reviewer_and_blinded(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    row = AuditRow(record_id="r1", stratum="all", human_label=1, reviewer="auditor-a", blinded=True)

    store.write_audit_rows("cfg-1", [row])

    assert store.read_audit_rows() == [row]


# --- RunStore: validation record snapshot -------------------------------------------


def test_validation_record_snapshot_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    payload = {"schema_version": "1.1", "ensemble_config_id": "cfg-1"}

    store.write_validation_record_snapshot(payload)

    assert store.read_validation_record_snapshot() == payload


def test_validation_record_snapshot_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_validation_record_snapshot() is None


# --- RunStore: sentinel baseline and evaluations -----------------------------------


def _sentinel_baseline() -> SentinelBaseline:
    return capture_baseline(
        "sentinel-set-1",
        "cfg-1",
        "epoch-1",
        {"v1": {"s0": -1, "s1": 0, "s2": 1}},
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_sentinel_baseline_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    baseline = _sentinel_baseline()

    store.write_sentinel_baseline(baseline)

    assert store.read_sentinel_baseline() == baseline


def test_sentinel_baseline_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_sentinel_baseline() is None


def test_sentinel_baseline_rejects_a_different_sentinel_set(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_sentinel_baseline(_sentinel_baseline())

    other = capture_baseline("sentinel-set-2", "cfg-1", "epoch-1", {"v1": {"s0": -1}})
    with pytest.raises(StoreError):
        store.write_sentinel_baseline(other)


def test_sentinel_evaluations_accumulate_history(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    baseline = _sentinel_baseline()
    store.write_sentinel_baseline(baseline)

    first = evaluate_sentinel(
        baseline,
        {"v1": {"s0": -1, "s1": 0, "s2": 1}},
        evaluated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    second = evaluate_sentinel(
        baseline,
        {"v1": {"s0": -1, "s1": 0, "s2": -1}},
        evaluated_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    store.write_sentinel_evaluation(first)
    store.write_sentinel_evaluation(second)

    restored = store.read_sentinel_evaluations()
    assert restored == [first, second]


def test_sentinel_evaluations_read_without_a_write_returns_empty_list(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_sentinel_evaluations() == []


# --- RunStore: adjudication provenance ----------------------------------------------


def test_adjudication_record_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    item = AdjudicationItem(
        record_id="r1",
        ensemble_config_id="cfg-1",
        dispersion=1.0,
        boundary=True,
        selection_reason="boundary",
        human_label=1,
        reviewer="reviewer-a",
        resolved_at=datetime(2026, 1, 1, tzinfo=UTC),
        protocol_id="proto-1",
    )

    store.write_adjudication_record(item)

    records = store.read_adjudication_records()
    assert records["r1"]["human_label"] == 1
    assert records["r1"]["reviewer"] == "reviewer-a"
    assert records["r1"]["selection_reason"] == "boundary"
    assert records["r1"]["protocol_id"] == "proto-1"
    assert records["r1"]["resolved_at"] == "2026-01-01T00:00:00+00:00"


def test_adjudication_records_upsert_by_record_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_adjudication_record(
        AdjudicationItem(record_id="r1", ensemble_config_id="cfg-1", dispersion=1.0, boundary=True)
    )
    store.write_adjudication_record(
        AdjudicationItem(record_id="r2", ensemble_config_id="cfg-1", dispersion=0.5, boundary=False)
    )

    assert set(store.read_adjudication_records()) == {"r1", "r2"}


# --- RunStore: active-learning provenance -------------------------------------------


def test_active_learning_selections_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    selection = ActiveLearningSelection(
        record_id="r1",
        ensemble_config_id="cfg-1",
        dispersion=1.0,
        boundary=True,
        confidence=RecordConfidence(
            record_id="r1", median_probability=0.4, n_supporting=3, n_total=3, scored=True
        ),
        selection_reason=("boundary", "low_confidence"),
    )

    store.write_active_learning_selections("cfg-1", [selection])

    stored = store.read_active_learning_selections()
    assert stored["r1"]["selection_reason"] == ["boundary", "low_confidence"]
    assert stored["r1"]["confidence_median_probability"] == pytest.approx(0.4)


def test_active_learning_reviews_accumulate_history(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    review = ActiveLearningReview(
        record_id="r1",
        ensemble_config_id="cfg-1",
        selection_reason=("dispersion",),
        reviewer="reviewer-a",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
        protocol_id=None,
        notes="looks like a prompt gap",
    )

    store.write_active_learning_review(review)

    reviews = store.read_active_learning_reviews()
    assert len(reviews) == 1
    assert reviews[0]["reviewer"] == "reviewer-a"
    assert reviews[0]["notes"] == "looks like a prompt gap"


# --- RunStore: run manifest and offline integrity verification ---------------------


def test_manifest_hashes_existing_artifacts_and_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config()
    store.write_config(config)
    config_id = compute_ensemble_config_id(config)

    manifest = store.write_manifest(
        ensemble_config_id=config_id,
        input_hash="deadbeef",
        input_source="gold.json",
        seeds={"screen": 1},
    )

    assert manifest.artifact_hashes  # at least config.json was hashed
    assert store.read_manifest() == manifest


def test_manifest_read_without_a_write_returns_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    assert store.read_manifest() is None


def test_verify_raises_without_a_stored_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")

    with pytest.raises(StoreError):
        store.verify()


def test_verify_ok_when_artifacts_are_unchanged(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config()
    store.write_config(config)
    store.write_manifest(ensemble_config_id=compute_ensemble_config_id(config))

    report = store.verify()

    assert report.ok is True


def test_verify_detects_tampered_config(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config()
    store.write_config(config)
    store.write_manifest(ensemble_config_id=compute_ensemble_config_id(config))

    # Simulate tampering: overwrite config.json directly on disk, bypassing the store.
    (store.root / "config.json").write_text('{"tampered": true}', encoding="utf-8")

    report = store.verify()

    assert report.ok is False
    assert any(p.artifact == "config.json" for p in report.problems)
