"""Tests for attest.planes: adjudication, active learning, and the recall-audit firewall."""

from __future__ import annotations

import random

import pytest

from attest.ensemble.aggregate import g
from attest.ensemble.votes import build_vote_vector
from attest.planes import PLANE_ACTIVE_LEARNING, PLANE_ADJUDICATION, PLANE_RECALL_AUDIT
from attest.planes.active_learning import (
    ActiveLearningError,
    select_for_review,
)
from attest.planes.adjudication import (
    AdjudicationError,
    AdjudicationQueue,
    final_label,
)
from attest.planes.recall_audit import (
    AuditError,
    AuditRow,
    ExcludedRecord,
    build_strata,
    draw_audit_sample,
    ingest_audit_labels,
)
from attest.stats.recall import stratified_recall

_CONFIG_ID = "config-abc"


def _votes(record_id: str, ratings: tuple[int, ...]):
    return build_vote_vector(
        record_id, _CONFIG_ID, {f"vendor{i}": r for i, r in enumerate(ratings)}
    )


# --- adjudication.py ---------------------------------------------------------


def test_final_label_keeps_ensemble_label_for_agreed_decision() -> None:
    decision = g(_votes("r1", (1, 1, 1)), tau=0.0)

    assert final_label("r1", decision) == 1


def test_final_label_requires_human_label_for_escalated_decision() -> None:
    decision = g(_votes("r2", (-1, 1)), tau=0.0)
    assert decision.escalate is True

    with pytest.raises(AdjudicationError):
        final_label("r2", decision)


def test_final_label_human_label_is_authoritative_for_escalated_decision() -> None:
    decision = g(_votes("r3", (-1, 1)), tau=0.0)

    assert final_label("r3", decision, human_label=-1) == -1
    assert final_label("r3", decision, human_label=1) == 1
    assert final_label("r3", decision, human_label=0) == 0


def test_final_label_rejects_human_label_on_agreed_decision() -> None:
    decision = g(_votes("r4", (1, 1, 1)), tau=0.0)

    with pytest.raises(AdjudicationError):
        final_label("r4", decision, human_label=1)


def test_final_label_rejects_invalid_human_label() -> None:
    decision = g(_votes("r5", (-1, 1)), tau=0.0)

    with pytest.raises(AdjudicationError):
        final_label("r5", decision, human_label=2)


def test_queue_enqueue_rejects_agreed_decision() -> None:
    decision = g(_votes("r6", (1, 1, 1)), tau=0.0)
    queue = AdjudicationQueue()

    with pytest.raises(AdjudicationError):
        queue.enqueue("r6", _CONFIG_ID, decision)


def test_queue_routes_escalated_decision_and_resolve_is_authoritative() -> None:
    decision = g(_votes("r7", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()

    item = queue.enqueue("r7", _CONFIG_ID, decision)
    assert item.plane == PLANE_ADJUDICATION
    assert item.resolved is False
    assert queue.pending() == [item]

    resolved = queue.resolve("r7", 1)
    assert resolved.human_label == 1
    assert resolved.resolved is True
    assert queue.pending() == []


def test_queue_resolve_rejects_unknown_record() -> None:
    queue = AdjudicationQueue()

    with pytest.raises(AdjudicationError):
        queue.resolve("missing", 1)


def test_queue_resolve_rejects_invalid_human_label() -> None:
    decision = g(_votes("r8", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()
    queue.enqueue("r8", _CONFIG_ID, decision)

    with pytest.raises(AdjudicationError):
        queue.resolve("r8", 7)


# --- active_learning.py -------------------------------------------------------


def test_select_for_review_picks_high_dispersion_and_boundary_votes() -> None:
    votes = [
        _votes("low", (1, 1, 1)),  # zero dispersion, not boundary -> excluded
        _votes("boundary", (-1, 1)),  # boundary -> included regardless of dispersion
        _votes("high", (-1, 0, 1)),  # high dispersion and boundary -> included
    ]

    selections = select_for_review(votes, dispersion_threshold=0.5)

    record_ids = {s.record_id for s in selections}
    assert record_ids == {"boundary", "high"}


def test_select_for_review_excludes_boundary_when_disabled() -> None:
    votes = [_votes("boundary_only", (-1, 1))]

    selections = select_for_review(votes, dispersion_threshold=100.0, include_boundary=False)

    assert selections == []


def test_select_for_review_orders_by_dispersion_descending() -> None:
    votes = [
        _votes("mid", (0, 1)),
        _votes("high", (-1, 0, 1)),
        _votes("low", (0, 0, 1)),
    ]

    selections = select_for_review(votes, dispersion_threshold=0.0)

    dispersions = [s.dispersion for s in selections]
    assert dispersions == sorted(dispersions, reverse=True)


def test_select_for_review_respects_limit() -> None:
    votes = [_votes(f"r{i}", (-1, 1)) for i in range(5)]

    selections = select_for_review(votes, limit=2)

    assert len(selections) == 2


def test_select_for_review_rejects_negative_limit() -> None:
    with pytest.raises(ActiveLearningError):
        select_for_review([], limit=-1)


def test_selections_are_tagged_active_learning_plane() -> None:
    votes = [_votes("r1", (-1, 1)), _votes("r2", (-1, 0, 1))]

    selections = select_for_review(votes)

    assert selections
    assert all(s.plane == PLANE_ACTIVE_LEARNING for s in selections)


# --- recall_audit.py: drawing -------------------------------------------------


def _population(track_a: int, track_b: int) -> list[ExcludedRecord]:
    records = [ExcludedRecord(record_id=f"a{i}", track="a") for i in range(track_a)]
    records += [ExcludedRecord(record_id=f"b{i}", track="b") for i in range(track_b)]
    return records


def test_draw_audit_sample_draws_exactly_n() -> None:
    population = _population(track_a=8, track_b=2)

    rows = draw_audit_sample(population, 5, rng=random.Random(0))

    assert len(rows) == 5
    assert len({r.record_id for r in rows}) == 5


def test_draw_audit_sample_rejects_non_positive_n() -> None:
    population = _population(track_a=5, track_b=0)

    with pytest.raises(AuditError):
        draw_audit_sample(population, 0, rng=random.Random(0))


def test_draw_audit_sample_rejects_n_exceeding_population() -> None:
    population = _population(track_a=3, track_b=0)

    with pytest.raises(AuditError):
        draw_audit_sample(population, 4, rng=random.Random(0))


def test_draw_audit_sample_is_deterministic_with_seeded_rng() -> None:
    population = _population(track_a=10, track_b=10)

    first = draw_audit_sample(population, 6, rng=random.Random(42))
    second = draw_audit_sample(population, 6, rng=random.Random(42))

    assert [r.record_id for r in first] == [r.record_id for r in second]


def test_stratified_draw_respects_strata_sizes() -> None:
    population = _population(track_a=8, track_b=2)

    rows = draw_audit_sample(population, 5, stratify_by_track=True, rng=random.Random(0))

    by_stratum: dict[str, int] = {}
    for row in rows:
        by_stratum[row.stratum] = by_stratum.get(row.stratum, 0) + 1

    assert by_stratum == {"a": 4, "b": 1}
    assert all(rid.startswith("b") for rid in (r.record_id for r in rows if r.stratum == "b"))


def test_stratified_draw_never_exceeds_a_strata_population() -> None:
    population = _population(track_a=3, track_b=97)

    rows = draw_audit_sample(population, 100, stratify_by_track=True, rng=random.Random(0))

    by_stratum: dict[str, int] = {}
    for row in rows:
        by_stratum[row.stratum] = by_stratum.get(row.stratum, 0) + 1

    assert by_stratum["a"] <= 3
    assert by_stratum["b"] <= 97
    assert sum(by_stratum.values()) == 100


# --- recall_audit.py: ingestion and the firewall ------------------------------


def test_ingest_audit_labels_attaches_labels_and_keeps_plane_tag() -> None:
    rows = [AuditRow(record_id="r1", stratum="all"), AuditRow(record_id="r2", stratum="all")]

    labeled = ingest_audit_labels(rows, {"r1": 1, "r2": -1})

    assert [r.human_label for r in labeled] == [1, -1]
    assert all(r.plane == PLANE_RECALL_AUDIT for r in labeled)


def test_ingest_audit_labels_rejects_missing_label() -> None:
    rows = [AuditRow(record_id="r1", stratum="all")]

    with pytest.raises(AuditError):
        ingest_audit_labels(rows, {})


def test_ingest_audit_labels_rejects_invalid_label() -> None:
    rows = [AuditRow(record_id="r1", stratum="all")]

    with pytest.raises(AuditError):
        ingest_audit_labels(rows, {"r1": 5})


def test_build_strata_produces_counts_usable_by_stats_recall() -> None:
    rows = [AuditRow(record_id=f"r{i}", stratum="all") for i in range(20)]
    labels = {f"r{i}": (1 if i < 3 else -1) for i in range(20)}
    labeled = ingest_audit_labels(rows, labels)

    strata = build_strata(labeled, population_sizes={"all": 200})

    assert len(strata) == 1
    stratum = strata[0]
    assert stratum.name == "all"
    assert stratum.n == 20
    assert stratum.m == 3
    assert stratum.population == 200

    # The whole point: strata built here plug straight into stats.recall.
    estimate = stratified_recall(strata, true_positives=100)
    assert 0.0 <= estimate.point <= 1.0


def test_build_strata_rejects_unlabeled_rows() -> None:
    rows = [AuditRow(record_id="r1", stratum="all")]

    with pytest.raises(AuditError):
        build_strata(rows, population_sizes={"all": 10})


def test_build_strata_rejects_unknown_stratum_population() -> None:
    rows = ingest_audit_labels([AuditRow(record_id="r1", stratum="all")], {"r1": -1})

    with pytest.raises(AuditError):
        build_strata(rows, population_sizes={})


def test_build_strata_refuses_active_learning_selections() -> None:
    votes = [_votes("r1", (-1, 1))]
    selections = select_for_review(votes)
    assert selections

    with pytest.raises(AuditError):
        build_strata(selections, population_sizes={PLANE_ACTIVE_LEARNING: 10})


def test_build_strata_refuses_adjudication_items() -> None:
    decision = g(_votes("r1", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()
    item = queue.enqueue("r1", _CONFIG_ID, decision)
    resolved = queue.resolve("r1", 1)
    assert resolved.plane == PLANE_ADJUDICATION
    assert item.plane == PLANE_ADJUDICATION

    with pytest.raises(AuditError):
        build_strata([resolved], population_sizes={PLANE_ADJUDICATION: 10})
