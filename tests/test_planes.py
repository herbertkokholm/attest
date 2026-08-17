"""Tests for attest.planes: adjudication, active learning, and the recall-audit firewall."""

from __future__ import annotations

import math
import random

import pytest

from attest.ensemble.aggregate import g
from attest.ensemble.confidence import UNSCORED_TIER
from attest.ensemble.votes import build_vote_vector
from attest.planes import PLANE_ACTIVE_LEARNING, PLANE_ADJUDICATION, PLANE_RECALL_AUDIT
from attest.planes.active_learning import (
    SELECTION_REASON_BOUNDARY,
    SELECTION_REASON_DISPERSION,
    SELECTION_REASON_LOW_CONFIDENCE,
    ActiveLearningError,
    review_selection,
    select_for_review,
)
from attest.planes.adjudication import (
    SELECTION_REASON_BOUNDARY as ADJ_SELECTION_REASON_BOUNDARY,
)
from attest.planes.adjudication import (
    SELECTION_REASON_DISPERSION as ADJ_SELECTION_REASON_DISPERSION,
)
from attest.planes.adjudication import (
    SELECTION_REASON_TIE as ADJ_SELECTION_REASON_TIE,
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


def test_enqueue_records_boundary_selection_reason() -> None:
    decision = g(_votes("r9", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()

    item = queue.enqueue("r9", _CONFIG_ID, decision)

    assert item.selection_reason == ADJ_SELECTION_REASON_BOUNDARY


def test_enqueue_records_dispersion_selection_reason() -> None:
    decision = g(_votes("r10", (-1, 0, 1)), tau=0.0)
    queue = AdjudicationQueue()

    # (-1, 0, 1) is both boundary and high-dispersion; boundary wins as the
    # more specific reason (see escalation_reason's docstring).
    item = queue.enqueue("r10", _CONFIG_ID, decision)
    assert item.selection_reason == ADJ_SELECTION_REASON_BOUNDARY

    # A pure dispersion escalation with no boundary: three vendors split
    # (-1, 0, -1) around a non-zero, non-boundary mean under a tight tau.
    dispersion_only = g(_votes("r10b", (-1, 0, -1)), tau=0.1)
    assert dispersion_only.boundary is False
    assert dispersion_only.escalate is True
    item2 = queue.enqueue("r10b", _CONFIG_ID, dispersion_only)
    assert item2.selection_reason == ADJ_SELECTION_REASON_DISPERSION


def test_enqueue_records_tie_selection_reason_for_mean_zero_escalation() -> None:
    decision = g(_votes("r11", (0, 0)), tau=0.0)
    assert decision.boundary is False
    assert decision.dispersion == 0.0
    queue = AdjudicationQueue()

    item = queue.enqueue("r11", _CONFIG_ID, decision)

    assert item.selection_reason == ADJ_SELECTION_REASON_TIE


def test_resolve_records_reviewer_and_protocol_provenance() -> None:
    decision = g(_votes("r12", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()
    queue.enqueue("r12", _CONFIG_ID, decision)

    resolved = queue.resolve("r12", 1, reviewer="reviewer-a", protocol_id="proto-1")

    assert resolved.reviewer == "reviewer-a"
    assert resolved.protocol_id == "proto-1"
    assert resolved.resolved_at is not None


def test_resolve_without_reviewer_leaves_provenance_fields_none() -> None:
    decision = g(_votes("r13", (-1, 1)), tau=0.0)
    queue = AdjudicationQueue()
    queue.enqueue("r13", _CONFIG_ID, decision)

    resolved = queue.resolve("r13", 1)

    assert resolved.reviewer is None
    assert resolved.protocol_id is None
    assert resolved.resolved_at is not None  # always stamped, even without a reviewer id


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


def _confidence_votes(record_id: str, ratings: tuple[int, int, int]):
    return build_vote_vector(
        record_id,
        _CONFIG_ID,
        {"openai": ratings[0], "mistral": ratings[1], "together": ratings[2]},
    )


def _openai_shaped(logprob: float) -> dict:
    return {"logprobs": {"content": [{"token": "x", "logprob": logprob}]}}


def test_select_for_review_picks_unanimous_low_confidence_record() -> None:
    # (1, 1, 1): zero dispersion, no boundary -- invisible to the
    # disagreement signal alone, but every vendor's own confidence in that
    # "1" is weak (median ~0.4), so the confidence signal should still
    # surface it.
    votes = [_confidence_votes("shaky", (1, 1, 1))]
    raw_responses = {
        "shaky": {
            "openai": _openai_shaped(math.log(0.4)),
            "mistral": _openai_shaped(math.log(0.4)),
            "together": _openai_shaped(math.log(0.4)),
        }
    }

    selections = select_for_review(votes, dispersion_threshold=0.5, raw_responses=raw_responses)

    assert {s.record_id for s in selections} == {"shaky"}
    assert selections[0].confidence.scored is True
    assert selections[0].confidence.median_probability == pytest.approx(0.4)


def test_select_for_review_does_not_select_unanimous_high_confidence_record() -> None:
    votes = [_confidence_votes("solid", (1, 1, 1))]
    raw_responses = {
        "solid": {
            "openai": _openai_shaped(math.log(0.8)),
            "mistral": _openai_shaped(math.log(0.95)),
            "together": _openai_shaped(math.log(0.85)),
        }
    }

    selections = select_for_review(votes, dispersion_threshold=0.5, raw_responses=raw_responses)

    assert selections == []


def test_select_for_review_ignores_confidence_below_minimum_coverage() -> None:
    # Only openai and together carry a raw response -- mistral's entry is
    # absent, so n_supporting=2 < MIN_SUPPORTING_VOTES=3 and the record must
    # stay unscored, even though both available votes look low-confidence.
    votes = [_confidence_votes("undercovered", (1, 1, 1))]
    raw_responses = {
        "undercovered": {
            "openai": _openai_shaped(math.log(0.1)),
            "together": _openai_shaped(math.log(0.1)),
        }
    }

    selections = select_for_review(votes, dispersion_threshold=0.5, raw_responses=raw_responses)

    assert selections == []


def test_select_for_review_without_raw_responses_never_triggers_confidence() -> None:
    # Reproduces this function's pre-confidence behavior exactly.
    votes = [_confidence_votes("no_raw", (1, 1, 1))]

    selections = select_for_review(votes, dispersion_threshold=0.5)

    assert selections == []


def test_select_for_review_attaches_confidence_to_dispersion_driven_selections() -> None:
    votes = [_votes("boundary", (-1, 1))]

    [selection] = select_for_review(votes, dispersion_threshold=100.0)

    assert selection.confidence.record_id == "boundary"
    assert selection.confidence.scored is False


def test_select_for_review_sorts_by_dispersion_then_confidence_ascending() -> None:
    votes = [
        _votes("high_dispersion", (-1, 0, 1)),
        _confidence_votes("less_shaky", (1, 1, 1)),
        _confidence_votes("shaky", (1, 1, 1)),
    ]
    raw_responses = {
        "shaky": {
            "openai": _openai_shaped(math.log(0.1)),
            "mistral": _openai_shaped(math.log(0.1)),
            "together": _openai_shaped(math.log(0.1)),
        },
        "less_shaky": {
            "openai": _openai_shaped(math.log(0.45)),
            "mistral": _openai_shaped(math.log(0.45)),
            "together": _openai_shaped(math.log(0.45)),
        },
    }

    selections = select_for_review(votes, dispersion_threshold=0.5, raw_responses=raw_responses)

    assert [s.record_id for s in selections] == ["high_dispersion", "shaky", "less_shaky"]


def test_select_for_review_rejects_confidence_threshold_outside_unit_interval() -> None:
    with pytest.raises(ActiveLearningError):
        select_for_review([], confidence_threshold=1.5)


def test_selection_reason_reports_boundary_and_dispersion() -> None:
    votes = [_votes("boundary", (-1, 1)), _votes("dispersion", (-1, 0, 1, 1))]

    selections = select_for_review(votes, dispersion_threshold=0.5)

    by_id = {s.record_id: s for s in selections}
    assert SELECTION_REASON_BOUNDARY in by_id["boundary"].selection_reason
    assert SELECTION_REASON_DISPERSION in by_id["dispersion"].selection_reason
    assert SELECTION_REASON_BOUNDARY in by_id["dispersion"].selection_reason


def test_selection_reason_reports_low_confidence_for_unanimous_shaky_record() -> None:
    votes = [_confidence_votes("shaky", (1, 1, 1))]
    raw_responses = {
        "shaky": {
            "openai": _openai_shaped(math.log(0.4)),
            "mistral": _openai_shaped(math.log(0.4)),
            "together": _openai_shaped(math.log(0.4)),
        }
    }

    [selection] = select_for_review(votes, dispersion_threshold=0.5, raw_responses=raw_responses)

    assert selection.selection_reason == (SELECTION_REASON_LOW_CONFIDENCE,)


def test_review_selection_records_reviewer_provenance() -> None:
    votes = [_votes("r1", (-1, 1))]
    [selection] = select_for_review(votes)

    review = review_selection(
        selection, reviewer="reviewer-a", protocol_id="proto-1", notes="prompt needs clarifying"
    )

    assert review.record_id == "r1"
    assert review.reviewer == "reviewer-a"
    assert review.protocol_id == "proto-1"
    assert review.notes == "prompt needs clarifying"
    assert review.selection_reason == selection.selection_reason
    assert review.plane == PLANE_ACTIVE_LEARNING


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


def _population_by_confidence(low: int, high: int, unscored: int) -> list[ExcludedRecord]:
    records = [
        ExcludedRecord(record_id=f"low{i}", track="t", confidence_tier="low") for i in range(low)
    ]
    records += [
        ExcludedRecord(record_id=f"high{i}", track="t", confidence_tier="high") for i in range(high)
    ]
    records += [
        ExcludedRecord(record_id=f"unscored{i}", track="t", confidence_tier=UNSCORED_TIER)
        for i in range(unscored)
    ]
    return records


def test_stratified_draw_by_confidence_respects_strata_sizes() -> None:
    population = _population_by_confidence(low=8, high=2, unscored=0)

    rows = draw_audit_sample(population, 5, stratify_by_confidence=True, rng=random.Random(0))

    by_stratum: dict[str, int] = {}
    for row in rows:
        by_stratum[row.stratum] = by_stratum.get(row.stratum, 0) + 1

    assert by_stratum == {"low": 4, "high": 1}


def test_stratified_draw_by_confidence_gives_unscored_its_own_stratum() -> None:
    population = _population_by_confidence(low=5, high=5, unscored=10)

    rows = draw_audit_sample(population, 20, stratify_by_confidence=True, rng=random.Random(0))

    strata = {row.stratum for row in rows}
    assert strata == {"low", "high", UNSCORED_TIER}


def test_draw_audit_sample_rejects_combined_track_and_confidence_stratification() -> None:
    population = _population_by_confidence(low=5, high=5, unscored=0)

    with pytest.raises(AuditError):
        draw_audit_sample(
            population,
            5,
            stratify_by_track=True,
            stratify_by_confidence=True,
            rng=random.Random(0),
        )


def test_draw_audit_sample_by_confidence_rejects_record_with_no_tier() -> None:
    population = [ExcludedRecord(record_id="r1", track="t")]  # confidence_tier defaults to None

    with pytest.raises(AuditError):
        draw_audit_sample(population, 1, stratify_by_confidence=True, rng=random.Random(0))


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
