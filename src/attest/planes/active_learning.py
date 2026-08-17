"""Active-learning plane: sample selection, statistically separate from the audit plane.

Selections are chosen for either of two independent uncertainty signals:
*between*-vendor disagreement (high dispersion or a boundary split among
ensemble votes) or *within*-vendor low confidence (see
`attest.ensemble.confidence`). These catch different failure modes: a
unanimous vote vector like `(1, 1, 1)` has zero dispersion and no boundary,
so it is invisible to the disagreement signal alone -- but if every vendor's
own logprob-derived confidence in that `1` was weak (e.g. each medianing
around 0.4), the ensemble's apparent agreement is shakier than it looks, and
that is exactly the class of record the confidence signal exists to surface.
Reuses `attest.ensemble.confidence.record_confidence`/`confidence_tier`
unchanged -- the same coverage-gated, median-based, exclude-don't-impute
figure `attest.planes.recall_audit` stratifies audit draws with (see
`docs/logprob_support.md`), not a separate statistic invented for this
plane. Every selection is tagged `PLANE_ACTIVE_LEARNING` so it can never be
mistaken for, or fed into, the random recall audit: an active-learning
sample is not a probability sample of the excluded population, and using it
to estimate recall would silently bias the estimate.
`attest.planes.recall_audit.build_strata` refuses any row tagged this way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from attest.ensemble.aggregate import dispersion, is_boundary
from attest.ensemble.confidence import (
    DEFAULT_LOW_THRESHOLD,
    TIER_LOW,
    RecordConfidence,
    confidence_tier,
    record_confidence,
)
from attest.ensemble.votes import VoteVector
from attest.planes import PLANE_ACTIVE_LEARNING

SELECTION_REASON_DISPERSION = "dispersion"
SELECTION_REASON_BOUNDARY = "boundary"
SELECTION_REASON_LOW_CONFIDENCE = "low_confidence"


class ActiveLearningError(ValueError):
    """Raised when an active-learning selection request violates its invariants."""


def _selection_reasons(
    *, dispersion_hit: bool, boundary_hit: bool, confidence_hit: bool
) -> tuple[str, ...]:
    """Every independent signal that qualified a record, most-specific first.

    A record can be selected by more than one signal at once (e.g. a
    boundary split that is also low-confidence); all that fired are
    recorded, not just the first, since a reviewer benefits from seeing the
    complete picture of why a record was routed here.
    """
    reasons: list[str] = []
    if boundary_hit:
        reasons.append(SELECTION_REASON_BOUNDARY)
    if dispersion_hit:
        reasons.append(SELECTION_REASON_DISPERSION)
    if confidence_hit:
        reasons.append(SELECTION_REASON_LOW_CONFIDENCE)
    return tuple(reasons)


@dataclass(frozen=True)
class ActiveLearningSelection:
    """One vote vector selected for active-learning review.

    Attributes:
        record_id: Id of the selected record.
        ensemble_config_id: Ensemble configuration in force when votes were cast.
        dispersion: The vote vector's dispersion.
        boundary: Whether the vote vector straddled the exclude/include boundary.
        confidence: This record's coverage-gated ensemble confidence (see
            `attest.ensemble.confidence.record_confidence`), attached
            regardless of which signal actually triggered selection -- a
            record selected for dispersion/boundary may also be
            low-confidence, and a reviewer should be able to see that
            without recomputing it.
        selection_reason: Every independent signal that qualified this
            record for selection (see `_selection_reasons`), e.g.
            `("boundary", "low_confidence")`. Provenance only -- ranking
            (`select_for_review`'s sort) still uses `dispersion`/`confidence`
            directly, not this field.
        plane: Fixed to `PLANE_ACTIVE_LEARNING`.
    """

    record_id: str
    ensemble_config_id: str
    dispersion: float
    boundary: bool
    confidence: RecordConfidence
    selection_reason: tuple[str, ...] = ()
    plane: str = field(default=PLANE_ACTIVE_LEARNING, init=False)


def select_for_review(
    votes: Sequence[VoteVector],
    *,
    dispersion_threshold: float = 0.0,
    include_boundary: bool = True,
    confidence_threshold: float = DEFAULT_LOW_THRESHOLD,
    raw_responses: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = None,
) -> list[ActiveLearningSelection]:
    """Select vote vectors for active-learning review, ranked by uncertainty.

    A vote vector is selected if its dispersion strictly exceeds
    `dispersion_threshold`; or (when `include_boundary`) it straddles the
    exclude/include boundary; or its coverage-gated confidence (see
    `attest.ensemble.confidence.record_confidence`) is scored and at or
    below `confidence_threshold`. A record with fewer than
    `attest.ensemble.confidence.MIN_SUPPORTING_VOTES` logprob-supporting
    votes can never qualify via the confidence signal alone -- there is no
    evidence either way for it, so only dispersion/boundary can still select
    it.

    Selections are returned most-uncertain first: primarily by dispersion
    descending, then -- among equal-dispersion selections, most commonly the
    many zero-dispersion unanimous records the confidence signal newly makes
    visible -- by ascending confidence (least confident first). Disagreement
    between vendors is treated as the stronger signal; low confidence within
    an otherwise-unanimous vote is a secondary tiebreak, not merged into a
    single combined score, since no particular weighting of the two is
    statistically justified here.

    Args:
        votes: Candidate vote vectors to consider for selection.
        dispersion_threshold: Minimum dispersion (exclusive) to qualify.
        include_boundary: Whether a boundary split alone (regardless of
            dispersion) also qualifies a vote vector for selection.
        confidence_threshold: Median confidence at or below which a scored
            record qualifies via the confidence signal alone. See
            `attest.ensemble.confidence.confidence_tier` for why 0.5 is the
            default and why this is a run-provenance policy choice, not a
            hash-versioned one.
        raw_responses: Per-record, per-vendor raw rater responses (i.e.
            `attest.vendors.base.EnsembleRun.raw_responses`), needed to
            derive `confidence`. `None` (the default) is equivalent to
            supplying an empty mapping: every record is then coverage-gated
            `scored=False` and the confidence signal never contributes,
            reproducing this function's pre-confidence behavior exactly.
        limit: Maximum number of selections to return, or None for no cap.

    Returns:
        Selected vote vectors as `ActiveLearningSelection`s, every one
        tagged `PLANE_ACTIVE_LEARNING`, most-uncertain first.

    Raises:
        ActiveLearningError: If `limit` is negative, or `confidence_threshold`
            is outside `[0, 1]`.
    """
    if limit is not None and limit < 0:
        raise ActiveLearningError(f"limit must be non-negative, got {limit}")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ActiveLearningError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
        )

    selections = []
    for vote_vector in votes:
        s = dispersion(vote_vector.ratings)
        b = is_boundary(vote_vector.ratings)
        record_raw = (raw_responses or {}).get(vote_vector.record_id, {})
        confidence = record_confidence(vote_vector, record_raw)
        low_confidence = confidence_tier(confidence, low_threshold=confidence_threshold) == TIER_LOW
        dispersion_hit = s > dispersion_threshold
        boundary_hit = include_boundary and b
        if dispersion_hit or boundary_hit or low_confidence:
            selections.append(
                ActiveLearningSelection(
                    record_id=vote_vector.record_id,
                    ensemble_config_id=vote_vector.ensemble_config_id,
                    dispersion=s,
                    boundary=b,
                    confidence=confidence,
                    selection_reason=_selection_reasons(
                        dispersion_hit=dispersion_hit,
                        boundary_hit=boundary_hit,
                        confidence_hit=low_confidence,
                    ),
                )
            )

    selections.sort(
        key=lambda selection: (
            selection.dispersion,
            1 - selection.confidence.median_probability if selection.confidence.scored else 0.0,
        ),
        reverse=True,
    )
    if limit is not None:
        selections = selections[:limit]
    return selections


@dataclass(frozen=True)
class ActiveLearningReview:
    """Provenance record of a human reviewing an active-learning selection.

    Recording a review is deliberately the only thing this plane does with
    it: attest never auto-tunes a prompt or threshold from active-learning
    output (see `attest.planes.active_learning`'s module docstring and
    README's boundary rule) -- this dataclass exists so the auditable trail
    exists, not to drive any automation. A reviewer's `notes` may record
    that this review motivated a later explicit config change; the actual
    change is then a separate, human-initiated
    `attest.provenance.changelog.ConfigChangeEvent`, not something this
    review triggers by itself.

    Attributes:
        record_id: Id of the reviewed record.
        ensemble_config_id: Ensemble configuration in force when reviewed.
        selection_reason: The selection's own `ActiveLearningSelection.selection_reason`.
        reviewer: Id or pseudonym of the reviewing human.
        reviewed_at: When the review occurred.
        protocol_id: Id of the adjudication/review protocol in force, if any.
        notes: Free-text reviewer notes, e.g. "prompt needs a clarified
            eligibility criterion" -- the human judgment call that a
            follow-up config change, if any, is based on.
        plane: Fixed to `PLANE_ACTIVE_LEARNING`.
    """

    record_id: str
    ensemble_config_id: str
    selection_reason: tuple[str, ...]
    reviewer: str
    reviewed_at: datetime
    protocol_id: str | None = None
    notes: str = ""
    plane: str = field(default=PLANE_ACTIVE_LEARNING, init=False)


def review_selection(
    selection: ActiveLearningSelection,
    *,
    reviewer: str,
    protocol_id: str | None = None,
    notes: str = "",
    reviewed_at: datetime | None = None,
) -> ActiveLearningReview:
    """Record a human review of an active-learning selection.

    Args:
        selection: The selection being reviewed.
        reviewer: Id or pseudonym of the reviewing human.
        protocol_id: Id of the adjudication/review protocol in force, if any.
        notes: Free-text reviewer notes.
        reviewed_at: When the review occurred; defaults to now (UTC).

    Returns:
        The `ActiveLearningReview` provenance record.
    """
    return ActiveLearningReview(
        record_id=selection.record_id,
        ensemble_config_id=selection.ensemble_config_id,
        selection_reason=selection.selection_reason,
        reviewer=reviewer,
        reviewed_at=reviewed_at or datetime.now(UTC),
        protocol_id=protocol_id,
        notes=notes,
    )
