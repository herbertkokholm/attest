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


class ActiveLearningError(ValueError):
    """Raised when an active-learning selection request violates its invariants."""


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
        plane: Fixed to `PLANE_ACTIVE_LEARNING`.
    """

    record_id: str
    ensemble_config_id: str
    dispersion: float
    boundary: bool
    confidence: RecordConfidence
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
        if s > dispersion_threshold or (include_boundary and b) or low_confidence:
            selections.append(
                ActiveLearningSelection(
                    record_id=vote_vector.record_id,
                    ensemble_config_id=vote_vector.ensemble_config_id,
                    dispersion=s,
                    boundary=b,
                    confidence=confidence,
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
