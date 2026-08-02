"""Active-learning plane: sample selection, statistically separate from the audit plane.

Selections here are chosen for their disagreement signal -- high dispersion
or a boundary split among ensemble votes -- to prioritize human review where
it teaches the most. Every selection is tagged `PLANE_ACTIVE_LEARNING` so it
can never be mistaken for, or fed into, the random recall audit: an
active-learning sample is not a probability sample of the excluded
population, and using it to estimate recall would silently bias the
estimate. `attest.planes.recall_audit.build_strata` refuses any row tagged
this way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from attest.ensemble.aggregate import dispersion, is_boundary
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
        plane: Fixed to `PLANE_ACTIVE_LEARNING`.
    """

    record_id: str
    ensemble_config_id: str
    dispersion: float
    boundary: bool
    plane: str = field(default=PLANE_ACTIVE_LEARNING, init=False)


def select_for_review(
    votes: Sequence[VoteVector],
    *,
    dispersion_threshold: float = 0.0,
    include_boundary: bool = True,
    limit: int | None = None,
) -> list[ActiveLearningSelection]:
    """Select vote vectors for active-learning review, ranked by uncertainty.

    A vote vector is selected if its dispersion strictly exceeds
    `dispersion_threshold`, or (when `include_boundary`) it straddles the
    exclude/include boundary. Selections are returned most-uncertain first
    (dispersion descending), then optionally truncated to `limit` -- the
    review budget for this round.

    Args:
        votes: Candidate vote vectors to consider for selection.
        dispersion_threshold: Minimum dispersion (exclusive) to qualify.
        include_boundary: Whether a boundary split alone (regardless of
            dispersion) also qualifies a vote vector for selection.
        limit: Maximum number of selections to return, or None for no cap.

    Returns:
        Selected vote vectors as `ActiveLearningSelection`s, every one
        tagged `PLANE_ACTIVE_LEARNING`, most-uncertain first.

    Raises:
        ActiveLearningError: If `limit` is negative.
    """
    if limit is not None and limit < 0:
        raise ActiveLearningError(f"limit must be non-negative, got {limit}")

    selections = []
    for vote_vector in votes:
        s = dispersion(vote_vector.ratings)
        b = is_boundary(vote_vector.ratings)
        if s > dispersion_threshold or (include_boundary and b):
            selections.append(
                ActiveLearningSelection(
                    record_id=vote_vector.record_id,
                    ensemble_config_id=vote_vector.ensemble_config_id,
                    dispersion=s,
                    boundary=b,
                )
            )

    selections.sort(key=lambda selection: selection.dispersion, reverse=True)
    if limit is not None:
        selections = selections[:limit]
    return selections
