"""Vote model: per-vendor ordinal votes retained as a vote vector for one record.

Raw votes are the source of truth. The aggregate label computed in
``aggregate.py`` is a derived view over a `VoteVector`; the vector itself is
always retained so the original per-vendor votes can be recovered later
(re-aggregated under a different rule, audited, or inspected for
disagreement) rather than discarded in favour of the aggregate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

VALID_RATINGS = (-1, 0, 1)


class VoteError(ValueError):
    """Raised when a vote or vote vector violates the ordinal-scale invariant."""


@dataclass(frozen=True)
class Vote:
    """One ensemble member's ordinal rating for a record.

    Attributes:
        vendor: Name of the voting ensemble member (e.g. "openai").
        rating: Ordinal rating: -1 (exclude), 0 (related/uncertain), or +1 (include).
    """

    vendor: str
    rating: int

    def __post_init__(self) -> None:
        if self.rating not in VALID_RATINGS:
            raise VoteError(
                f"vendor '{self.vendor}': rating must be one of {VALID_RATINGS}, "
                f"got {self.rating}"
            )


@dataclass(frozen=True)
class VoteVector:
    """R_j: the retained set of per-vendor votes for one record.

    Stamped with the `ensemble_config_id` in force when the votes were cast,
    so a vote vector is always traceable to the exact vendor/model/prompt
    configuration that produced it.

    Attributes:
        record_id: Id of the record these votes were cast on.
        ensemble_config_id: Content-derived id of the ensemble configuration
            in force when these votes were cast.
        votes: The individual per-vendor votes, in casting order.
    """

    record_id: str
    ensemble_config_id: str
    votes: tuple[Vote, ...]

    def __post_init__(self) -> None:
        if not self.votes:
            raise VoteError(
                f"record '{self.record_id}': vote vector must contain at least one vote"
            )

    @property
    def ratings(self) -> tuple[int, ...]:
        """The raw ordinal ratings, in casting order, recoverable from the vote vector."""
        return tuple(vote.rating for vote in self.votes)


def build_vote_vector(
    record_id: str, ensemble_config_id: str, ratings: Mapping[str, int] | Sequence[Vote]
) -> VoteVector:
    """Build a `VoteVector` from either a vendor-to-rating mapping or a sequence of `Vote`s.

    Args:
        record_id: Id of the record these votes were cast on.
        ensemble_config_id: Content-derived id of the ensemble configuration
            in force when these votes were cast.
        ratings: Either a mapping of vendor name to ordinal rating, or an
            already-built sequence of `Vote`s.

    Returns:
        A `VoteVector` stamped with `ensemble_config_id`.
    """
    votes: tuple[Vote, ...]
    if isinstance(ratings, Mapping):
        votes = tuple(Vote(vendor=vendor, rating=rating) for vendor, rating in ratings.items())
    else:
        votes = tuple(ratings)
    return VoteVector(record_id=record_id, ensemble_config_id=ensemble_config_id, votes=votes)
