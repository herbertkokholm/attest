"""Aggregation of a retained vote vector R_j into an auto-label or an escalation.

Implements the decision function g(R_j, aggregation, tau, zero_policy) ->
{auto-label, escalate}. The default rule escalates whenever the vote vector
straddles the boundary between exclude and include, or its dispersion
exceeds tau; otherwise it auto-labels with the sign of the mean vote. The
rule is selectable so that alternative aggregation strategies (e.g.
majority, unanimity) can be added later without changing the decision
function's shape.

`auto_label == 0` ("uncertain") is never a terminal decision: ordinal 0 is
not a gold category (gold is binary +1/-1), so a terminal 0 has no defined
disposition in the confusion matrix or the excluded/included populations,
and would silently fabricate false negatives or false positives depending on
where it happened to land downstream. `zero_policy` makes what happens to a
would-be 0 auto-label an explicit, hashed-into-the-config choice instead:
`ZERO_POLICY_ESCALATE` (the default, recall-safe) routes it to a human via
escalation; `ZERO_POLICY_INCLUDE` folds it into `+1`. There is deliberately
no "exclude" option -- auto-excluding on ensemble uncertainty is the one
disposition that silently destroys recall.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from attest.ensemble.votes import VoteVector

AGGREGATION_BOUNDARY_DISPERSION = "boundary_dispersion"
AGGREGATION_MAJORITY = "majority"
AGGREGATION_UNANIMITY = "unanimity"

_KNOWN_AGGREGATIONS = (
    AGGREGATION_BOUNDARY_DISPERSION,
    AGGREGATION_MAJORITY,
    AGGREGATION_UNANIMITY,
)

ZERO_POLICY_ESCALATE = "escalate"
ZERO_POLICY_INCLUDE = "include"

KNOWN_ZERO_POLICIES = (ZERO_POLICY_ESCALATE, ZERO_POLICY_INCLUDE)


@dataclass(frozen=True)
class Decision:
    """The outcome of applying an aggregation rule to a vote vector.

    Attributes:
        auto_label: The auto-assigned ordinal label, -1 or +1, or None when
            the vote vector escalates instead of being auto-labeled. Never
            0: `g` never auto-commits the uncertain category as a final
            label (see `zero_policy` on `g`) -- a would-be 0 is either
            escalated (`auto_label` is None) or folded into `+1`.
        escalate: Whether the vote vector escalates rather than auto-labels.
        dispersion: Sample standard deviation of the vote vector's ratings.
        boundary: Whether the vote vector straddles the exclude/include boundary.
    """

    auto_label: int | None
    escalate: bool
    dispersion: float
    boundary: bool


def dispersion(ratings: Sequence[int]) -> float:
    """Compute the sample standard deviation of a vote vector's ordinal ratings.

    s(R_j) = sqrt(sum((r - mean)^2) / (x - 1)), where x = len(ratings). A
    single-vote vector (x == 1) has zero dispersion by definition, since
    sample variance is undefined for x == 1.

    Args:
        ratings: The ordinal ratings of a vote vector.

    Returns:
        The sample standard deviation, or 0.0 when there is only one rating.
    """
    x = len(ratings)
    if x <= 1:
        return 0.0
    mean = sum(ratings) / x
    variance = sum((r - mean) ** 2 for r in ratings) / (x - 1)
    return math.sqrt(variance)


def is_boundary(ratings: Sequence[int]) -> bool:
    """Return whether a vote vector straddles the exclude/include boundary.

    B(R_j) = 1 iff the vote vector contains both a -1 (exclude) and a +1
    (include) vote.

    Args:
        ratings: The ordinal ratings of a vote vector.

    Returns:
        True iff `ratings` contains both -1 and +1.
    """
    return min(ratings) == -1 and max(ratings) == 1


def _sign_label(mean: float) -> int:
    if mean > 0:
        return 1
    if mean < 0:
        return -1
    return 0


def _boundary_dispersion(
    ratings: Sequence[int], tau: float, zero_policy: str = ZERO_POLICY_ESCALATE
) -> Decision:
    s = dispersion(ratings)
    boundary = is_boundary(ratings)
    sign = _sign_label(sum(ratings) / len(ratings))
    tie = sign == 0
    if zero_policy == ZERO_POLICY_ESCALATE:
        escalate = boundary or s > tau or tie
        auto_label = None if escalate else sign
    elif zero_policy == ZERO_POLICY_INCLUDE:
        escalate = boundary or s > tau
        auto_label = None if escalate else (1 if tie else sign)
    else:
        raise ValueError(
            f"unknown zero_policy '{zero_policy}': expected one of {KNOWN_ZERO_POLICIES}"
        )
    return Decision(auto_label=auto_label, escalate=escalate, dispersion=s, boundary=boundary)


def g(
    votes: VoteVector,
    aggregation: str = AGGREGATION_BOUNDARY_DISPERSION,
    tau: float = 0.0,
    zero_policy: str = ZERO_POLICY_ESCALATE,
) -> Decision:
    """Apply an aggregation rule to a vote vector: g(R_j, aggregation, tau, zero_policy).

    Never auto-commits the uncertain category (ordinal 0) as a final label:
    a vote vector whose mean is exactly 0 (and which is not otherwise
    escalated by the boundary or dispersion rules) is either escalated
    (under `ZERO_POLICY_ESCALATE`) or folded into `+1` (under
    `ZERO_POLICY_INCLUDE`) -- `Decision.auto_label` is never 0.

    Args:
        votes: The retained vote vector R_j for a record.
        aggregation: Name of the aggregation rule to apply. Defaults to
            `AGGREGATION_BOUNDARY_DISPERSION`, which escalates whenever the
            vote vector straddles the exclude/include boundary or its
            dispersion exceeds `tau`, and otherwise auto-labels with the
            sign of the mean vote (subject to `zero_policy`, below).
        tau: Dispersion threshold used by the boundary+dispersion rule.
        zero_policy: Disposition of a would-be `auto_label == 0` under the
            boundary+dispersion rule. `ZERO_POLICY_ESCALATE` (the default)
            escalates it, exactly like a boundary or dispersion escalation.
            `ZERO_POLICY_INCLUDE` auto-labels it `+1` instead. There is
            deliberately no "exclude" option.

    Returns:
        A `Decision` carrying either an auto-label or an escalation, along
        with the dispersion and boundary values that produced it.

    Raises:
        NotImplementedError: If `aggregation` names a recognized but not yet
            implemented rule (`AGGREGATION_MAJORITY`, `AGGREGATION_UNANIMITY`).
        ValueError: If `aggregation` names an unrecognized rule, or
            `zero_policy` names an unrecognized policy.
    """
    if aggregation == AGGREGATION_BOUNDARY_DISPERSION:
        return _boundary_dispersion(votes.ratings, tau, zero_policy)
    if aggregation in _KNOWN_AGGREGATIONS:
        raise NotImplementedError(f"aggregation rule '{aggregation}' is not yet implemented")
    raise ValueError(
        f"unknown aggregation rule '{aggregation}': expected one of {_KNOWN_AGGREGATIONS}"
    )
