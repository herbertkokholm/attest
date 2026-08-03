"""Tests for attest.ensemble: vote vectors and g(R, aggregation, tau) aggregation."""

from __future__ import annotations

import itertools

import pytest

from attest.ensemble.aggregate import (
    KNOWN_ZERO_POLICIES,
    ZERO_POLICY_ESCALATE,
    ZERO_POLICY_INCLUDE,
    dispersion,
    g,
)
from attest.ensemble.votes import VALID_RATINGS, VoteVector, build_vote_vector

_CONFIG_ID = "config-abc"


def _votes(record_id: str, ratings: tuple[int, ...]) -> VoteVector:
    return build_vote_vector(
        record_id,
        _CONFIG_ID,
        {f"vendor{i}": rating for i, rating in enumerate(ratings)},
    )


def test_boundary_votes_escalate_for_any_tau() -> None:
    votes = _votes("r1", (-1, 1, 1))

    for tau in (0.0, 0.5, 1.0, 100.0):
        decision = g(votes, tau=tau)
        assert decision.escalate is True
        assert decision.boundary is True
        assert decision.auto_label is None


def test_low_dispersion_votes_auto_label_include_when_tau_covers_dispersion() -> None:
    ratings = (0, 1, 1)
    votes = _votes("r2", ratings)
    s = dispersion(ratings)

    decision = g(votes, tau=s)

    assert decision.boundary is False
    assert decision.escalate is False
    assert decision.auto_label == 1
    assert decision.dispersion == s


def test_low_dispersion_votes_escalate_when_tau_below_dispersion() -> None:
    ratings = (0, 1, 1)
    votes = _votes("r2b", ratings)
    s = dispersion(ratings)

    decision = g(votes, tau=s - 0.01)

    assert decision.escalate is True
    assert decision.auto_label is None


def test_unanimous_votes_have_zero_dispersion_and_auto_label() -> None:
    votes = _votes("r3", (1, 1, 1))

    decision = g(votes, tau=0.0)

    assert decision.dispersion == 0.0
    assert decision.boundary is False
    assert decision.escalate is False
    assert decision.auto_label == 1


def test_unanimous_exclude_votes_auto_label_exclude() -> None:
    votes = _votes("r4", (-1, -1))

    decision = g(votes, tau=0.0)

    assert decision.auto_label == -1
    assert decision.escalate is False


def test_single_vote_has_zero_dispersion() -> None:
    assert dispersion((1,)) == 0.0
    assert dispersion(()) == 0.0


def test_raw_votes_remain_recoverable_from_vote_vector() -> None:
    votes = build_vote_vector("r5", _CONFIG_ID, {"openai": -1, "anthropic": 0, "gemini": 1})

    assert votes.ratings == (-1, 0, 1)
    assert [v.vendor for v in votes.votes] == ["openai", "anthropic", "gemini"]
    assert votes.ensemble_config_id == _CONFIG_ID
    assert votes.record_id == "r5"

    # Aggregation does not discard or mutate the underlying votes.
    g(votes, tau=0.0)
    assert votes.ratings == (-1, 0, 1)


# --- zero_policy: auto_label == 0 is never a terminal decision -----------------


@pytest.mark.parametrize("x", [2, 3, 4, 5])
@pytest.mark.parametrize("zero_policy", KNOWN_ZERO_POLICIES)
@pytest.mark.parametrize("tau", [0.0, 0.3, 0.5, 1.0, 100.0])
def test_auto_label_is_never_zero_over_every_vote_vector(
    x: int, zero_policy: str, tau: float
) -> None:
    for ratings in itertools.product(VALID_RATINGS, repeat=x):
        votes = _votes("r", ratings)
        decision = g(votes, tau=tau, zero_policy=zero_policy)
        assert decision.auto_label != 0


def test_all_uncertain_tie_escalates_under_zero_policy_escalate() -> None:
    votes = _votes("r-tie", (0, 0, 0, 0))

    decision = g(votes, tau=1.0, zero_policy=ZERO_POLICY_ESCALATE)

    assert decision.escalate is True
    assert decision.auto_label is None
    assert decision.boundary is False


def test_all_uncertain_tie_includes_under_zero_policy_include() -> None:
    votes = _votes("r-tie", (0, 0, 0, 0))

    decision = g(votes, tau=1.0, zero_policy=ZERO_POLICY_INCLUDE)

    assert decision.escalate is False
    assert decision.auto_label == 1


def test_mixed_tie_escalates_under_zero_policy_escalate() -> None:
    # (-1, -1, 1, 1): mean 0, but this IS a boundary vote (contains both -1
    # and +1), so it escalates via the boundary rule regardless of zero_policy.
    votes = _votes("r-mixed-tie", (-1, -1, 1, 1))

    decision = g(votes, tau=100.0, zero_policy=ZERO_POLICY_ESCALATE)

    assert decision.escalate is True
    assert decision.boundary is True
    assert decision.auto_label is None


def test_mixed_tie_escalates_under_zero_policy_include_too() -> None:
    # The boundary rule dominates zero_policy: a boundary tie escalates
    # under "include" as well, since the boundary check happens regardless.
    votes = _votes("r-mixed-tie", (-1, -1, 1, 1))

    decision = g(votes, tau=100.0, zero_policy=ZERO_POLICY_INCLUDE)

    assert decision.escalate is True
    assert decision.boundary is True
    assert decision.auto_label is None


def test_zero_policy_defaults_to_escalate() -> None:
    votes = _votes("r-tie", (0, 0))

    default_decision = g(votes, tau=1.0)
    explicit_decision = g(votes, tau=1.0, zero_policy=ZERO_POLICY_ESCALATE)

    assert default_decision == explicit_decision
    assert default_decision.escalate is True


def test_g_raises_for_unknown_zero_policy() -> None:
    votes = _votes("r", (1, 1))

    with pytest.raises(ValueError, match="unknown zero_policy"):
        g(votes, tau=0.0, zero_policy="exclude")


def test_g_outputs_unchanged_for_non_zero_mean_vote_vectors() -> None:
    # zero_policy only changes behavior for exact ties (mean == 0); every
    # other vote vector must decide identically under both policies.
    for ratings in itertools.product(VALID_RATINGS, repeat=4):
        if sum(ratings) == 0:
            continue  # exact ties are the one case this workstream changes.
        votes = _votes("r", ratings)
        escalate_decision = g(votes, tau=0.5, zero_policy=ZERO_POLICY_ESCALATE)
        include_decision = g(votes, tau=0.5, zero_policy=ZERO_POLICY_INCLUDE)
        assert escalate_decision == include_decision
