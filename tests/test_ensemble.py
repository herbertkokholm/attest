"""Tests for attest.ensemble: vote vectors and g(R, aggregation, tau) aggregation."""

from __future__ import annotations

from attest.ensemble.aggregate import dispersion, g
from attest.ensemble.votes import VoteVector, build_vote_vector

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
