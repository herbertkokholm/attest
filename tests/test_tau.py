"""Tests for attest.ensemble.tau: tau as a derived, validated, self-documenting quantity."""

from __future__ import annotations

import math

import pytest

from attest.ensemble.tau import (
    TIE_POLICY_ESCALATE,
    TIE_POLICY_TOLERATE,
    TauReport,
    describe_tau,
    reachable_dispersions,
    resolve_tau,
    validate_tau,
)

# --- reachable_dispersions ------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (3, (0.0, 0.5773502691896257)),
        (4, (0.0, 0.5, 0.5773502691896257)),
        (5, (0.0, 0.4472135954999579, 0.5477225575051661)),
    ],
)
def test_reachable_dispersions_matches_known_sets(x: int, expected: tuple[float, ...]) -> None:
    reachable = reachable_dispersions(x)

    assert len(reachable) == len(expected)
    for actual, want in zip(reachable, expected, strict=True):
        assert actual == pytest.approx(want)


def test_reachable_dispersions_always_includes_zero() -> None:
    for x in range(1, 8):
        assert 0.0 in reachable_dispersions(x)


def test_reachable_dispersions_is_sorted_and_distinct() -> None:
    reachable = reachable_dispersions(6)

    assert list(reachable) == sorted(reachable)
    assert len(set(reachable)) == len(reachable)


def test_reachable_dispersions_excludes_boundary_by_default() -> None:
    without_boundary = reachable_dispersions(2)
    with_boundary = reachable_dispersions(2, include_boundary=True)

    # (-1, 1) is a boundary vector with sd = sqrt(2) ~= 1.414, only reachable
    # when boundary vectors are included.
    assert math.sqrt(2) not in without_boundary
    assert any(math.isclose(v, math.sqrt(2)) for v in with_boundary)


def test_reachable_dispersions_raises_for_x_below_one() -> None:
    with pytest.raises(ValueError, match="x must be >= 1"):
        reachable_dispersions(0)


def test_reachable_dispersions_raises_for_empty_domain() -> None:
    with pytest.raises(ValueError, match="domain must not be empty"):
        reachable_dispersions(3, domain=())


def test_reachable_dispersions_x_one_is_only_zero() -> None:
    assert reachable_dispersions(1) == (0.0,)


# --- describe_tau: structure -----------------------------------------------------


def test_describe_tau_returns_report_for_this_tau_and_x() -> None:
    report = describe_tau(0.3, 4)

    assert report.tau == 0.3
    assert report.x == 4
    assert report.reachable == reachable_dispersions(4)
    assert isinstance(report, TauReport)


def test_describe_tau_effective_breakpoint_is_largest_reachable_strictly_below_tau() -> None:
    reachable = reachable_dispersions(4)  # (0.0, 0.5, 0.577...)

    report = describe_tau(0.3, 4)
    assert report.effective_breakpoint == 0.0

    report_above_all = describe_tau(10.0, 4)
    assert report_above_all.effective_breakpoint == reachable[-1]

    report_below_all = describe_tau(-1.0, 4)
    assert report_below_all.effective_breakpoint is None


def test_describe_tau_canonical_interval_brackets_tau() -> None:
    report = describe_tau(0.3, 4)

    lo, hi = report.canonical_interval
    assert lo == 0.0
    assert hi == pytest.approx(0.5)


def test_describe_tau_canonical_interval_is_unbounded_below_and_above_extremes() -> None:
    below = describe_tau(-5.0, 4)
    assert below.canonical_interval[0] == -math.inf

    above = describe_tau(100.0, 4)
    assert above.canonical_interval[1] == math.inf


def test_describe_tau_justification_lists_every_reachable_value_and_its_outcome() -> None:
    report = describe_tau(0.3, 4)

    for value in report.reachable:
        assert str(value) in report.justification
    assert "ESCALATE" in report.justification
    assert "auto-label" in report.justification


# --- describe_tau: warning paths --------------------------------------------------


def test_describe_tau_warns_on_reachable_value() -> None:
    reachable = reachable_dispersions(4)

    report = describe_tau(reachable[1], 4)

    assert report.on_reachable_value is True
    assert any("exactly equals a reachable" in w for w in report.warnings)


def test_describe_tau_warns_when_dispersion_inert() -> None:
    reachable = reachable_dispersions(4)

    report = describe_tau(reachable[-1] + 1.0, 4)

    assert report.dispersion_inert is True
    assert any("dispersion term" in w for w in report.warnings)


def test_describe_tau_warns_when_not_on_canonical_midpoint() -> None:
    report = describe_tau(0.3, 4)  # interval (0.0, 0.5), midpoint 0.25

    assert any("behaviorally identical" in w for w in report.warnings)


def test_describe_tau_no_warnings_for_a_canonical_midpoint_tau() -> None:
    canonical = resolve_tau(0.0, 4)

    report = describe_tau(canonical, 4)

    assert report.warnings == ()


def test_describe_tau_no_spurious_midpoint_warning_when_interval_is_unbounded_above() -> None:
    reachable = reachable_dispersions(4)
    report = describe_tau(reachable[-1] + 1.0, 4)

    assert not any("behaviorally identical" in w for w in report.warnings)


# --- validate_tau: raise paths ----------------------------------------------------


def test_validate_tau_raises_for_non_finite_tau() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_tau(math.nan, 4)
    with pytest.raises(ValueError, match="finite"):
        validate_tau(math.inf, 4)


def test_validate_tau_raises_for_non_positive_tau_when_x_greater_than_one() -> None:
    with pytest.raises(ValueError, match="tau must be > 0"):
        validate_tau(0.0, 4)
    with pytest.raises(ValueError, match="tau must be > 0"):
        validate_tau(-0.1, 2)


def test_validate_tau_allows_non_positive_tau_when_x_is_one() -> None:
    report = validate_tau(0.0, 1)

    assert report.x == 1


def test_validate_tau_does_not_raise_for_merely_suspicious_tau() -> None:
    reachable = reachable_dispersions(4)

    # on_reachable_value and dispersion_inert are warning-only conditions.
    report = validate_tau(reachable[1], 4)
    assert report.on_reachable_value is True


def test_validate_tau_returns_the_same_report_as_describe_tau_for_valid_input() -> None:
    assert validate_tau(0.3, 4) == describe_tau(0.3, 4)


# --- resolve_tau: fractional policy -----------------------------------------------


@pytest.mark.parametrize("x", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("policy", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_resolve_tau_never_returns_a_reachable_value(x: int, policy: float) -> None:
    reachable = reachable_dispersions(x)

    tau = resolve_tau(policy, x)

    assert not any(math.isclose(tau, v, rel_tol=1e-9, abs_tol=1e-12) for v in reachable)


def test_resolve_tau_zero_policy_sits_in_the_tightest_regime() -> None:
    reachable = reachable_dispersions(4)

    tau = resolve_tau(0.0, 4)

    assert 0.0 < tau < reachable[1]


def test_resolve_tau_one_policy_makes_dispersion_inert() -> None:
    reachable = reachable_dispersions(4)

    tau = resolve_tau(1.0, 4)

    assert tau > max(reachable)
    assert describe_tau(tau, 4).dispersion_inert is True


def test_resolve_tau_raises_for_policy_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_tau(1.5, 4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_tau(-0.1, 4)


# --- resolve_tau: tie policy -------------------------------------------------------


def test_resolve_tau_escalate_ties_makes_a_clean_tie_escalate() -> None:
    from attest.ensemble.aggregate import dispersion

    tau = resolve_tau(TIE_POLICY_ESCALATE, 4)
    tie_vector = (-1, -1, 0, 0)

    assert dispersion(tie_vector) > tau


def test_resolve_tau_tolerate_ties_makes_a_clean_tie_not_escalate() -> None:
    from attest.ensemble.aggregate import dispersion

    tau = resolve_tau(TIE_POLICY_TOLERATE, 4)
    tie_vector = (-1, -1, 0, 0)

    assert dispersion(tie_vector) <= tau


def test_resolve_tau_tie_policy_never_returns_a_reachable_value() -> None:
    reachable = reachable_dispersions(6)

    for policy in (TIE_POLICY_ESCALATE, TIE_POLICY_TOLERATE):
        tau = resolve_tau(policy, 6)
        assert not any(math.isclose(tau, v, rel_tol=1e-9, abs_tol=1e-12) for v in reachable)


def test_resolve_tau_tie_policy_raises_for_odd_x() -> None:
    with pytest.raises(ValueError, match="even ensemble size"):
        resolve_tau(TIE_POLICY_ESCALATE, 5)


def test_resolve_tau_tie_policy_raises_for_small_domain() -> None:
    with pytest.raises(ValueError, match="at least 3 values"):
        resolve_tau(TIE_POLICY_ESCALATE, 4, domain=(-1, 1))


def test_resolve_tau_raises_for_unknown_string_policy() -> None:
    with pytest.raises(ValueError, match="unknown tie policy"):
        resolve_tau("bogus_policy", 4)  # type: ignore[arg-type]


# --- TauReport: to_dict / from_dict round trip ------------------------------------


def test_tau_report_round_trips_through_dict() -> None:
    report = describe_tau(0.3, 4)

    restored = TauReport.from_dict(report.to_dict())

    assert restored == report


def test_tau_report_round_trips_unbounded_canonical_interval() -> None:
    reachable = reachable_dispersions(4)
    report = describe_tau(reachable[-1] + 1.0, 4)
    assert report.canonical_interval[1] == math.inf

    payload = report.to_dict()
    assert payload["canonical_interval"][1] is None

    restored = TauReport.from_dict(payload)
    assert restored == report


def test_tau_report_round_trips_unbounded_below() -> None:
    report = describe_tau(-1.0, 4)
    assert report.canonical_interval[0] == -math.inf

    payload = report.to_dict()
    assert payload["canonical_interval"][0] is None

    restored = TauReport.from_dict(payload)
    assert restored == report
