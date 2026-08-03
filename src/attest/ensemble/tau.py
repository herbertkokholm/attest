"""tau as a derived, validated, self-documenting quantity.

`Config.tau` (see `attest.provenance.config`) is a bare float threshold on
the sample standard deviation of a vote vector, used by
`attest.ensemble.aggregate.g`'s boundary+dispersion rule: escalate whenever
`s(R) > tau`. But on the ordinal domain ``{-1, 0, +1}``, `s(R)` cannot take
an arbitrary real value.

Proof (stated plainly, "bevis"): for `x` ratings drawn from a finite ordinal
domain, the sample standard deviation

    s(R) = sqrt(sum((r - mean(R)) ** 2 for r in R) / (x - 1))

depends on `R` only through the *multiset* of ratings cast (not their
order), and `R` ranges over a finite set of multisets: the number of
multisets of size `x` drawn from a domain of size `d` is
``comb(x + d - 1, x)``, all finite for finite `x` and `d`. Excluding the
*boundary* multisets (those containing both `-1` and `+1`, which escalate
unconditionally under `g` regardless of dispersion, per
`attest.ensemble.aggregate.is_boundary`) still leaves a finite,
enumerable set of non-boundary multisets, and therefore a finite,
enumerable set of *attainable* values of `s(R)` -- `reachable_dispersions`
below computes exactly this set.

Consequently, the escalation predicate ``s(R) > tau`` is a step function of
`tau`: as `tau` increases continuously, the predicate's truth value for
every attainable `s(R)` only ever changes at the finitely many points where
`tau` crosses an attainable value. Between two consecutive attainable
values, every `tau` in the open interval yields the exact same partition of
attainable dispersions into "escalates" / "does not escalate", and
therefore identical decisions for every possible vote vector of size `x`.
`tau` is therefore only ever *identified* up to the open interval between
two consecutive attainable values (its `canonical_interval`, computed by
`describe_tau`) -- not to its literal decimal value -- and `tau` is not
comparable in strength across different ensemble sizes `x`, since the
attainable set itself depends on `x` (see `reachable_dispersions` and the
`x`-sweep caveat in `attest.ablation.xsweep`).

Regimes for a given `x`: the attainable, non-boundary dispersion values
sorted ascending, `r_0 = 0 < r_1 < ... < r_n`, partition tau into `n + 1`
regimes: ``(-inf, r_0)``, ``[r_0, r_1)``, ..., ``[r_n, +inf)`` -- read as:
"at or above `r_i`, strictly below `r_{i+1}`" is one regime (recall `g`'s
comparison is strict `>`, so `tau == r_i` produces the exact same decisions
as any tau in `(r_i, r_{i+1})` -- neither escalates a vector whose
dispersion is exactly `r_i` -- and a *different* partition from any tau
just below `r_i`; see `TauReport.on_reachable_value`). Changing which
regime `tau` falls in is a real behavior change; moving `tau` within a
regime is not. A canonical, safe representative for a given regime is the
midpoint between its two bounding attainable values (`resolve_tau` always
returns such a midpoint, never a value exactly on an attainable value).

This module is pure: standard library only, no I/O, no network, and no
dependency on `attest.provenance.config` or `attest.vendors` -- it only
reuses `attest.ensemble.aggregate.is_boundary`, which is equally pure. None
of the functions here call `warnings.warn`; `describe_tau`/`validate_tau`
return warnings as text in `TauReport.warnings` for a caller (see
`attest.cli`) to decide whether and how to surface.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from attest.ensemble.aggregate import is_boundary

TIE_POLICY_ESCALATE = "escalate_ties"
TIE_POLICY_TOLERATE = "tolerate_ties"

TiePolicy = Literal["escalate_ties", "tolerate_ties"]

_DEFAULT_DOMAIN = (-1, 0, 1)
_ISCLOSE_REL_TOL = 1e-9
_ISCLOSE_ABS_TOL = 1e-12


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_ISCLOSE_REL_TOL, abs_tol=_ISCLOSE_ABS_TOL)


def _sample_variance(combo: Sequence[int]) -> Fraction:
    """Exact sample variance of `combo`, computed over `Fraction`s.

    Using exact rational arithmetic (rather than floats) until the very last
    step means two multisets whose variance is the same rational number
    always collapse to the same `Fraction` key before the single, final
    `math.sqrt` conversion to float -- so `reachable_dispersions` can never
    under- or over-count distinct attainable values due to floating-point
    noise in intermediate sums.
    """
    n = len(combo)
    if n <= 1:
        return Fraction(0)
    total = sum(combo)
    mean = Fraction(total, n)
    squared_sum = sum(((Fraction(r) - mean) ** 2 for r in combo), start=Fraction(0))
    return squared_sum / (n - 1)


def reachable_dispersions(
    x: int, domain: Sequence[int] = _DEFAULT_DOMAIN, *, include_boundary: bool = False
) -> tuple[float, ...]:
    """Enumerate every sample-SD value attainable by an `x`-vote vector on `domain`.

    Args:
        x: Ensemble size: the number of ratings in each candidate vote vector.
        domain: The ordinal domain ratings are drawn from, e.g. `(-1, 0, 1)`.
        include_boundary: If False (the default), multisets containing both
            `-1` and `+1` -- which escalate unconditionally under `g`'s
            boundary rule regardless of dispersion -- are excluded, so the
            returned set reflects only the dispersions `tau` actually gates.

    Returns:
        A sorted tuple of distinct attainable sample standard deviations.
        Always contains `0.0` for `x >= 1` (every domain value, repeated `x`
        times, is a non-boundary multiset with zero dispersion).

    Raises:
        ValueError: If `x < 1` or `domain` is empty.
    """
    if x < 1:
        raise ValueError(f"x must be >= 1, got {x}")
    distinct_domain = sorted(set(domain))
    if not distinct_domain:
        raise ValueError("domain must not be empty")

    variances: set[Fraction] = set()
    for combo in itertools.combinations_with_replacement(distinct_domain, x):
        if not include_boundary and is_boundary(combo):
            continue
        variances.add(_sample_variance(combo))

    return tuple(sorted({math.sqrt(float(variance)) for variance in variances}))


@dataclass(frozen=True)
class TauReport:
    """The full account of what a given `tau` does at a given ensemble size `x`.

    Attributes:
        tau: The tau value described.
        x: Ensemble size described.
        reachable: The sorted, distinct non-boundary sample-SD values
            attainable at this `x` (see `reachable_dispersions`).
        effective_breakpoint: The largest reachable value strictly below
            `tau` -- the real active threshold, ignoring the degenerate case
            where `tau` sits exactly on a reachable value (see
            `on_reachable_value`). None if no reachable value is below `tau`.
        canonical_interval: The open interval `(lo, hi)` between two
            consecutive reachable values that `tau` sits in, where `lo` is
            the largest reachable value `<= tau` (or `-inf` if none) and
            `hi` is the smallest reachable value `> tau` (or `+inf` if
            none). Any tau in this interval produces identical decisions.
        dispersion_inert: True if `tau` is at or above the maximum reachable
            value, so the dispersion term of `g`'s boundary+dispersion rule
            can never fire and only the boundary rule drives escalation.
        on_reachable_value: True if `tau` exactly equals a reachable value.
        justification: Human-readable derivation of the above for this `x`,
            reproducible by a reviewer without re-deriving the proof.
        warnings: Human-readable notes on suspicious-but-not-invalid `tau`
            choices (see `describe_tau`), for a caller to `warnings.warn`.
    """

    tau: float
    x: int
    reachable: tuple[float, ...]
    effective_breakpoint: float | None
    canonical_interval: tuple[float, float]
    dispersion_inert: bool
    on_reachable_value: bool
    justification: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return this report as a plain, JSON-serializable dict.

        `canonical_interval` endpoints of `+-inf` serialize as `None`
        (JSON has no infinity literal); `from_dict` reverses this.
        """
        lo, hi = self.canonical_interval
        return {
            "tau": self.tau,
            "x": self.x,
            "reachable": list(self.reachable),
            "effective_breakpoint": self.effective_breakpoint,
            "canonical_interval": [
                lo if math.isfinite(lo) else None,
                hi if math.isfinite(hi) else None,
            ],
            "dispersion_inert": self.dispersion_inert,
            "on_reachable_value": self.on_reachable_value,
            "justification": self.justification,
            "warnings": list(self.warnings),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> TauReport:
        """Reconstruct a `TauReport` from `to_dict`'s output."""
        lo_raw, hi_raw = payload["canonical_interval"]
        lo = lo_raw if lo_raw is not None else -math.inf
        hi = hi_raw if hi_raw is not None else math.inf
        return TauReport(
            tau=payload["tau"],
            x=payload["x"],
            reachable=tuple(payload["reachable"]),
            effective_breakpoint=payload["effective_breakpoint"],
            canonical_interval=(lo, hi),
            dispersion_inert=payload["dispersion_inert"],
            on_reachable_value=payload["on_reachable_value"],
            justification=payload["justification"],
            warnings=tuple(payload["warnings"]),
        )


def _build_justification(
    tau: float, x: int, domain: Sequence[int], reachable: tuple[float, ...], lo: float, hi: float
) -> str:
    lines = [
        f"x={x} ratings drawn from domain {tuple(domain)}: the sample SD of a "
        f"non-boundary vote vector can only take one of {len(reachable)} attainable "
        f"values: {reachable}.",
        f"The escalation predicate 's(R) > tau' with tau={tau} evaluates, at each "
        "attainable value, as:",
    ]
    for value in reachable:
        outcome = "ESCALATE (s > tau)" if value > tau else "auto-label (s <= tau)"
        lines.append(f"  s={value}: {outcome}")
    lines.append(
        f"Any tau in the open interval ({lo}, {hi}) produces this exact same partition "
        f"of attainable values, and therefore identical decisions, for x={x}."
    )
    return "\n".join(lines)


def describe_tau(tau: float, x: int, domain: Sequence[int] = _DEFAULT_DOMAIN) -> TauReport:
    """Pure computation of everything a given `tau` does at ensemble size `x`.

    Never raises for a merely suspicious `tau` (e.g. sitting exactly on a
    reachable value); it records those as `TauReport.warnings` for the
    caller to act on. See `validate_tau` for the raising variant.

    Args:
        tau: The tau value to describe.
        x: Ensemble size (see `reachable_dispersions`).
        domain: The ordinal domain (see `reachable_dispersions`).

    Returns:
        The full `TauReport` for `(tau, x, domain)`.

    Raises:
        ValueError: If `x < 1` or `domain` is empty (propagated from
            `reachable_dispersions`).
    """
    reachable = reachable_dispersions(x, domain)
    max_reachable = max(reachable)

    below = [v for v in reachable if v < tau]
    effective_breakpoint = max(below) if below else None

    at_or_below = [v for v in reachable if v <= tau]
    above = [v for v in reachable if v > tau]
    lo = max(at_or_below) if at_or_below else -math.inf
    hi = min(above) if above else math.inf

    dispersion_inert = tau >= max_reachable
    on_reachable_value = any(_isclose(tau, v) for v in reachable)

    report_warnings: list[str] = []
    if on_reachable_value:
        report_warnings.append(
            f"tau={tau} exactly equals a reachable dispersion value for x={x}; because "
            "g's escalation predicate is strict (s > tau), vote vectors with that exact "
            "dispersion do NOT escalate -- usually the opposite of intent. Consider "
            "resolve_tau() instead, which always returns a midpoint, never a reachable value."
        )
    if dispersion_inert:
        report_warnings.append(
            f"tau={tau} is >= the maximum reachable non-boundary dispersion "
            f"({max_reachable}) for x={x}; the dispersion term of g's boundary+dispersion "
            "rule can never fire here, and only the boundary rule (both -1 and +1 present) "
            "drives escalation."
        )
    if math.isfinite(hi):
        midpoint = (lo + hi) / 2
        if not _isclose(tau, midpoint):
            report_warnings.append(
                f"tau={tau} is behaviorally identical, for x={x}, to the canonical "
                f"midpoint tau={midpoint} of the open interval ({lo}, {hi}); consider "
                "using the canonical value for clarity."
            )

    justification = _build_justification(tau, x, domain, reachable, lo, hi)

    return TauReport(
        tau=tau,
        x=x,
        reachable=reachable,
        effective_breakpoint=effective_breakpoint,
        canonical_interval=(lo, hi),
        dispersion_inert=dispersion_inert,
        on_reachable_value=on_reachable_value,
        justification=justification,
        warnings=tuple(report_warnings),
    )


def validate_tau(tau: float, x: int, domain: Sequence[int] = _DEFAULT_DOMAIN) -> TauReport:
    """Validate `tau` and describe it, raising only on genuinely nonsensical input.

    Args:
        tau: The tau value to validate and describe.
        x: Ensemble size (see `reachable_dispersions`).
        domain: The ordinal domain (see `reachable_dispersions`).

    Returns:
        `describe_tau(tau, x, domain)`, if `tau` is not nonsensical.

    Raises:
        ValueError: If `tau` is not finite, or if `tau <= 0` while `x > 1`
            (dispersion is always zero for `x <= 1`, so a non-positive tau
            is harmless there). Also propagated from `reachable_dispersions`
            for a nonsensical `x`/`domain`.
    """
    if not math.isfinite(tau):
        raise ValueError(f"tau must be a finite number, got {tau}")
    if x > 1 and tau <= 0:
        raise ValueError(f"tau must be > 0 when x > 1 (x={x}), got {tau}")
    return describe_tau(tau, x, domain)


def _tie_dispersion(x: int, domain: Sequence[int]) -> float:
    if x % 2 != 0:
        raise ValueError(f"a tie policy requires an even ensemble size x, got {x}")
    distinct_domain = sorted(set(domain))
    if len(distinct_domain) < 3:
        raise ValueError(f"a tie policy requires a domain of at least 3 values, got {domain}")
    low, mid = distinct_domain[0], distinct_domain[1]
    # The "clean tie" is the maximally spread non-boundary split confined to
    # one side of the domain's midpoint: half the votes at the low end, half
    # at the middle value (e.g. half exclude, half uncertain).
    combo = (low,) * (x // 2) + (mid,) * (x // 2)
    return math.sqrt(float(_sample_variance(combo)))


def _resolve_tie_policy(policy: TiePolicy, x: int, domain: Sequence[int]) -> float:
    reachable = reachable_dispersions(x, domain)
    tie_value = _tie_dispersion(x, domain)

    if policy == TIE_POLICY_ESCALATE:
        # tau must sit strictly below tie_value so a clean-tie vector's
        # dispersion (== tie_value) satisfies s > tau and escalates.
        lower_candidates = [v for v in reachable if v < tie_value]
        lower = max(lower_candidates) if lower_candidates else 0.0
        return (lower + tie_value) / 2

    if policy == TIE_POLICY_TOLERATE:
        # tau must sit at or above tie_value so s > tau is false for it.
        higher_candidates = [v for v in reachable if v > tie_value]
        if higher_candidates:
            higher = min(higher_candidates)
            return (tie_value + higher) / 2
        return tie_value + 1.0

    raise ValueError(
        f"unknown tie policy {policy!r}: expected one of "
        f"{(TIE_POLICY_ESCALATE, TIE_POLICY_TOLERATE)}"
    )


def resolve_tau(
    policy: float | TiePolicy, x: int, domain: Sequence[int] = _DEFAULT_DOMAIN
) -> float:
    """Compute a raw `tau` from a declared policy, always positioned off any reachable value.

    Args:
        policy: Either a fraction in `[0, 1]` of the maximum attainable
            non-boundary dispersion at this `x` (0.0 is the tightest regime,
            1.0 makes dispersion inert), or one of `TIE_POLICY_ESCALATE` /
            `TIE_POLICY_TOLERATE`, selecting whether a clean, maximally
            split non-boundary tie (half the votes at the domain's low end,
            half at its middle value) escalates.
        x: Ensemble size (see `reachable_dispersions`).
        domain: The ordinal domain (see `reachable_dispersions`).

    Returns:
        A `tau` positioned at the midpoint of the intended interval between
        two reachable values (or `max_reachable + 1.0` when the intended
        regime is above every reachable value) -- by construction, never
        exactly on a reachable value, so `s(R) > tau` is unambiguous for
        every attainable dispersion.

    Raises:
        ValueError: If `policy` is a float outside `[0, 1]`, an unrecognized
            string, or a tie policy requested for an odd `x` or a domain
            with fewer than 3 values. Also propagated from
            `reachable_dispersions` for a nonsensical `x`/`domain`.
    """
    if isinstance(policy, str):
        return _resolve_tie_policy(policy, x, domain)

    if not (0.0 <= policy <= 1.0):
        raise ValueError(f"fractional policy must be in [0, 1], got {policy}")

    reachable = reachable_dispersions(x, domain)
    max_reachable = max(reachable)
    target = policy * max_reachable

    index = bisect.bisect_right(reachable, target)
    if index < len(reachable):
        lower = reachable[index - 1] if index > 0 else 0.0
        higher = reachable[index]
        return (lower + higher) / 2
    return max_reachable + 1.0 if max_reachable > 0 else 1.0


__all__ = [
    "TIE_POLICY_ESCALATE",
    "TIE_POLICY_TOLERATE",
    "TauReport",
    "TiePolicy",
    "describe_tau",
    "reachable_dispersions",
    "resolve_tau",
    "validate_tau",
]
