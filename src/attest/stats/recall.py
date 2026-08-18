"""Screening recall estimated from the random recall-audit plane, with a conservative floor.

Recall can only be estimated from the random recall audit plane (see
`attest.planes.recall_audit`), never from adjudication or ad hoc review,
because it requires a probability sample of the screen-excluded population.
Definitions, per audit stratum h:

    n_h    number of screen-excluded records from stratum h drawn into the
           random recall audit and gold-checked.
    m_h    number of those n_h records that gold-checking revealed to be
           truly relevant, i.e. false negatives found in the audit sample.
    q_h    = m_h / n_h, the sample exclusion error rate.
    |U_h|  total number of screen-excluded records in stratum h -- the
           population the audit sample was drawn from.
    FN_h   = q_h * |U_h|, the estimated number of missed relevant records
           in stratum h, projecting the sample error rate onto the full
           population of excludes.

Screening recall is reported as a sensitivity:

    recall = TP / (TP + sum_h FN_h)

This is deliberately NOT the correct-exclusion rate `1 - q`: that answers a
different question ("how often was an exclusion decision correct?", which
is dominated by the large, easy majority of clearly-irrelevant excludes)
rather than "what fraction of everything actually relevant did screening
find?", which is what recall means and what a systematic review needs.

Because audit samples are typically small, a point estimate alone
overstates confidence. This module always returns a conservative recall
floor alongside the point estimate: for each stratum, the floor substitutes
a conservative upper bound on q_h for the raw q_h, using the rule-of-three
bound (q_h = 3 / n_h) when zero false negatives were observed (m_h == 0),
and the Wilson score upper bound otherwise. Zero observed events is exactly
where the Wilson interval is least trustworthy as an upper bound (it
degenerates toward the sample rate as m -> 0), so this module does not fall
back to it there; the rule-of-three bound is the standard, citable
approximation to a one-sided ~95% upper confidence bound on a zero-event
binomial rate (Hanley & Lippman-Hand, 1983; Eypasch et al., 1995) and is
used for that stratum regardless of the requested confidence level -- a
fixed worst-case convention, not a recomputed interval bound. Callers that
need a floor calibrated to a non-95% confidence level should treat this as
a documented approximation, not an exact one.

Both of those bounds assume sampling with replacement from an infinite
population, which overstates uncertainty once the audit sample covers a
non-trivial fraction of |U_h|. The floor corrects for this with the
standard SRSWOR finite-population correction, sqrt((|U_h| - n_h) / (|U_h| -
1)), which shrinks the bound's excess over the point estimate toward zero
as n_h approaches |U_h|. At a full-population audit (`audit-draw --size
all`, n_h == |U_h|) the correction is exactly 0, so the floor collapses to
the exact point estimate -- a census has no sampling uncertainty left to
bound, and q_h is then an exact count, not an estimate. At the tiny sampling
fractions these audits ordinarily draw, the correction is close to 1 and
this module's prior (uncorrected) behavior is preserved. Note that at m_h
== 0 the floor uses rule-of-three * FPC as a documented approximation to
the upper confidence bound, not the exact hypergeometric zero-event bound
(which would also reach 0 at a census); this approximation is only ever
anti-conservative below the tiny-fraction regime it was already used in,
and it agrees with the exact bound at both endpoints (tiny fraction and
census).

Because that approximation is only ever anti-conservative in a specific,
bounded regime rather than provably valid everywhere, this module also
computes an exact, design-based alternative: `exclusion_count_upper_bound`
finds the one-sided hypergeometric upper confidence limit on K_h directly
(the population count of missed-relevant records in stratum h), rather
than bounding a rate and rescaling it. This requires no separate
finite-population correction -- the hypergeometric distribution *is* the
exact finite-population sampling distribution -- and it collapses to
`m_h` exactly at a census, matching the heuristic floor's boundary
behavior with a provable bound rather than an approximation everywhere in
between. `stratified_recall`'s `exact_floor` combines these across strata
at Bonferroni-adjusted per-stratum confidence (dividing `1 - confidence`
by the stratum count), the standard simultaneous-coverage correction for
reporting several one-sided bounds together; with one stratum this
reduces to the unadjusted per-stratum bound.

`scipy` is imported locally within functions so that `contracts`,
`prefilter`, `ensemble`, and `provenance` remain importable without it
installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_RULE_OF_THREE_NUMERATOR = 3.0


class RecallError(ValueError):
    """Raised when recall inputs violate the audit design's invariants."""


@dataclass(frozen=True)
class Stratum:
    """One audit stratum's exclusion-error inputs.

    Attributes:
        name: Stratum identifier (e.g. a track or dispersion band).
        n: Number of screen-excluded records from this stratum drawn into
            the random recall audit and gold-checked (n_h).
        m: Number of those audited records that gold-checking revealed to
            be truly relevant, i.e. false negatives found in the audit
            sample (m_h).
        population: Total number of screen-excluded records in this
            stratum (|U_h|), the population the audit sample was drawn from.
    """

    name: str
    n: int
    m: int
    population: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise RecallError(f"stratum '{self.name}': n must be positive, got {self.n}")
        if not (0 <= self.m <= self.n):
            raise RecallError(
                f"stratum '{self.name}': m must be between 0 and n ({self.n}), got {self.m}"
            )
        if self.population < self.n:
            raise RecallError(
                f"stratum '{self.name}': population ({self.population}) must be >= "
                f"audit sample size n ({self.n})"
            )


@dataclass(frozen=True)
class RecallEstimate:
    """Screening recall point estimate and conservative floor for one or more strata.

    Attributes:
        point: TP / (TP + sum_h FN_h), the sensitivity point estimate.
        floor: TP / (TP + sum_h FN_h_floor), substituting each stratum's
            conservative q_h bound. Always <= `point`.
        ci: Interval on `point`, propagated from the per-stratum binomial
            uncertainty in q_h under the stratified design. See
            `stratified_recall` for the propagation method.
        true_positives: TP used in both `point` and `floor`.
        estimated_fn: Point estimate of sum_h FN_h.
        estimated_fn_floor: Conservative estimate of sum_h FN_h underlying `floor`.
        exact_floor: TP / (TP + sum_h K_h_upper), substituting each
            stratum's exact one-sided hypergeometric upper confidence bound
            on missed-relevant count (see `exclusion_count_upper_bound`),
            combined at Bonferroni-adjusted per-stratum confidence. A
            genuine design-based bound rather than `floor`'s asymptotic
            approximation -- always <= `point`, but not guaranteed to sit
            on either side of `floor`.
        estimated_fn_exact_floor: Sum of each stratum's exact K_h upper
            bound underlying `exact_floor`.
    """

    point: float
    floor: float
    ci: tuple[float, float]
    true_positives: int
    estimated_fn: float
    estimated_fn_floor: float
    exact_floor: float
    estimated_fn_exact_floor: float


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion.

    Uses the Wilson score interval (Wilson, 1927) rather than the normal
    (Wald) approximation, which is unreliable for small n or proportions
    near 0 or 1 -- exactly the regime recall audits operate in.

    Args:
        successes: Number of observed successes (e.g. m_h).
        n: Number of trials (e.g. n_h).
        confidence: Two-sided confidence level, in (0, 1). Defaults to 0.95.

    Returns:
        `(low, high)`, the bounds of the Wilson score interval, clipped to
        `[0, 1]`. `low <= successes / n <= high` always holds.

    Raises:
        RecallError: If `n <= 0`, `successes` is outside `[0, n]`, or
            `confidence` is not strictly between 0 and 1.
    """
    if n <= 0:
        raise RecallError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise RecallError(f"successes must be between 0 and n ({n}), got {successes}")
    if not (0.0 < confidence < 1.0):
        raise RecallError(f"confidence must be strictly between 0 and 1, got {confidence}")

    from scipy import stats as scipy_stats

    z = float(scipy_stats.norm.ppf(1 - (1 - confidence) / 2))

    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (low, high)


def exclusion_error_rate(stratum: Stratum) -> float:
    """Compute q_h = m_h / n_h, the sample exclusion error rate for a stratum."""
    return stratum.m / stratum.n


def _finite_population_correction(n: int, population: int) -> float:
    """SRSWOR correction sqrt((N - n) / (N - 1)): 1.0 at a tiny sampling
    fraction, 0.0 at a census (n >= N) or a degenerate population."""
    if population <= 1 or n >= population:
        return 0.0
    return math.sqrt((population - n) / (population - 1))


def exclusion_error_rate_floor(stratum: Stratum, confidence: float = 0.95) -> float:
    """Compute a conservative upper-bound estimate of q_h for the recall floor.

    Substitutes the rule-of-three bound (3 / n_h) when no false negatives
    were observed in the audit sample (m_h == 0); otherwise uses the Wilson
    score upper bound at `confidence`. See the module docstring for why the
    rule-of-three bound is used, and used as a fixed convention, at m_h == 0.
    That infinite-population bound is then scaled toward the point estimate
    by the SRSWOR finite-population correction, so the floor collapses to
    the exact rate `exclusion_error_rate(stratum)` for a census (n_h ==
    |U_h|) and tightens smoothly as the sampled fraction of |U_h| grows.

    Args:
        stratum: The audit stratum.
        confidence: Confidence level for the Wilson upper bound, used only
            when m_h > 0. Ignored when m_h == 0.

    Returns:
        The conservative (upper-bound) estimate of q_h, always >=
        `exclusion_error_rate(stratum)`.
    """
    point = exclusion_error_rate(stratum)
    if stratum.m == 0:
        infinite_floor = _RULE_OF_THREE_NUMERATOR / stratum.n
    else:
        _low, infinite_floor = wilson_interval(stratum.m, stratum.n, confidence=confidence)
    fpc = _finite_population_correction(stratum.n, stratum.population)
    return min(1.0, point + (infinite_floor - point) * fpc)


def hypergeometric_upper_bound(m: int, n: int, population: int, confidence: float = 0.95) -> int:
    """Exact one-sided upper confidence bound on a finite population's total relevant-count K.

    Given `m` relevant records found in a simple random sample of `n` drawn
    without replacement from a population of size `population`, returns the
    largest integer K in `[m, population]` such that

        P(X <= m | X ~ Hypergeometric(population, K, n)) >= 1 - confidence

    -- the finite-population, exact analog of a one-sided Clopper-Pearson
    upper bound: for any true population count above the returned K, seeing
    this few or fewer relevant records in the sample would have been rarer
    than `1 - confidence`. `P(X <= m | K)` is non-increasing in K (more
    relevant records in the population makes seeing few in the sample less
    likely), which is what makes the returned K well-defined and the binary
    search below correct.

    At `n == population` (a full census), this returns exactly `m`: sampling
    the whole population leaves no uncertainty about K, and this bound
    correctly reflects that with no separate finite-population correction
    needed -- the hypergeometric distribution already models the finite
    population directly.

    Args:
        m: Number of relevant records found in the sample (m_h).
        n: Sample size (n_h).
        population: Total population size the sample was drawn from (|U_h|).
        confidence: One-sided confidence level, in (0, 1). Defaults to 0.95.

    Returns:
        The exact upper confidence bound on K, as an integer count.

    Raises:
        RecallError: If `0 <= m <= n <= population` does not hold, or
            `confidence` is not strictly between 0 and 1.
    """
    if not (0 <= m <= n <= population):
        raise RecallError(
            f"require 0 <= m <= n <= population, got m={m}, n={n}, population={population}"
        )
    if not (0.0 < confidence < 1.0):
        raise RecallError(f"confidence must be strictly between 0 and 1, got {confidence}")

    from scipy import stats as scipy_stats

    alpha = 1 - confidence

    def survives(k: int) -> bool:
        return bool(scipy_stats.hypergeom.cdf(m, population, k, n) >= alpha)

    lo, hi = m, population
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if survives(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def exclusion_count_upper_bound(stratum: Stratum, confidence: float = 0.95) -> int:
    """Exact one-sided upper confidence bound on K_h, the total missed-relevant count in a stratum.

    Thin wrapper over `hypergeometric_upper_bound` for a `Stratum`'s own
    fields. Unlike `exclusion_error_rate_floor`, this bounds the absolute
    missed count directly rather than a rate that then gets multiplied back
    by `stratum.population` -- no separate finite-population correction is
    needed or applied.

    Args:
        stratum: The audit stratum.
        confidence: One-sided confidence level for the bound.

    Returns:
        The exact upper confidence bound on K_h.
    """
    return hypergeometric_upper_bound(
        stratum.m, stratum.n, stratum.population, confidence=confidence
    )


def estimated_missed_fn(
    stratum: Stratum, *, floor: bool = False, confidence: float = 0.95
) -> float:
    """Compute FN_h = q_h * |U_h|, the estimated number of missed relevant records in a stratum.

    Args:
        stratum: The audit stratum.
        floor: If True, use the conservative `exclusion_error_rate_floor`
            instead of the point estimate `exclusion_error_rate`.
        confidence: Confidence level passed through to
            `exclusion_error_rate_floor` when `floor` is True.

    Returns:
        The (point or floor) estimate of FN_h.
    """
    q = (
        exclusion_error_rate_floor(stratum, confidence=confidence)
        if floor
        else exclusion_error_rate(stratum)
    )
    return q * stratum.population


def stratified_recall(
    strata: Sequence[Stratum],
    true_positives: int,
    confidence: float = 0.95,
) -> RecallEstimate:
    """Combine per-stratum exclusion-error estimates into an overall screening recall.

    Point estimate: `recall = TP / (TP + sum_h FN_h)`. Floor: the same
    formula with each stratum's conservative q_h bound (see
    `exclusion_error_rate_floor`) substituted in.

    The confidence interval on `point` combines each stratum's Wilson
    interval on q_h into an interval on `sum_h FN_h`, via MOVER (Method Of
    Variance Estimates Recovery: Zou & Donner, 2008; Newcombe, 1998). Each
    stratum's Wilson interval `(q_h_low, q_h_high)` scales into an interval
    on `FN_h` as `(q_h_low * |U_h|, q_h_high * |U_h|)`; MOVER treats half of
    that interval's width as a standard-error surrogate for `FN_h` and
    combines strata, which are independent draws, by root-sum-of-squares:
    `SE(sum_h FN_h) = sqrt(sum_h SE(FN_h)^2)`. This is preferred over a
    closed-form Wald variance (`|U_h|^2 * q_h(1 - q_h) / n_h`) because the
    Wald form collapses to zero width whenever `q_h == 0` -- exactly the
    zero-observed-error stratum this module's floor logic exists to guard
    against -- while the Wilson interval it draws from remains well-behaved
    at that boundary. The combined half-width is added to and subtracted
    from the point estimate `sum_h FN_h` (not from the sum of Wilson
    centers, so the interval is always centered on the reported point
    estimate). Because `recall = TP / (TP + FN)` is monotonically
    decreasing in `FN`, the lower/upper bounds on `FN` map to the
    upper/lower bounds on recall respectively.

    `exact_floor` is a separate, exact alternative to `floor`: each
    stratum's `exclusion_count_upper_bound` is computed at Bonferroni-
    adjusted per-stratum confidence (`1 - (1 - confidence) / len(strata)`,
    so the strata's *simultaneous* one-sided coverage is still >=
    `confidence`) and combined by direct summation of the exact counts --
    no MOVER/half-width propagation needed, since each bound is already a
    single-sided limit, not a two-sided interval to recombine. `confidence`
    is interpreted one-sided here, unlike its two-sided use for `floor`'s
    Wilson interval and `ci` above: a one-sided reading is the natural, and
    tighter, interpretation for a bound that is only ever reported as an
    upper limit.

    Args:
        strata: One or more audit strata. Passing a single stratum reduces
            exactly to the single-stratum (unstratified) case, and the
            Bonferroni adjustment on `exact_floor` becomes a no-op.
        true_positives: TP: records included by screening that are truly
            relevant, from the confusion matrix against gold labels.
        confidence: Two-sided confidence level for `ci` and for the
            Wilson-based part of `floor`; one-sided simultaneous confidence
            level for `exact_floor` (see above). Defaults to 0.95.

    Returns:
        A `RecallEstimate` with point, floor, exact_floor, and interval.

    Raises:
        RecallError: If `strata` is empty or `true_positives` is negative.
    """
    if not strata:
        raise RecallError("stratified_recall requires at least one stratum")
    if true_positives < 0:
        raise RecallError(f"true_positives must be non-negative, got {true_positives}")

    fn_point = sum(estimated_missed_fn(s) for s in strata)
    fn_floor = sum(estimated_missed_fn(s, floor=True, confidence=confidence) for s in strata)

    fn_half_width_sq_sum = 0.0
    for s in strata:
        q_low, q_high = wilson_interval(s.m, s.n, confidence=confidence)
        half_width = (q_high - q_low) * s.population / 2
        fn_half_width_sq_sum += half_width**2
    fn_half_width = math.sqrt(fn_half_width_sq_sum)
    fn_low = max(0.0, fn_point - fn_half_width)
    fn_high = fn_point + fn_half_width

    bonferroni_confidence = 1 - (1 - confidence) / len(strata)
    fn_exact_floor = float(
        sum(exclusion_count_upper_bound(s, confidence=bonferroni_confidence) for s in strata)
    )

    point = _recall(true_positives, fn_point)
    floor = _recall(true_positives, fn_floor)
    exact_floor = _recall(true_positives, fn_exact_floor)
    # recall is decreasing in FN, so FN's lower/upper bounds give recall's upper/lower bounds.
    ci_high = _recall(true_positives, fn_low)
    ci_low = _recall(true_positives, fn_high)

    return RecallEstimate(
        point=point,
        floor=floor,
        ci=(ci_low, ci_high),
        true_positives=true_positives,
        estimated_fn=fn_point,
        estimated_fn_floor=fn_floor,
        exact_floor=exact_floor,
        estimated_fn_exact_floor=fn_exact_floor,
    )


def recall_with_floor(
    stratum: Stratum,
    true_positives: int,
    confidence: float = 0.95,
) -> RecallEstimate:
    """Compute screening recall point estimate and floor for a single audit stratum.

    Convenience entry point for the common single-stratum case; exactly
    equivalent to `stratified_recall([stratum], true_positives, confidence)`.

    Args:
        stratum: The single audit stratum.
        true_positives: TP: records included by screening that are truly relevant.
        confidence: Two-sided confidence level for `ci` and the floor.

    Returns:
        A `RecallEstimate` with point, floor, and interval.
    """
    return stratified_recall([stratum], true_positives, confidence=confidence)


def _recall(true_positives: int, false_negatives: float) -> float:
    total = true_positives + false_negatives
    if total <= 0:
        return 0.0
    return true_positives / total
