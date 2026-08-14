"""Controlled ablation of ensemble size x over vendor subsets, on a frozen gold set.

This module answers a single causal question: as ensemble size x grows, how
do agreement, recall, precision, and escalation cost trade off, and which
vendors actually drive that trade-off? It runs entirely offline, on
`VoteVector`s already collected (e.g. via `attest.vendors.base.run_ensemble`
once, over the full candidate vendor pool) and a frozen, fully gold-labeled
dataset -- no network call happens here, only re-aggregation of stored votes
over vendor subsets.

This is deliberately independent of `attest.contracts.validation_record` (the
longitudinal, per-epoch self-validation record for a live ensemble
configuration) and of `attest.planes.recall_audit` (the online,
probability-sample-based recall estimator for a live configuration's
screen-excluded population). Both of those track a *running* configuration's
real-world performance over time; this module instead runs a one-off,
controlled comparison across hypothetical configurations on a dataset where
every gold label is already known. The two must never be conflated:
recall and precision here are computed by exact confusion counts against
gold labels on every record, not estimated from a sample, and under an
assumption specific to this controlled setting -- that an escalated
(human-adjudicated) record resolves to its gold label -- which does not
hold, and is not used, for the production recall estimate.

Ablation caveat on `tau`: `sweep` calls `attest.ensemble.aggregate.g` with a
single, fixed `tau` across every swept ensemble size `x'`. As
`attest.ensemble.tau` proves, `tau`'s effective meaning -- which regime of
the escalation step function it falls in -- depends on `x'`, since the set
of dispersion values attainable by an `x'`-vote vector changes with `x'`.
A fixed raw `tau` is therefore not a fixed "amount of strictness" across the
sweep: it may be firmly inside one regime at a small `x'` and effectively
inert (or on a knife's edge) at a larger one. This module deliberately does
NOT re-resolve `tau` per `x'` -- doing so would change `g`'s decisions for
existing callers and fixtures, which is out of scope for a sweep whose whole
point is to compare fixed-`tau` behavior across `x'`. Instead, each
`SubsetResult.tau_report` documents what the sweep's fixed `tau` actually
does at that result's own `x`, and `sweep` emits a single `warnings.warn`
when it spans more than one `x'` value, so callers reading raw metrics
across `x'` are not misled into treating `tau` as calibrated between them.
"""

from __future__ import annotations

import itertools
import math
import random
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from attest.ensemble.aggregate import AGGREGATION_BOUNDARY_DISPERSION, ZERO_POLICY_ESCALATE, g
from attest.ensemble.tau import TauReport, describe_tau
from attest.ensemble.votes import VoteVector
from attest.stats.agreement import AgreementError, agreement_report
from attest.stats.confusion import confusion_matrix

DEFAULT_MAX_SUBSETS_PER_X = 50


class AblationError(ValueError):
    """Raised when ablation sweep inputs are missing, inconsistent, or infeasible."""


@dataclass(frozen=True)
class LeaveOneOutContribution:
    """One vendor's marginal contribution to a vendor subset's metrics.

    Computed as `subset_metric - metric(subset without this vendor)`. For
    alpha, raw_agreement, recall, and precision, a positive delta means the
    vendor's presence improved that metric. For escalation_rate, a positive
    delta means the vendor's presence *raised* the escalation rate, i.e.
    increased the cost of the subset.

    Attributes:
        removed_vendor: The vendor whose marginal contribution this is.
        delta_alpha: Change in ordinal Krippendorff's alpha, or None if
            alpha is undefined for the subset with or without this vendor
            (fewer than two raters shared a rated record).
        delta_raw_agreement: Change in raw percent agreement, or None under
            the same conditions as `delta_alpha`.
        delta_recall: Change in recall against gold labels.
        delta_precision: Change in precision against gold labels.
        delta_escalation_rate: Change in escalation rate.
    """

    removed_vendor: str
    delta_alpha: float | None
    delta_raw_agreement: float | None
    delta_recall: float
    delta_precision: float
    delta_escalation_rate: float


@dataclass(frozen=True)
class SubsetResult:
    """Metrics for one (ensemble size x, vendor subset) point in the sweep.

    Attributes:
        x: Ensemble size: the number of vendors in `vendors`.
        vendors: The vendor subset this result was computed over, as a
            sorted tuple.
        alpha: Ordinal Krippendorff's alpha over this subset's votes on the
            gold set, or None if undefined (fewer than two raters shared a
            rated record).
        raw_agreement: Raw percent agreement, always reported alongside
            alpha rather than alpha alone (see `attest.stats.agreement`).
            None under the same conditions as `alpha`.
        recall: Recall against gold labels: TP / (TP + FN). TP/FN are
            computed from each record's final label -- the aggregation's
            auto-label where its vote vector doesn't escalate, or the gold
            label itself where it does (see module docstring).
        precision: Precision against gold labels: TP / (TP + FP), using the
            same final-label convention as `recall`.
        escalation_rate: Fraction of this subset's records whose vote
            vector escalated rather than auto-labeled.
        n_records: Number of gold-set records this subset had at least one
            vote on. Equal to the full gold set size unless some vendor in
            the pool has missing votes on some records.
        leave_one_out: Per-vendor marginal contribution of removing that
            vendor from this subset, keyed by the removed vendor's name.
        tau_report: What the sweep's single, fixed `tau` (see `sweep`'s
            `tau` argument) actually does at this subset's own `x` (see
            `attest.ensemble.tau.describe_tau`) -- not a re-resolved,
            per-`x` tau. `g` is called with the sweep's fixed `tau` for
            every subset, unchanged; this field only documents, after the
            fact, what that fixed value means at this particular `x`, since
            the same raw `tau` is not comparable in strength across `x`
            (see the module docstring's ablation caveat).
    """

    x: int
    vendors: tuple[str, ...]
    alpha: float | None
    raw_agreement: float | None
    recall: float
    precision: float
    escalation_rate: float
    n_records: int
    leave_one_out: dict[str, LeaveOneOutContribution]
    tau_report: TauReport


@dataclass(frozen=True)
class AblationReport:
    """Full ablation sweep result: one `SubsetResult` per (x, vendor subset) point.

    Attributes:
        candidate_vendors: The full candidate vendor pool the sweep was run
            over.
        results: Every computed subset's metrics, keyed by
            `(x, vendor_subset)` where `vendor_subset` is a sorted tuple of
            vendor names. Suitable for plotting quality (alpha, recall,
            precision) and cost (escalation_rate) against x to find the
            marginal-return "knee."
    """

    candidate_vendors: tuple[str, ...]
    results: dict[tuple[int, tuple[str, ...]], SubsetResult]


@dataclass(frozen=True)
class _Metrics:
    alpha: float | None
    raw_agreement: float | None
    recall: float
    precision: float
    escalation_rate: float
    n_records: int


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _filtered_votes(votes: Sequence[VoteVector], subset: frozenset[str]) -> list[VoteVector]:
    filtered: list[VoteVector] = []
    for vv in votes:
        kept = tuple(vote for vote in vv.votes if vote.vendor in subset)
        if kept:
            filtered.append(
                VoteVector(
                    record_id=vv.record_id, ensemble_config_id=vv.ensemble_config_id, votes=kept
                )
            )
    return filtered


def _compute_metrics(
    filtered: Sequence[VoteVector],
    truths: Mapping[str, int],
    aggregation: str,
    tau: float,
    zero_policy: str,
) -> _Metrics:
    if not filtered:
        raise AblationError("vendor subset has zero overlapping votes with the gold set")

    try:
        # krippendorff.alpha divides by the expected-disagreement sum, which
        # is exactly zero (and warns rather than raising) when every rater
        # in the subset produced only one category; the nan check below
        # already turns that into the documented "undefined" result, so the
        # warning is expected noise, not a signal, and is suppressed here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report = agreement_report(filtered)
        alpha: float | None = report.alpha
        raw_agreement: float | None = report.raw_agreement
        if alpha is not None and math.isnan(alpha):
            alpha = None
    except (AgreementError, ValueError):
        alpha, raw_agreement = None, None

    final_labels: dict[str, int] = {}
    escalations = 0
    for vv in filtered:
        if vv.record_id not in truths:
            raise AblationError(f"record '{vv.record_id}' has no gold label in truths")
        truth = truths[vv.record_id]
        decision = g(vv, aggregation=aggregation, tau=tau, zero_policy=zero_policy)
        if decision.escalate:
            escalations += 1
            # Perfect-adjudication assumption specific to this offline,
            # fully gold-labeled ablation -- see module docstring. An
            # escalated tie (mean exactly 0) follows this same assumption;
            # it needs no special case.
            final_labels[vv.record_id] = truth
        else:
            # A non-escalated decision must carry a non-None, non-zero
            # auto_label: g() never auto-commits the uncertain category
            # (ordinal 0) as a final label -- see attest.ensemble.aggregate's
            # module docstring -- so there is no third, undefined
            # disposition for a predicted 0 (or a missing label) to
            # silently fall into here. Checked explicitly (rather than via
            # `assert`) so a future regression in `g()` is caught even when
            # Python runs with optimizations enabled, instead of silently
            # miscounting this record as excluded.
            auto_label = decision.auto_label
            if auto_label is None or auto_label == 0:
                raise AblationError(
                    f"record '{vv.record_id}': g() returned auto_label={auto_label!r} for "
                    "a non-escalated decision, which must never happen (see "
                    "attest.ensemble.aggregate)"
                )
            final_labels[vv.record_id] = auto_label

    matrix = confusion_matrix(final_labels, truths)
    n_records = len(filtered)
    return _Metrics(
        alpha=alpha,
        raw_agreement=raw_agreement,
        recall=_ratio(matrix.tp, matrix.tp + matrix.fn),
        precision=_ratio(matrix.tp, matrix.tp + matrix.fp),
        escalation_rate=escalations / n_records,
        n_records=n_records,
    )


def _delta(subset_value: float | None, reduced_value: float | None) -> float | None:
    if subset_value is None or reduced_value is None:
        return None
    return subset_value - reduced_value


def _leave_one_out(
    subset: tuple[str, ...],
    subset_metrics: _Metrics,
    votes: Sequence[VoteVector],
    truths: Mapping[str, int],
    aggregation: str,
    tau: float,
    zero_policy: str,
) -> dict[str, LeaveOneOutContribution]:
    contributions: dict[str, LeaveOneOutContribution] = {}
    for removed in subset:
        reduced = tuple(v for v in subset if v != removed)
        reduced_filtered = _filtered_votes(votes, frozenset(reduced))
        reduced_metrics = _compute_metrics(reduced_filtered, truths, aggregation, tau, zero_policy)
        contributions[removed] = LeaveOneOutContribution(
            removed_vendor=removed,
            delta_alpha=_delta(subset_metrics.alpha, reduced_metrics.alpha),
            delta_raw_agreement=_delta(subset_metrics.raw_agreement, reduced_metrics.raw_agreement),
            delta_recall=subset_metrics.recall - reduced_metrics.recall,
            delta_precision=subset_metrics.precision - reduced_metrics.precision,
            delta_escalation_rate=subset_metrics.escalation_rate - reduced_metrics.escalation_rate,
        )
    return contributions


def _stratified_sample_subsets(
    vendors: Sequence[str], x: int, max_subsets: int, rng: random.Random
) -> list[tuple[str, ...]]:
    """Draw a stratified sample of `max_subsets` distinct size-x vendor subsets.

    "Stratified" here means: each draw weights vendor selection inversely to
    how often that vendor has already appeared among the subsets chosen so
    far, so that no vendor is systematically starved of coverage across the
    sample -- unlike uniform random sampling of combinations, which can by
    chance under-represent a vendor when the sample is small relative to the
    total number of combinations.
    """
    counts = {vendor: 0 for vendor in vendors}
    chosen: set[tuple[str, ...]] = set()
    max_attempts = max_subsets * 50
    attempts = 0
    while len(chosen) < max_subsets and attempts < max_attempts:
        attempts += 1
        pool = list(vendors)
        weights = [1.0 / (counts[v] + 1) for v in pool]
        picked: list[str] = []
        for _ in range(x):
            idx = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            picked.append(pool.pop(idx))
            weights.pop(idx)
        subset = tuple(sorted(picked))
        if subset in chosen:
            continue
        chosen.add(subset)
        for vendor in subset:
            counts[vendor] += 1
    return sorted(chosen)


def _select_subsets(
    vendors: Sequence[str], x: int, max_subsets: int, rng: random.Random
) -> list[tuple[str, ...]]:
    """Enumerate all size-x vendor subsets, or a documented stratified sample if infeasible."""
    total = math.comb(len(vendors), x)
    if total <= max_subsets:
        return [tuple(c) for c in itertools.combinations(vendors, x)]
    return _stratified_sample_subsets(vendors, x, max_subsets, rng)


def sweep(
    votes: Sequence[VoteVector],
    truths: Mapping[str, int],
    *,
    vendors: Sequence[str] | None = None,
    aggregation: str = AGGREGATION_BOUNDARY_DISPERSION,
    tau: float = 0.0,
    zero_policy: str = ZERO_POLICY_ESCALATE,
    max_subsets_per_x: int = DEFAULT_MAX_SUBSETS_PER_X,
    rng: random.Random | None = None,
) -> AblationReport:
    """Run a controlled ablation of ensemble size x over vendor subsets on a frozen gold set.

    For every ensemble size `x` from 2 to the size of the candidate vendor
    pool, enumerates every size-x vendor subset (or, when that is
    infeasible, draws a documented stratified sample of them -- see
    `_stratified_sample_subsets`) and computes, for each subset: ordinal
    Krippendorff's alpha, raw agreement, recall, precision, escalation
    rate, and each member vendor's leave-one-out marginal contribution.

    Args:
        votes: Stored vote vectors already collected for the full candidate
            vendor pool over the frozen gold set (e.g. from one prior call
            to `attest.vendors.base.run_ensemble`). No new votes are cast;
            every subset's metrics are computed by re-aggregating this
            stored data.
        truths: Mapping of record id to gold ordinal label, covering every
            record id present in `votes`.
        vendors: The candidate vendor pool to sweep over. Defaults to the
            sorted set of all vendor names appearing in `votes`.
        aggregation: Aggregation rule name passed to `attest.ensemble.aggregate.g`.
        tau: Dispersion threshold passed to `attest.ensemble.aggregate.g`.
        zero_policy: `attest.ensemble.aggregate.g`'s disposition of a
            would-be `auto_label == 0` decision, held fixed across every
            subset and every swept `x'` -- this is a property of the sweep
            call, not something the sweep varies. Defaults to
            `ZERO_POLICY_ESCALATE`, matching `g`'s own default.
        max_subsets_per_x: Maximum number of subsets to evaluate per
            ensemble size `x`. Sizes with more feasible subsets than this
            fall back to a stratified sample instead of exhaustive
            enumeration.
        rng: Source of randomness for stratified sampling; defaults to a
            fresh `random.Random()`. Pass a seeded `random.Random` for a
            reproducible sample.

    Returns:
        An `AblationReport` keyed by `(x, vendor subset)`.

    Raises:
        AblationError: If `votes` is empty, the candidate vendor pool has
            fewer than 2 vendors, `vendors` contains duplicates, or `truths`
            is missing a gold label for a record present in `votes`.
    """
    if not votes:
        raise AblationError("cannot run an ablation sweep over zero vote vectors")

    if vendors is not None:
        pool = tuple(vendors)
    else:
        pool = tuple(sorted({vote.vendor for vv in votes for vote in vv.votes}))
    if len(set(pool)) != len(pool):
        raise AblationError(f"candidate vendor pool contains duplicates: {pool}")
    if len(pool) < 2:
        raise AblationError(
            f"ablation sweep requires at least 2 candidate vendors, got {len(pool)}"
        )

    missing_truths = {vv.record_id for vv in votes} - set(truths)
    if missing_truths:
        raise AblationError(f"missing gold labels for record id(s): {sorted(missing_truths)}")

    active_rng = rng if rng is not None else random.Random()

    if len(pool) > 2:
        # More than one x' is swept (x ranges 2..len(pool)), so the single,
        # fixed tau below is not comparable in strength across x' -- see the
        # module docstring's ablation caveat. Warn once per sweep call,
        # rather than once per (x, subset), to avoid warning-spamming a
        # sweep that evaluates many subsets.
        warnings.warn(
            f"sweep() spans ensemble sizes x=2..{len(pool)} with a single fixed tau={tau}; "
            "tau's effective strictness is not comparable across x' (see "
            "attest.ensemble.tau) -- see each SubsetResult.tau_report for what tau "
            "actually does at that result's own x.",
            stacklevel=2,
        )

    tau_reports_by_x: dict[int, TauReport] = {}
    results: dict[tuple[int, tuple[str, ...]], SubsetResult] = {}
    for x in range(2, len(pool) + 1):
        tau_report = tau_reports_by_x.setdefault(x, describe_tau(tau, x))
        for subset in _select_subsets(pool, x, max_subsets_per_x, active_rng):
            filtered = _filtered_votes(votes, frozenset(subset))
            metrics = _compute_metrics(filtered, truths, aggregation, tau, zero_policy)
            leave_one_out = _leave_one_out(
                subset, metrics, votes, truths, aggregation, tau, zero_policy
            )
            results[(x, subset)] = SubsetResult(
                x=x,
                vendors=subset,
                alpha=metrics.alpha,
                raw_agreement=metrics.raw_agreement,
                recall=metrics.recall,
                precision=metrics.precision,
                escalation_rate=metrics.escalation_rate,
                n_records=metrics.n_records,
                leave_one_out=leave_one_out,
                tau_report=tau_report,
            )

    return AblationReport(candidate_vendors=pool, results=results)


__all__ = [
    "DEFAULT_MAX_SUBSETS_PER_X",
    "AblationError",
    "AblationReport",
    "LeaveOneOutContribution",
    "SubsetResult",
    "sweep",
]
