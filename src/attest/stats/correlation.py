"""Conditional error correlation on a gold set: the correlated-false-negative threat to recall.

An ensemble's redundancy is only useful if member errors are roughly
independent -- if one vendor misses a relevant record because another vendor
is likely to catch it. This module measures whether that assumption holds,
using the error indicator e = 1[pred != truth] for each ensemble member
against gold labels.

Two vendors' *unconditional* error correlation is a weak diagnostic: it
mixes together disagreement on clearly-excluded records (where correlated
errors are harmless, since both vendors correctly exclude) with disagreement
on truly relevant records (where a correlated error is a correlated missed
inclusion). Only the latter threatens recall. This module therefore exposes
`conditional_fn_correlation`, which restricts the error indicators to gold
records with truth == +1 (truly relevant) before computing correlation --
on that subset, e = 1[pred != truth] is exactly "this vendor produced a
false negative on this record."

`scipy` is imported locally within functions so that `contracts`,
`prefilter`, `ensemble`, and `provenance` remain importable without it
installed.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from attest.ensemble.votes import VoteVector


class CorrelationError(ValueError):
    """Raised when inputs to a correlation computation are inconsistent."""


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson correlation of two vendors' binary error indicators, with the joint counts.

    Attributes:
        correlation: Pearson correlation coefficient of the two 0/1 error
            vectors, in [-1, 1]. `None` if undefined: fewer than two paired
            records, or either vendor's error indicator has zero variance
            over the subset (it made either no errors or errors on every
            record), in which case correlation is mathematically undefined
            rather than zero.
        n: Number of records the correlation was computed over.
        both: Records where both vendors' indicators were True (e.g., both
            produced a false negative, for `conditional_fn_correlation`).
        only_a: Records where only vendor A's indicator was True.
        only_b: Records where only vendor B's indicator was True.
        neither: Records where neither indicator was True.

    `both`, `only_a`, `only_b`, and `neither` always sum to `n` and are
    always well-defined -- including when `correlation` is `None` -- since a
    sparse stratum's joint counts (e.g. "both missed the same 2 of 3
    relevant records") are exactly the evidence a reader needs to judge an
    undefined correlation, rather than that pair silently vanishing from
    the report.
    """

    correlation: float | None
    n: int
    both: int = 0
    only_a: int = 0
    only_b: int = 0
    neither: int = 0


def error_indicators(predictions: Sequence[int], truths: Sequence[int]) -> list[bool]:
    """Compute e = 1[pred != truth] for one vendor over a set of gold-labeled records.

    Args:
        predictions: The vendor's predicted labels.
        truths: Gold labels, same order and length as `predictions`.

    Returns:
        A list of error indicators, one per record.

    Raises:
        CorrelationError: If `predictions` and `truths` differ in length.
    """
    if len(predictions) != len(truths):
        raise CorrelationError(
            f"predictions and truths must have equal length, got {len(predictions)} "
            f"and {len(truths)}"
        )
    return [p != t for p, t in zip(predictions, truths, strict=True)]


def relevant_mask(truths: Sequence[int]) -> list[bool]:
    """Return a boolean mask selecting gold records that are truly relevant (truth == +1)."""
    return [t == 1 for t in truths]


def error_correlation(errors_a: Sequence[bool], errors_b: Sequence[bool]) -> CorrelationResult:
    """Compute the Pearson correlation between two vendors' binary error indicators.

    For 0/1-valued indicators this is exactly the phi coefficient. This is
    the general building block; `conditional_fn_correlation` is the
    recall-relevant specialization that restricts the indicators to truly
    relevant gold records before calling this.

    Args:
        errors_a: Vendor A's error indicators.
        errors_b: Vendor B's error indicators, same order and length.

    Returns:
        A `CorrelationResult`. `correlation` is `None` when undefined (see
        `CorrelationResult`), never silently coerced to 0.0.

    Raises:
        CorrelationError: If `errors_a` and `errors_b` differ in length.
    """
    if len(errors_a) != len(errors_b):
        raise CorrelationError(
            f"errors_a and errors_b must have equal length, got {len(errors_a)} and {len(errors_b)}"
        )
    n = len(errors_a)
    both = sum(1 for a, b in zip(errors_a, errors_b, strict=True) if a and b)
    only_a = sum(1 for a, b in zip(errors_a, errors_b, strict=True) if a and not b)
    only_b = sum(1 for a, b in zip(errors_a, errors_b, strict=True) if b and not a)
    neither = n - both - only_a - only_b
    joint_counts = {"both": both, "only_a": only_a, "only_b": only_b, "neither": neither}

    if n < 2:
        return CorrelationResult(correlation=None, n=n, **joint_counts)

    x = [1.0 if e else 0.0 for e in errors_a]
    y = [1.0 if e else 0.0 for e in errors_b]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return CorrelationResult(correlation=None, n=n, **joint_counts)

    from scipy import stats as scipy_stats

    r, _p_value = scipy_stats.pearsonr(x, y)
    return CorrelationResult(correlation=float(r), n=n, **joint_counts)


def conditional_fn_correlation(
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    truths: Sequence[int],
) -> CorrelationResult:
    """Correlate two vendors' false-negative errors, restricted to truly relevant gold records.

    This is the quantity that actually threatens recall. Restricting to
    truth == +1 records means each error indicator is exactly "this vendor
    wrongly failed to include this relevant record" -- a false negative. A
    high positive correlation here means the ensemble's redundancy is
    illusory on the class that matters: when one vendor misses a relevant
    record, the other is likely to miss it too.

    Args:
        predictions_a: Vendor A's predicted labels over the gold set.
        predictions_b: Vendor B's predicted labels over the same gold set,
            same order and length as `predictions_a`.
        truths: Gold labels over the same records, same order.

    Returns:
        A `CorrelationResult` computed only over the subset of records with
        `truths == +1`.

    Raises:
        CorrelationError: If `predictions_a`, `predictions_b`, and `truths`
            do not all have equal length.
    """
    if not (len(predictions_a) == len(predictions_b) == len(truths)):
        raise CorrelationError(
            "predictions_a, predictions_b, and truths must all have equal length, got "
            f"{len(predictions_a)}, {len(predictions_b)}, {len(truths)}"
        )
    mask = relevant_mask(truths)
    errors_a = [p != t for p, t, keep in zip(predictions_a, truths, mask, strict=True) if keep]
    errors_b = [p != t for p, t, keep in zip(predictions_b, truths, mask, strict=True) if keep]
    return error_correlation(errors_a, errors_b)


def build_predictions_by_vendor(
    votes: Sequence[VoteVector], truths: Mapping[str, int]
) -> tuple[dict[str, list[int]], list[int]]:
    """Reshape vote vectors into per-vendor prediction lists aligned for `pairwise_fn_correlation`.

    Restricted to gold-labeled records that every vendor present in `votes`
    cast a vote on, since `pairwise_fn_correlation` requires equal-length,
    aligned sequences per vendor.

    Args:
        votes: One vote vector per record.
        truths: Mapping of record id to gold ordinal label.

    Returns:
        `(predictions_by_vendor, ordered_truths)`: a mapping of vendor name
        to that vendor's predicted labels, and the gold labels in the same
        record order, both restricted to the fully-covered gold record ids.
    """
    by_record: dict[str, dict[str, int]] = {
        vv.record_id: {vote.vendor: vote.rating for vote in vv.votes} for vv in votes
    }
    vendors = sorted({vendor for ratings in by_record.values() for vendor in ratings})
    gold_ids = sorted(
        record_id
        for record_id in truths
        if record_id in by_record and all(vendor in by_record[record_id] for vendor in vendors)
    )
    predictions = {
        vendor: [by_record[record_id][vendor] for record_id in gold_ids] for vendor in vendors
    }
    ordered_truths = [truths[record_id] for record_id in gold_ids]
    return predictions, ordered_truths


def pairwise_fn_correlation(
    predictions_by_vendor: Mapping[str, Sequence[int]],
    truths: Sequence[int],
) -> dict[str, CorrelationResult]:
    """Compute conditional FN correlation for every pair of vendors.

    Args:
        predictions_by_vendor: Mapping of vendor name to that vendor's
            predicted labels over the gold set, all in the same record order.
        truths: Gold labels over the same records, same order.

    Returns:
        A mapping of `"vendorA|vendorB"` (vendors in sorted order) to that
        pair's `CorrelationResult`, matching the key format of
        `attest.contracts.validation_record.ErrorCorrelation.pairwise_fn_on_relevant`.
    """
    vendors = sorted(predictions_by_vendor)
    result: dict[str, CorrelationResult] = {}
    for a, b in itertools.combinations(vendors, 2):
        key = f"{a}|{b}"
        result[key] = conditional_fn_correlation(
            predictions_by_vendor[a], predictions_by_vendor[b], truths
        )
    return result
