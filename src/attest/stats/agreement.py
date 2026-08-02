"""Ordinal Krippendorff's alpha over the ensemble vote matrix, and a pairwise vendor alpha matrix.

Krippendorff's alpha corrects raw agreement for chance agreement expected
under the observed value distribution. That correction is exactly what
makes a single alpha value hard to read in isolation under class imbalance:
the same raw agreement can correspond to very different alpha values
depending on how skewed the rating distribution is, and vice versa. Every
report in this module therefore returns raw percent agreement alongside
alpha rather than alpha alone.

A vendor that did not vote on a given record is a missing cell, not a 0
(related/uncertain) vote. Missing cells are represented as NaN in the
reliability matrix and excluded from both the alpha and raw-agreement
computations, per Krippendorff's original treatment of missing data: a unit
with fewer than two ratings contributes nothing to either statistic.

Alpha is computed with the `krippendorff` package's ordinal metric, which
implements Krippendorff's own distance function for ordinal scales: the
disagreement between two categories scales with how many observed values
fall between them (weighted by their frequency), not merely their rank
distance. `krippendorff` is imported locally within functions so that
`contracts`, `prefilter`, `ensemble`, and `provenance` remain importable
without it installed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from attest.ensemble.votes import VoteVector

ORDINAL_VALUE_DOMAIN = (-1, 0, 1)


class AgreementError(ValueError):
    """Raised when agreement cannot be computed from the given data."""


@dataclass(frozen=True)
class AgreementReport:
    """Krippendorff's alpha alongside raw percent agreement.

    Attributes:
        alpha: Chance-corrected ordinal agreement coefficient. 1.0 is
            perfect agreement, 0.0 is agreement at chance level; it can be
            negative when observed agreement is worse than chance.
        raw_agreement: Proportion of rater-pairs, among pairs of raters that
            both rated the same unit, whose ratings are identical. Not
            chance-corrected, so it is easily inflated by class imbalance --
            report it alongside alpha, never as a substitute.
        n_units: Number of units (records) in the reliability matrix.
        n_raters: Number of distinct raters (vendors) in the reliability matrix.
    """

    alpha: float
    raw_agreement: float
    n_units: int
    n_raters: int


def build_reliability_matrix(
    vote_vectors: Sequence[VoteVector],
) -> tuple[list[str], list[list[float]]]:
    """Build a raters-by-units reliability matrix from a batch of vote vectors.

    Args:
        vote_vectors: One vote vector per unit (record). Vendors that did
            not vote on a given record simply omit a `Vote` for it; they do
            not need to appear in every vector.

    Returns:
        `(vendors, matrix)`, where `vendors` is the sorted list of distinct
        vendor names across all vote vectors, and `matrix[i][j]` is
        `vendors[i]`'s rating for `vote_vectors[j]`, or `nan` if that vendor
        did not vote on that record.

    Raises:
        AgreementError: If `vote_vectors` is empty or contains no votes.
    """
    if not vote_vectors:
        raise AgreementError("cannot build a reliability matrix from zero vote vectors")
    vendors = sorted({vote.vendor for vv in vote_vectors for vote in vv.votes})
    if not vendors:
        raise AgreementError("no votes present in the given vote vectors")

    vendor_index = {vendor: i for i, vendor in enumerate(vendors)}
    matrix: list[list[float]] = [[math.nan] * len(vote_vectors) for _ in vendors]
    for unit_index, vv in enumerate(vote_vectors):
        for vote in vv.votes:
            matrix[vendor_index[vote.vendor]][unit_index] = float(vote.rating)
    return vendors, matrix


def raw_agreement(reliability_matrix: Sequence[Sequence[float]]) -> float:
    """Compute raw pairwise percent agreement over a reliability matrix.

    For each unit, every pair of raters that both rated it either agrees
    (identical rating) or disagrees; this is the fraction of such pairs,
    across all units, that agree. Unlike Krippendorff's alpha, this ignores
    chance correction and value prevalence entirely.

    Args:
        reliability_matrix: A raters-by-units matrix as built by
            `build_reliability_matrix`, with `nan` for missing cells.

    Returns:
        The proportion of agreeing rater-pairs, in [0, 1].

    Raises:
        AgreementError: If no unit has ratings from at least two raters.
    """
    n_raters = len(reliability_matrix)
    n_units = len(reliability_matrix[0]) if n_raters else 0

    agreeing_pairs = 0
    total_pairs = 0
    for unit_index in range(n_units):
        column = [
            reliability_matrix[rater_index][unit_index]
            for rater_index in range(n_raters)
            if not math.isnan(reliability_matrix[rater_index][unit_index])
        ]
        for a, b in itertools.combinations(column, 2):
            total_pairs += 1
            if a == b:
                agreeing_pairs += 1

    if total_pairs == 0:
        raise AgreementError(
            "no unit has ratings from at least two raters; raw agreement is undefined"
        )
    return agreeing_pairs / total_pairs


def krippendorff_alpha(
    reliability_matrix: Sequence[Sequence[float]],
    value_domain: Sequence[int] = ORDINAL_VALUE_DOMAIN,
) -> float:
    """Compute ordinal Krippendorff's alpha over a raters-by-units reliability matrix.

    Args:
        reliability_matrix: A raters-by-units matrix as built by
            `build_reliability_matrix`, with `nan` for missing cells.
        value_domain: The ordered set of valid ordinal categories. Defaults
            to the attest ordinal scale: exclude (-1), related/uncertain
            (0), include (+1).

    Returns:
        Krippendorff's alpha under the ordinal difference function.

    Raises:
        ValueError: Propagated from `krippendorff.alpha` if fewer than two
            raters rated any single unit (alpha is undefined with zero
            observed pairs) or if the matrix contains values outside
            `value_domain`.
    """
    from krippendorff import alpha as _kd_alpha

    return float(
        _kd_alpha(
            reliability_data=[list(row) for row in reliability_matrix],
            level_of_measurement="ordinal",
            value_domain=list(value_domain),
        )
    )


def agreement_report(
    vote_vectors: Sequence[VoteVector],
    value_domain: Sequence[int] = ORDINAL_VALUE_DOMAIN,
) -> AgreementReport:
    """Compute overall alpha and raw agreement across all vendors in one batch of votes.

    Args:
        vote_vectors: One vote vector per unit (record).
        value_domain: The ordered set of valid ordinal categories.

    Returns:
        An `AgreementReport` with alpha and raw agreement side by side.
    """
    vendors, matrix = build_reliability_matrix(vote_vectors)
    return AgreementReport(
        alpha=krippendorff_alpha(matrix, value_domain=value_domain),
        raw_agreement=raw_agreement(matrix),
        n_units=len(vote_vectors),
        n_raters=len(vendors),
    )


def pairwise_alpha(
    vote_vectors: Sequence[VoteVector],
    value_domain: Sequence[int] = ORDINAL_VALUE_DOMAIN,
) -> dict[str, float]:
    """Compute Krippendorff's alpha for every pair of vendors.

    Args:
        vote_vectors: One vote vector per unit (record).
        value_domain: The ordered set of valid ordinal categories.

    Returns:
        A mapping of `"vendorA|vendorB"` (vendors in sorted order) to that
        pair's alpha, matching the key format of
        `attest.contracts.validation_record.Agreement.pairwise`. A pair is
        omitted if it shares no unit rated by both vendors, since alpha is
        undefined for a pair with zero observed joint ratings.
    """
    vendors, matrix = build_reliability_matrix(vote_vectors)
    result: dict[str, float] = {}
    for i, j in itertools.combinations(range(len(vendors)), 2):
        key = f"{vendors[i]}|{vendors[j]}"
        try:
            result[key] = krippendorff_alpha([matrix[i], matrix[j]], value_domain=value_domain)
        except ValueError:
            continue
    return result
