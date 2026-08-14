"""Tests for attest.stats: ordinal Krippendorff alpha, conditional FN correlation, recall."""

from __future__ import annotations

import math

import pytest

from attest.ensemble.votes import VoteVector, build_vote_vector
from attest.stats.agreement import (
    AgreementError,
    agreement_report,
    build_reliability_matrix,
    krippendorff_alpha,
    pairwise_alpha,
    raw_agreement,
)
from attest.stats.confusion import RELEVANT_LABEL, ConfusionMatrix, confusion_matrix
from attest.stats.correlation import (
    CorrelationError,
    build_predictions_by_vendor,
    conditional_fn_correlation,
    error_correlation,
    error_indicators,
    pairwise_fn_correlation,
)
from attest.stats.recall import (
    RecallError,
    Stratum,
    exclusion_error_rate,
    exclusion_error_rate_floor,
    recall_with_floor,
    stratified_recall,
    wilson_interval,
)
from attest.stats.recall import _finite_population_correction as fpc

# --- agreement.py -----------------------------------------------------------
#
# Hand-computed reference: 2 raters, 4 units, ratings (r1, r2):
#   u1: (-1, -1)   u2: (0, 0)   u3: (1, 1)   u4: (-1, 1)
#
# Coincidence matrix o (value order [-1, 0, 1]), from each unit's
# n_v-vector contribution ((n_c*n_c' - diag) / (n_unit - 1)):
#   u1 -> o[0][0] += 2   u2 -> o[1][1] += 2   u3 -> o[2][2] += 2
#   u4 (n_vec=(1,0,1))  -> o[0][2] += 1, o[2][0] += 1
# n_v = column sums = (3, 2, 3); N = sum(n_v) = 8
# random coincidences e = (outer(n_v, n_v) - diag(n_v)) / (N - 1), e.g.
#   e[0][2] = (9 - 0) / 7 = 9/7,  e[0][1] = (6 - 0) / 7 = 6/7
# ordinal distances (value_domain=(-1, 0, 1), indices 0,1,2):
#   d[0][1] = d[1][2] = (5 - 2.5)^2 = 6.25;  d[0][2] = (8 - 3)^2 = 25
# observed disagreement sum (o * d) = 2*(1*25) = 50  (only u4's off-diagonal
#   cells contribute; all perfectly-agreeing units land on d==0 diagonal)
# expected disagreement sum (e * d) = 4*(6/7 * 6.25) + 2*(9/7 * 25) = 600/7
# alpha = 1 - 50 / (600/7) = 1 - 7/12 = 5/12


def _vv(record_id: str, ratings: dict[str, int]) -> VoteVector:
    return build_vote_vector(record_id, "cfg", ratings)


def _hand_matrix() -> list[list[float]]:
    votes = [
        _vv("u1", {"r1": -1, "r2": -1}),
        _vv("u2", {"r1": 0, "r2": 0}),
        _vv("u3", {"r1": 1, "r2": 1}),
        _vv("u4", {"r1": -1, "r2": 1}),
    ]
    _vendors, matrix = build_reliability_matrix(votes)
    return matrix


def test_alpha_matches_hand_computed_value_on_tiny_matrix() -> None:
    matrix = _hand_matrix()

    assert krippendorff_alpha(matrix) == pytest.approx(5 / 12)


def test_raw_agreement_matches_hand_computed_value_on_tiny_matrix() -> None:
    matrix = _hand_matrix()

    # 4 units, 1 rater-pair each; 3 agree (u1, u2, u3), 1 disagrees (u4).
    assert raw_agreement(matrix) == pytest.approx(0.75)


def test_perfect_agreement_gives_alpha_one() -> None:
    votes = [
        _vv("u1", {"r1": -1, "r2": -1}),
        _vv("u2", {"r1": 0, "r2": 0}),
        _vv("u3", {"r1": 1, "r2": 1}),
    ]
    _vendors, matrix = build_reliability_matrix(votes)

    assert krippendorff_alpha(matrix) == pytest.approx(1.0)
    assert raw_agreement(matrix) == pytest.approx(1.0)


def test_pairwise_alpha_matches_overall_alpha_for_two_vendors() -> None:
    votes = [
        _vv("u1", {"r1": -1, "r2": -1}),
        _vv("u2", {"r1": 0, "r2": 0}),
        _vv("u3", {"r1": 1, "r2": 1}),
        _vv("u4", {"r1": -1, "r2": 1}),
    ]

    pairwise = pairwise_alpha(votes)

    assert pairwise == {"r1|r2": pytest.approx(5 / 12)}


def test_agreement_report_bundles_alpha_and_raw_agreement() -> None:
    votes = [
        _vv("u1", {"r1": -1, "r2": -1}),
        _vv("u2", {"r1": 0, "r2": 0}),
        _vv("u3", {"r1": 1, "r2": 1}),
        _vv("u4", {"r1": -1, "r2": 1}),
    ]

    report = agreement_report(votes)

    assert report.alpha == pytest.approx(5 / 12)
    assert report.raw_agreement == pytest.approx(0.75)
    assert report.n_units == 4
    assert report.n_raters == 2


def test_missing_votes_are_excluded_not_treated_as_zero() -> None:
    # r3 only votes on u1 and u3; its missing vote on u2 must not count as a
    # 0 (related/uncertain) rating.
    votes = [
        _vv("u1", {"r1": -1, "r2": -1, "r3": -1}),
        _vv("u2", {"r1": 0, "r2": 0}),
        _vv("u3", {"r1": 1, "r2": 1, "r3": 1}),
    ]

    report = agreement_report(votes)

    assert report.n_raters == 3
    assert report.alpha == pytest.approx(1.0)


def test_build_reliability_matrix_rejects_empty_input() -> None:
    with pytest.raises(AgreementError):
        build_reliability_matrix([])


def test_krippendorff_alpha_raises_when_every_rater_used_only_one_category() -> None:
    # Zero variance in the data: both observed and expected (chance-corrected)
    # disagreement are exactly zero, so alpha is 0/0 -- undefined, not 1.0.
    votes = [
        _vv("u1", {"r1": 1, "r2": 1}),
        _vv("u2", {"r1": 1, "r2": 1}),
        _vv("u3", {"r1": 1, "r2": 1}),
    ]
    _vendors, matrix = build_reliability_matrix(votes)

    with pytest.raises(AgreementError, match="undefined"):
        krippendorff_alpha(matrix)


def test_agreement_report_alpha_is_none_when_undefined_but_raw_agreement_survives() -> None:
    votes = [
        _vv("u1", {"r1": 1, "r2": 1}),
        _vv("u2", {"r1": 1, "r2": 1}),
        _vv("u3", {"r1": 1, "r2": 1}),
    ]

    report = agreement_report(votes)

    assert report.alpha is None
    assert report.raw_agreement == pytest.approx(1.0)
    assert report.n_units == 3
    assert report.n_raters == 2


def test_pairwise_alpha_omits_a_pair_that_agreed_on_everything() -> None:
    # r1/r2 agree on every unit (alpha undefined, 0/0) and must be omitted,
    # exactly like a pair sharing zero units -- never surfaced as NaN.
    votes = [
        _vv("u1", {"r1": 1, "r2": 1, "r3": -1}),
        _vv("u2", {"r1": 1, "r2": 1, "r3": 1}),
    ]

    pairwise = pairwise_alpha(votes)

    assert "r1|r2" not in pairwise
    assert "r1|r3" in pairwise
    assert "r2|r3" in pairwise


# --- correlation.py ----------------------------------------------------------
#
# Hand-computed reference for the 5 truly-relevant records (truth == 1):
#   errors_a = [T, T, F, F, F] -> x = [1, 1, 0, 0, 0]
#   errors_b = [T, F, T, F, F] -> y = [1, 0, 1, 0, 0]
# n=5, sum(x)=2, sum(y)=2, sum(xy)=1, sum(x^2)=2, sum(y^2)=2
# r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2)-sum(x)^2)*(n*sum(y^2)-sum(y)^2))
#   = (5*1 - 2*2) / sqrt((10-4)*(10-4)) = 1 / 6


def test_error_indicators_computes_pred_neq_truth() -> None:
    assert error_indicators([1, -1, 0], [1, 1, 0]) == [False, True, False]


def test_error_indicators_rejects_length_mismatch() -> None:
    with pytest.raises(CorrelationError):
        error_indicators([1, -1], [1])


def test_conditional_fn_correlation_matches_hand_computed_value() -> None:
    # Records 0-4 are truly relevant (truth == 1); records 5-6 are not, and
    # are deliberately crafted so that including them would change the
    # correlation -- confirming the restriction to truth == +1 actually applies.
    truths = [1, 1, 1, 1, 1, -1, 0]
    predictions_a = [-1, -1, 1, 1, 1, 1, -1]
    predictions_b = [-1, 1, -1, 1, 1, -1, 1]

    result = conditional_fn_correlation(predictions_a, predictions_b, truths)

    assert result.n == 5
    assert result.correlation == pytest.approx(1 / 6)


def test_conditional_fn_correlation_rejects_length_mismatch() -> None:
    with pytest.raises(CorrelationError):
        conditional_fn_correlation([1, -1], [1, -1, 0], [1, 1, 1])


def test_error_correlation_identical_errors_gives_plus_one() -> None:
    result = error_correlation([True, True, False, False], [True, True, False, False])

    assert result.correlation == pytest.approx(1.0)


def test_error_correlation_opposite_errors_gives_minus_one() -> None:
    result = error_correlation([True, True, False, False], [False, False, True, True])

    assert result.correlation == pytest.approx(-1.0)


def test_error_correlation_zero_variance_is_undefined_not_zero() -> None:
    # Vendor A errs on every record: zero variance, correlation undefined.
    result = error_correlation([True, True, True], [True, False, True])

    assert result.correlation is None
    assert result.n == 3


def test_error_correlation_fewer_than_two_records_is_undefined() -> None:
    result = error_correlation([True], [False])

    assert result.correlation is None


def test_pairwise_fn_correlation_covers_all_vendor_pairs() -> None:
    truths = [1, 1, 1, 1]
    predictions_by_vendor = {
        "openai": [1, 1, -1, -1],
        "anthropic": [1, -1, 1, -1],
        "gemini": [1, 1, 1, 1],
    }

    result = pairwise_fn_correlation(predictions_by_vendor, truths)

    assert set(result) == {"anthropic|gemini", "anthropic|openai", "gemini|openai"}
    # gemini never errs -> zero variance -> undefined against anyone.
    assert result["anthropic|gemini"].correlation is None
    assert result["gemini|openai"].correlation is None


def test_build_predictions_by_vendor_aligns_and_restricts_to_fully_covered_gold_ids() -> None:
    # "r2" is missing "b"'s vote, so it must be dropped even though it has a
    # gold label -- pairwise_fn_correlation requires every vendor to have an
    # aligned entry for every included record.
    votes = [
        _vv("r1", {"a": 1, "b": -1}),
        _vv("r2", {"a": -1}),
        _vv("r3", {"a": 1, "b": 1}),
    ]
    truths = {"r1": 1, "r2": -1, "r3": 1}

    predictions, ordered_truths = build_predictions_by_vendor(votes, truths)

    assert predictions == {"a": [1, 1], "b": [-1, 1]}
    assert ordered_truths == [1, 1]


def test_build_predictions_by_vendor_ignores_records_with_no_gold_label() -> None:
    votes = [_vv("r1", {"a": 1}), _vv("ungoverned", {"a": -1})]
    truths = {"r1": 1}

    predictions, ordered_truths = build_predictions_by_vendor(votes, truths)

    assert predictions == {"a": [1]}
    assert ordered_truths == [1]


# --- recall.py ----------------------------------------------------------------


def test_exclusion_error_rate_is_m_over_n() -> None:
    stratum = Stratum(name="a", n=40, m=5, population=400)

    assert exclusion_error_rate(stratum) == pytest.approx(5 / 40)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    for successes, n in [(5, 20), (0, 20), (20, 20), (1, 3), (50, 100)]:
        low, high = wilson_interval(successes, n)
        p_hat = successes / n

        assert low <= p_hat <= high
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_rejects_invalid_inputs() -> None:
    with pytest.raises(RecallError):
        wilson_interval(1, 0)
    with pytest.raises(RecallError):
        wilson_interval(-1, 10)
    with pytest.raises(RecallError):
        wilson_interval(5, 10, confidence=1.0)


def test_recall_point_vs_floor_differ_on_zero_cell_stratum() -> None:
    # m == 0 -> floor uses the rule-of-three bound (3/n), not Wilson, scaled
    # by the finite-population correction (n=20, population=200 is not a
    # tiny sampling fraction, so the FPC meaningfully shrinks the bound).
    stratum = Stratum(name="zero", n=20, m=0, population=200)

    estimate = recall_with_floor(stratum, true_positives=100)
    expected_q_floor = (3 / 20) * fpc(20, 200)

    assert exclusion_error_rate(stratum) == 0.0
    assert exclusion_error_rate_floor(stratum) == pytest.approx(expected_q_floor)
    assert estimate.point == pytest.approx(1.0)
    assert estimate.floor == pytest.approx(100 / (100 + expected_q_floor * 200))
    assert estimate.floor < estimate.point


def test_recall_floor_uses_wilson_upper_bound_when_errors_observed() -> None:
    stratum = Stratum(name="nonzero", n=40, m=5, population=400)

    floor_rate = exclusion_error_rate_floor(stratum)
    point = exclusion_error_rate(stratum)
    _low, wilson_high = wilson_interval(5, 40)
    expected = point + (wilson_high - point) * fpc(40, 400)

    assert floor_rate == pytest.approx(expected)
    assert floor_rate > point
    assert floor_rate < wilson_high


def test_stratified_recall_reduces_to_unstratified_for_single_stratum() -> None:
    stratum = Stratum(name="only", n=20, m=0, population=200)

    via_stratified = stratified_recall([stratum], true_positives=100)
    via_convenience = recall_with_floor(stratum, true_positives=100)

    assert via_stratified == via_convenience
    # Hand check: point = TP / (TP + 0) = 1.0; floor = TP / (TP + q_floor*200).
    expected_q_floor = (3 / 20) * fpc(20, 200)
    assert via_stratified.point == pytest.approx(1.0)
    assert via_stratified.floor == pytest.approx(100 / (100 + expected_q_floor * 200))


def test_stratified_recall_combines_multiple_strata() -> None:
    strata = [
        Stratum(name="a", n=20, m=0, population=200),
        Stratum(name="b", n=30, m=3, population=500),
    ]

    estimate = stratified_recall(strata, true_positives=100)

    fn_point = 0.0 * 200 + (3 / 30) * 500
    assert estimate.estimated_fn == pytest.approx(fn_point)
    assert estimate.point == pytest.approx(100 / (100 + fn_point))
    assert estimate.floor < estimate.point
    assert estimate.ci[0] <= estimate.point <= estimate.ci[1]


def test_stratified_recall_rejects_empty_strata() -> None:
    with pytest.raises(RecallError):
        stratified_recall([], true_positives=10)


def test_stratum_rejects_m_greater_than_n() -> None:
    with pytest.raises(RecallError):
        Stratum(name="bad", n=5, m=6, population=100)


def test_stratum_rejects_population_smaller_than_sample() -> None:
    with pytest.raises(RecallError):
        Stratum(name="bad", n=50, m=1, population=10)


def test_recall_is_never_nan_when_true_positives_and_fn_are_zero() -> None:
    stratum = Stratum(name="all-good", n=10, m=0, population=10)

    estimate = recall_with_floor(stratum, true_positives=0)

    assert not math.isnan(estimate.point)
    assert estimate.point == 0.0


# --- exclusion_error_rate_floor: finite-population correction -----------------


def test_census_zero_cell_floor_collapses_to_exact_recall() -> None:
    # audit-draw --size all: n == population, no sampling uncertainty left.
    stratum = Stratum(name="census", n=500, m=0, population=500)

    estimate = recall_with_floor(stratum, true_positives=100)

    assert exclusion_error_rate_floor(stratum) == 0.0
    assert estimate.floor == pytest.approx(1.0)


def test_census_nonzero_cell_floor_equals_point() -> None:
    stratum = Stratum(name="census", n=500, m=17, population=500)

    assert exclusion_error_rate_floor(stratum) == pytest.approx(exclusion_error_rate(stratum))
    assert exclusion_error_rate_floor(stratum) == pytest.approx(17 / 500)


def test_floor_tightens_monotonically_as_sampling_fraction_grows() -> None:
    # Same m/n ratio (0.1), increasingly large share of the population sampled.
    tiny_fraction = Stratum(name="tiny", n=20, m=2, population=200_000)
    large_fraction = Stratum(name="large", n=20, m=2, population=22)

    assert exclusion_error_rate_floor(large_fraction) < exclusion_error_rate_floor(tiny_fraction)


def test_floor_matches_pre_fix_value_at_tiny_sampling_fraction() -> None:
    # n / population ~ 0.0002 -- negligible enough that the FPC should barely move things.
    zero_cell = Stratum(name="zero", n=200, m=0, population=1_000_000)
    nonzero_cell = Stratum(name="nonzero", n=200, m=8, population=1_000_000)

    assert exclusion_error_rate_floor(zero_cell) == pytest.approx(3 / 200, abs=1e-4)
    _low, wilson_high = wilson_interval(8, 200)
    assert exclusion_error_rate_floor(nonzero_cell) == pytest.approx(wilson_high, abs=1e-4)


def test_finite_population_correction_guards_degenerate_inputs() -> None:
    assert fpc(10, 1) == 0.0
    assert fpc(10, 0) == 0.0
    assert fpc(50, 50) == 0.0
    assert fpc(60, 50) == 0.0


def test_floor_never_drops_below_point_estimate_with_fpc() -> None:
    strata = [
        Stratum(name="census-zero", n=500, m=0, population=500),
        Stratum(name="census-nonzero", n=500, m=17, population=500),
        Stratum(name="tiny", n=20, m=2, population=200_000),
        Stratum(name="large", n=20, m=2, population=22),
        Stratum(name="mid", n=20, m=0, population=200),
    ]

    for stratum in strata:
        assert exclusion_error_rate_floor(stratum) >= exclusion_error_rate(stratum)


# --- confusion.py -------------------------------------------------------------


def test_confusion_matrix_counts_tp_fp_fn_tn() -> None:
    predictions = {"a": 1, "b": 1, "c": -1, "d": -1}
    truths = {"a": 1, "b": -1, "c": 1, "d": -1}

    matrix = confusion_matrix(predictions, truths)

    assert matrix == ConfusionMatrix(tp=1, fp=1, fn=1, tn=1)


def test_confusion_matrix_skips_ids_missing_from_either_side() -> None:
    predictions = {"a": 1, "only_in_predictions": 1}
    truths = {"a": 1, "only_in_truths": 1}

    matrix = confusion_matrix(predictions, truths)

    assert matrix == ConfusionMatrix(tp=1, fp=0, fn=0, tn=0)


def test_confusion_matrix_empty_inputs_give_all_zero_counts() -> None:
    assert confusion_matrix({}, {}) == ConfusionMatrix(tp=0, fp=0, fn=0, tn=0)


def test_confusion_matrix_relevant_label_is_configurable() -> None:
    predictions = {"a": 0, "b": 1, "c": 0}
    truths = {"a": 0, "b": 0, "c": 1}

    default = confusion_matrix(predictions, truths)
    assert default == ConfusionMatrix(tp=0, fp=1, fn=1, tn=1)

    zero_as_relevant = confusion_matrix(predictions, truths, relevant_label=0)
    assert zero_as_relevant == ConfusionMatrix(tp=1, fp=1, fn=1, tn=0)


def test_relevant_label_constant_is_include() -> None:
    assert RELEVANT_LABEL == 1
