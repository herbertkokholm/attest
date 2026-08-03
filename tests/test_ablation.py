"""Tests for attest.ablation.xsweep: controlled ablation of ensemble size x."""

from __future__ import annotations

import math
import random
import warnings

import pytest

from attest.ablation.xsweep import AblationError, sweep
from attest.contracts.input import Record
from attest.ensemble.aggregate import ZERO_POLICY_ESCALATE, ZERO_POLICY_INCLUDE
from attest.ensemble.tau import describe_tau
from attest.ensemble.votes import build_vote_vector
from attest.provenance.config import Config, VendorSpec
from attest.vendors.base import DeterministicRater, run_ensemble

_SPEC = VendorSpec(model="deterministic-v1", model_version="v1", prompt_version="p1")


def _record(record_id: str) -> Record:
    return Record(id=record_id, title="t", abstract="a", track=0)


def _run(raters: list[DeterministicRater], record_ids: list[str]) -> list:
    records = [_record(rid) for rid in record_ids]
    config = Config(vendors={r.vendor: _SPEC for r in raters}, aggregation="boundary_dispersion")
    return run_ensemble(records, raters, config).votes


# --- subset enumeration -------------------------------------------------------


def test_sweep_enumerates_expected_subsets_for_small_pool() -> None:
    record_ids = ["r1", "r2", "r3", "r4", "r5"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b", "c"])]
    votes = _run(raters, record_ids)
    truths = {rid: (1 if i % 2 == 0 else -1) for i, rid in enumerate(record_ids)}

    report = sweep(votes, truths)

    expected_keys = {
        (2, ("a", "b")),
        (2, ("a", "c")),
        (2, ("b", "c")),
        (3, ("a", "b", "c")),
    }
    assert set(report.results) == expected_keys
    assert report.candidate_vendors == ("a", "b", "c")


def test_sweep_rejects_fewer_than_two_candidate_vendors() -> None:
    record_ids = ["r1", "r2"]
    raters = [DeterministicRater(vendor="only", seed=0)]
    votes = _run(raters, record_ids)
    truths = {rid: 1 for rid in record_ids}

    with pytest.raises(AblationError):
        sweep(votes, truths)


def test_sweep_rejects_missing_gold_labels() -> None:
    record_ids = ["r1", "r2"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b"])]
    votes = _run(raters, record_ids)

    with pytest.raises(AblationError):
        sweep(votes, {"r1": 1})


def test_sweep_falls_back_to_stratified_sample_when_infeasible() -> None:
    record_ids = [f"r{i}" for i in range(20)]
    vendor_names = ["a", "b", "c", "d", "e", "f"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(vendor_names)]
    votes = _run(raters, record_ids)
    truths = {rid: (1 if i % 3 == 0 else -1) for i, rid in enumerate(record_ids)}

    # C(6, 3) = 20 > max_subsets_per_x=5, so x=3 must fall back to sampling.
    report = sweep(votes, truths, max_subsets_per_x=5, rng=random.Random(42))

    x3_subsets = [key for key in report.results if key[0] == 3]
    assert len(x3_subsets) == 5
    # Sampling is deterministic given a seeded rng.
    report_again = sweep(votes, truths, max_subsets_per_x=5, rng=random.Random(42))
    assert set(report.results) == set(report_again.results)


# --- finite metrics ------------------------------------------------------------


def test_sweep_returns_finite_metrics_for_every_subset() -> None:
    record_ids = [f"r{i}" for i in range(8)]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b", "c", "d"])]
    votes = _run(raters, record_ids)
    truths = {rid: (1 if i % 2 == 0 else -1) for i, rid in enumerate(record_ids)}

    report = sweep(votes, truths)

    assert report.results
    for (x, subset), result in report.results.items():
        assert result.x == x
        assert result.vendors == subset
        assert result.n_records == len(record_ids)
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.escalation_rate <= 1.0
        assert not math.isnan(result.recall)
        assert not math.isnan(result.precision)
        assert not math.isnan(result.escalation_rate)
        if result.alpha is not None:
            assert math.isfinite(result.alpha)
        if result.raw_agreement is not None:
            assert 0.0 <= result.raw_agreement <= 1.0
        assert set(result.leave_one_out) == set(subset)
        for contribution in result.leave_one_out.values():
            assert not math.isnan(contribution.delta_recall)
            assert not math.isnan(contribution.delta_precision)
            assert not math.isnan(contribution.delta_escalation_rate)
            if contribution.delta_alpha is not None:
                assert math.isfinite(contribution.delta_alpha)


# --- tau_report attachment and the fixed-tau-across-x' caveat ------------------


def test_sweep_attaches_the_per_x_tau_report_to_each_subset_result() -> None:
    record_ids = ["r1", "r2", "r3", "r4", "r5"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b", "c"])]
    votes = _run(raters, record_ids)
    truths = {rid: (1 if i % 2 == 0 else -1) for i, rid in enumerate(record_ids)}

    report = sweep(votes, truths, tau=0.3)

    for (x, _subset), result in report.results.items():
        assert result.tau_report == describe_tau(0.3, x)
        assert result.tau_report.x == x


def test_sweep_warns_once_when_it_spans_more_than_one_x(recwarn: pytest.WarningsRecorder) -> None:
    record_ids = ["r1", "r2", "r3", "r4", "r5"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b", "c"])]
    votes = _run(raters, record_ids)
    truths = {rid: (1 if i % 2 == 0 else -1) for i, rid in enumerate(record_ids)}

    sweep(votes, truths)

    tau_warnings = [w for w in recwarn.list if "not comparable across x" in str(w.message)]
    assert len(tau_warnings) == 1


def test_sweep_does_not_warn_when_only_one_x_is_swept() -> None:
    # A 2-vendor pool only ever sweeps x=2, so there is no cross-x' comparison
    # to warn about.
    record_ids = ["r1", "r2", "r3"]
    raters = [DeterministicRater(vendor=v, seed=i) for i, v in enumerate(["a", "b"])]
    votes = _run(raters, record_ids)
    truths = {rid: 1 for rid in record_ids}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep(votes, truths)


# --- zero_policy is a swept-invariant, not part of the sweep -------------------


def test_sweep_zero_policy_changes_a_tied_records_escalation_but_not_recall_here() -> None:
    # Exact tie: mean 0, not a boundary vector.
    votes = [build_vote_vector("r1", "cfg", {"a": 0, "b": 0})]
    truths = {"r1": 1}

    escalate_report = sweep(votes, truths, zero_policy=ZERO_POLICY_ESCALATE)
    include_report = sweep(votes, truths, zero_policy=ZERO_POLICY_INCLUDE)

    escalate_result = escalate_report.results[(2, ("a", "b"))]
    include_result = include_report.results[(2, ("a", "b"))]

    # Escalate: routed to adjudication, resolves to gold under the sweep's
    # perfect-adjudication assumption -- see the module docstring.
    assert escalate_result.escalation_rate == pytest.approx(1.0)
    # Include: auto-labeled +1 directly, no escalation at all.
    assert include_result.escalation_rate == pytest.approx(0.0)
    # Both happen to score this single record correctly here (truth=1).
    assert escalate_result.recall == pytest.approx(1.0)
    assert include_result.recall == pytest.approx(1.0)


def test_sweep_zero_policy_defaults_to_escalate() -> None:
    votes = [build_vote_vector("r1", "cfg", {"a": 0, "b": 0})]
    truths = {"r1": 1}

    default_report = sweep(votes, truths)
    explicit_report = sweep(votes, truths, zero_policy=ZERO_POLICY_ESCALATE)

    assert default_report.results == explicit_report.results


# --- leave-one-out --------------------------------------------------------------


def test_leave_one_out_identifies_the_removed_vendor() -> None:
    # Seeds chosen so that "good" always matches gold, and "bad1"/"bad2"
    # both always predict exclude (-1) regardless of the true label.
    record_ids = ["r1", "r2", "r3", "r4"]
    truths = {"r1": 1, "r2": 1, "r3": -1, "r4": -1}
    raters = [
        DeterministicRater(vendor="good", seed=150),
        DeterministicRater(vendor="bad1", seed=148),
        DeterministicRater(vendor="bad2", seed=15),
    ]
    votes = _run(raters, record_ids)

    report = sweep(votes, truths)

    full = report.results[(3, ("bad1", "bad2", "good"))]
    # Removing "good" costs the most recall: without it, both remaining
    # voters unanimously (and wrongly) predict exclude on the two truly
    # relevant records.
    assert full.leave_one_out["good"].delta_recall == pytest.approx(1.0)
    assert full.leave_one_out["bad1"].delta_recall == pytest.approx(0.0)
    assert full.leave_one_out["bad2"].delta_recall == pytest.approx(0.0)

    most_impactful = max(full.leave_one_out.values(), key=lambda c: c.delta_recall)
    assert most_impactful.removed_vendor == "good"
