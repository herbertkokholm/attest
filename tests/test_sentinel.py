"""Tests for attest.provenance.sentinel: offline latent-vendor-drift detection.

Uses a small scripted, deterministic `Rater` double (not
`attest.vendors.base.DeterministicRater`, whose hash-derived ratings are not
practical to hand-pick for an exact polarity-crossing count) to exercise the
full baseline-capture -> re-evaluation pipeline end to end, entirely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from attest.contracts.input import Record
from attest.provenance.changelog import CHANGE_TYPE_SENTINEL_DRIFT
from attest.provenance.config import Config, VendorSpec
from attest.provenance.sentinel import (
    SentinelError,
    SentinelEvaluation,
    capture_baseline,
    check_sentinel_staleness,
    collect_sentinel_votes,
    compute_sentinel_set_id,
    evaluate_sentinel,
    evaluate_vendor,
    open_epoch_for_hard_trigger,
)

# A heavily exclude-dominated sentinel set, echoing the composition
# docs/sentinel_drift_rule.md's probe uses (mostly -1, a little 0 and +1),
# scaled down to ten records for a fast, hand-checkable test fixture.
_BASELINE_RATINGS = [-1, -1, -1, -1, -1, -1, -1, 0, 1, 1]


@dataclass
class _ScriptedRater:
    """A minimal, fully deterministic `Rater` double: fixed ratings by record id."""

    vendor: str
    ratings: dict[str, int]
    model: str = "scripted-v1"

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, Any]:
        return self.ratings[record.id], {"scripted": True}


def _sentinel_records(n: int = 10) -> list[Record]:
    return [
        Record(id=f"s{i}", title=f"title {i}", abstract=f"abstract {i}", track="sentinel")
        for i in range(n)
    ]


def _config() -> Config:
    return Config(
        vendors={
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0)
        },
        aggregation="boundary_dispersion",
        tau=0.5,
    )


def _capture(records: list[Record], ratings: list[int]) -> tuple[str, dict[str, dict[str, int]]]:
    sentinel_set_id = compute_sentinel_set_id(records)
    rater = _ScriptedRater(
        vendor="v1", ratings={r.id: v for r, v in zip(records, ratings, strict=True)}
    )
    votes = collect_sentinel_votes(records, [rater])
    return sentinel_set_id, votes


# --- compute_sentinel_set_id / collect_sentinel_votes ---------------------------


def test_compute_sentinel_set_id_stable_regardless_of_order() -> None:
    records = _sentinel_records()
    reversed_records = list(reversed(records))

    assert compute_sentinel_set_id(records) == compute_sentinel_set_id(reversed_records)


def test_compute_sentinel_set_id_changes_with_content() -> None:
    records = _sentinel_records()
    changed = list(records)
    changed[0] = Record(id="s0", title="different title", abstract="abstract 0", track="sentinel")

    assert compute_sentinel_set_id(records) != compute_sentinel_set_id(changed)


def test_compute_sentinel_set_id_rejects_empty_set() -> None:
    with pytest.raises(SentinelError):
        compute_sentinel_set_id([])


def test_collect_sentinel_votes_reads_one_rating_per_vendor_per_record() -> None:
    records = _sentinel_records(3)
    rater = _ScriptedRater(vendor="v1", ratings={"s0": -1, "s1": 0, "s2": 1})

    votes = collect_sentinel_votes(records, [rater])

    assert votes == {"v1": {"s0": -1, "s1": 0, "s2": 1}}


# --- capture_baseline / evaluate_vendor / evaluate_sentinel: the four required scenarios --


def test_no_drift_does_not_hard_trigger_or_flag_advisory() -> None:
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    _current_id, current_votes = _capture(records, _BASELINE_RATINGS)  # identical ratings
    evaluation = evaluate_sentinel(baseline, current_votes)

    result = evaluation.per_vendor["v1"]
    assert result.polarity_crossings == 0
    assert result.hard_trigger is False
    assert result.advisory_flag is False
    assert result.alpha == pytest.approx(1.0)
    assert evaluation.triggered is False


def test_one_benign_flip_does_not_hard_trigger() -> None:
    # index 7: baseline uncertain (0) swings to include (+1) -- a benign
    # swing, not the recall-critical 0/+1 -> -1 crossing this rule watches for.
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    current_ratings = list(_BASELINE_RATINGS)
    current_ratings[7] = 1
    _current_id, current_votes = _capture(records, current_ratings)

    evaluation = evaluate_sentinel(baseline, current_votes)

    result = evaluation.per_vendor["v1"]
    assert result.polarity_crossings == 0
    assert result.hard_trigger is False
    assert evaluation.triggered is False


def test_two_recall_critical_crossings_hard_triggers() -> None:
    # index 7 (baseline 0) and index 8 (baseline +1) both flip to -1: two
    # one-directional polarity crossings, meeting the default threshold.
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    current_ratings = list(_BASELINE_RATINGS)
    current_ratings[7] = -1
    current_ratings[8] = -1
    _current_id, current_votes = _capture(records, current_ratings)

    evaluation = evaluate_sentinel(baseline, current_votes)

    result = evaluation.per_vendor["v1"]
    assert result.polarity_crossings == 2
    assert set(result.crossed_record_ids) == {"s7", "s8"}
    assert result.hard_trigger is True
    assert evaluation.triggered is True
    assert evaluation.hard_trigger_vendors == ("v1",)


def test_advisory_only_drift_flags_without_hard_triggering() -> None:
    # Four non-crossing changes (two +1 -> 0 swings, one 0 -> +1 swing, one
    # -1 -> 0 swing): alpha drops to ~0.78 (below the 0.80 advisory bound)
    # while zero records land on the recall-critical -1 crossing.
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    current_ratings = list(_BASELINE_RATINGS)
    current_ratings[6] = 0  # -1 -> 0
    current_ratings[7] = 1  # 0 -> +1
    current_ratings[8] = 0  # +1 -> 0
    current_ratings[9] = 0  # +1 -> 0
    _current_id, current_votes = _capture(records, current_ratings)

    evaluation = evaluate_sentinel(baseline, current_votes)

    result = evaluation.per_vendor["v1"]
    assert result.polarity_crossings == 0
    assert result.hard_trigger is False
    assert result.alpha is not None
    assert result.alpha < 0.80
    assert result.advisory_flag is True
    assert evaluation.triggered is False
    assert evaluation.advisory_vendors == ("v1",)


# --- evaluate_vendor: direct, dict-level API -------------------------------------


def test_evaluate_vendor_rejects_no_shared_records() -> None:
    with pytest.raises(SentinelError):
        evaluate_vendor("v1", {"a": 1}, {"b": -1})


def test_evaluate_vendor_hard_trigger_threshold_is_configurable() -> None:
    baseline_ratings = {"a": 1, "b": 1, "c": 1}
    current_ratings = {"a": -1, "b": -1, "c": 1}  # 2 crossings

    strict = evaluate_vendor("v1", baseline_ratings, current_ratings, hard_trigger_crossings=3)
    lenient = evaluate_vendor("v1", baseline_ratings, current_ratings, hard_trigger_crossings=2)

    assert strict.hard_trigger is False
    assert lenient.hard_trigger is True


def test_evaluate_sentinel_rejects_no_matching_vendor() -> None:
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    with pytest.raises(SentinelError):
        evaluate_sentinel(baseline, {"different-vendor": {"s0": -1}})


# --- round-tripping (for RunStore persistence) -----------------------------------


def test_sentinel_baseline_round_trips_through_dict() -> None:
    records = _sentinel_records()
    sentinel_set_id, votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(
        sentinel_set_id, "cfg-1", "epoch-1", votes, recorded_at=datetime(2026, 1, 1, tzinfo=UTC)
    )

    restored = type(baseline).from_dict(baseline.to_dict())

    assert restored == baseline


def test_sentinel_evaluation_round_trips_through_dict() -> None:
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)
    _current_id, current_votes = _capture(records, _BASELINE_RATINGS)
    evaluation = evaluate_sentinel(
        baseline, current_votes, evaluated_at=datetime(2026, 1, 2, tzinfo=UTC)
    )

    restored = type(evaluation).from_dict(evaluation.to_dict())

    assert restored == evaluation


# --- open_epoch_for_hard_trigger: epoch + changelog event wiring ----------------


def _evaluation_at(when: datetime) -> SentinelEvaluation:
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)
    _current_id, current_votes = _capture(records, _BASELINE_RATINGS)
    return evaluate_sentinel(baseline, current_votes, evaluated_at=when)


# --- check_sentinel_staleness ---------------------------------------------------


def test_staleness_not_stale_within_cadence() -> None:
    evaluation = _evaluation_at(datetime(2026, 1, 1, tzinfo=UTC))

    result = check_sentinel_staleness(
        [evaluation], max_staleness_days=7, as_of=datetime(2026, 1, 3, tzinfo=UTC)
    )

    assert result.stale is False
    assert result.last_checked_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert result.staleness_days == pytest.approx(2.0)


def test_staleness_stale_beyond_cadence() -> None:
    evaluation = _evaluation_at(datetime(2026, 1, 1, tzinfo=UTC))

    result = check_sentinel_staleness(
        [evaluation], max_staleness_days=7, as_of=datetime(2026, 1, 10, tzinfo=UTC)
    )

    assert result.stale is True
    assert result.staleness_days == pytest.approx(9.0)


def test_staleness_stale_when_never_checked() -> None:
    result = check_sentinel_staleness(
        [], max_staleness_days=7, as_of=datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert result.stale is True
    assert result.last_checked_at is None
    assert result.staleness_days is None


def test_staleness_uses_the_most_recent_of_several_evaluations() -> None:
    older = _evaluation_at(datetime(2025, 1, 1, tzinfo=UTC))
    newer = _evaluation_at(datetime(2026, 1, 1, tzinfo=UTC))

    result = check_sentinel_staleness(
        [older, newer], max_staleness_days=7, as_of=datetime(2026, 1, 2, tzinfo=UTC)
    )

    assert result.last_checked_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert result.stale is False


def test_open_epoch_for_hard_trigger_requires_a_triggered_evaluation() -> None:
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)
    _current_id, current_votes = _capture(records, _BASELINE_RATINGS)  # no drift
    evaluation = evaluate_sentinel(baseline, current_votes)

    with pytest.raises(SentinelError):
        open_epoch_for_hard_trigger(_config(), evaluation)


def test_open_epoch_for_hard_trigger_opens_new_epoch_and_logs_sentinel_drift_event() -> None:
    config = _config()
    records = _sentinel_records()
    sentinel_set_id, baseline_votes = _capture(records, _BASELINE_RATINGS)
    baseline = capture_baseline(sentinel_set_id, "cfg-1", "epoch-1", baseline_votes)

    current_ratings = list(_BASELINE_RATINGS)
    current_ratings[7] = -1
    current_ratings[8] = -1
    _current_id, current_votes = _capture(records, current_ratings)
    evaluation = evaluate_sentinel(baseline, current_votes)

    new_epoch, event = open_epoch_for_hard_trigger(
        config, evaluation, opened_at=datetime(2026, 2, 1, tzinfo=UTC)
    )

    # The config's content hash is unaffected by drift -- before == after --
    # but the epoch id itself is genuinely new.
    assert event.before == event.after == new_epoch.ensemble_config_id
    assert event.change_type == CHANGE_TYPE_SENTINEL_DRIFT
    assert "v1" in event.reason
    assert sentinel_set_id in event.reason
    assert new_epoch.opened_at == datetime(2026, 2, 1, tzinfo=UTC)
