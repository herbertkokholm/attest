"""Latent-vendor-drift sentinel: offline, per-vendor drift detection against a frozen baseline.

Implements the hybrid rule recommended by `docs/sentinel_drift_rule.md`:

- **Hard trigger, opens an epoch:** a vendor's ratings on the frozen
  sentinel set show `>= hard_trigger_crossings` one-directional polarity
  crossings (`baseline in {0, +1} -> current == -1`) since its baseline was
  captured. This is the failure mode that threatens recall -- a vendor
  silently starting to exclude records it used to keep -- and is the only
  rule the doc's probe shows to be both direction-aware and stable under
  the class imbalance a screening track typically has.
- **Advisory-only signal, logged, never epoch-opening:** ordinal
  Krippendorff's alpha between baseline and current, reused unchanged from
  `attest.stats.agreement` (the doc's imbalance analysis shows alpha alone
  is not stable enough at typical sentinel-set sizes to trigger on).

Pure and offline: every function here takes already-collected per-vendor
ratings (a plain `{record_id: rating}` mapping) and never itself schedules,
polls, or persists a baseline store -- see the doc's "Split with the
runbook" section for why that operational half belongs to the runbook, not
here. `collect_sentinel_votes` is the one bridge to a live/deterministic
`Rater`, reusable for both baseline capture and later re-evaluation, and it
still makes no network call itself: whatever `Rater.rate` does is the
caller's concern, exactly as `attest.vendors.base.run_ensemble` documents.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from attest.contracts.input import Record
from attest.provenance.changelog import CHANGE_TYPE_SENTINEL_DRIFT, ConfigChangeEvent
from attest.provenance.config import Config
from attest.provenance.epochs import Epoch, open_epoch
from attest.stats.agreement import AgreementError, krippendorff_alpha, raw_agreement
from attest.vendors.base import Rater

DEFAULT_HARD_TRIGGER_CROSSINGS = 2
DEFAULT_ADVISORY_ALPHA_THRESHOLD = 0.80

# The one-directional crossing this rule watches for: a vendor that used to
# keep a record (baseline uncertain or include) now silently excludes it.
# The reverse crossing does not count -- see the doc's "imbalance" section
# for why counting it collapses this rule toward the unusable absolute rule.
_CROSSING_FROM = (0, 1)
_CROSSING_TO = -1


class SentinelError(ValueError):
    """Raised when sentinel inputs violate the frozen-set or baseline invariants."""


def compute_sentinel_set_id(records: Sequence[Record]) -> str:
    """Content hash of a frozen sentinel set: its record ids, titles, abstracts, and tracks.

    Args:
        records: The sentinel set's records.

    Returns:
        A hex-encoded SHA-256 digest, stable regardless of `records`' order.

    Raises:
        SentinelError: If `records` is empty.
    """
    if not records:
        raise SentinelError("sentinel set must contain at least one record")
    canonical = json.dumps(
        [
            {"id": r.id, "title": r.title, "abstract": r.abstract, "track": r.track}
            for r in sorted(records, key=lambda r: r.id)
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_sentinel_votes(
    records: Sequence[Record], raters: Sequence[Rater]
) -> dict[str, dict[str, int]]:
    """Run every rater over every sentinel record and collect ordinal ratings.

    Reusable for both baseline capture (first epoch) and later re-evaluation
    -- the same function, so a baseline and a current reading are always
    produced identically.

    Args:
        records: The frozen sentinel set's records.
        raters: The vendors to evaluate, e.g. built the same way
            `attest.cli._build_raters` builds an ensemble's raters.

    Returns:
        `vendor -> {record_id: rating}`.
    """
    votes: dict[str, dict[str, int]] = {rater.vendor: {} for rater in raters}
    for record in records:
        for rater in raters:
            ordinal, _raw = rater.rate(record)
            votes[rater.vendor][record.id] = ordinal
    return votes


@dataclass(frozen=True)
class SentinelBaseline:
    """One vendor set's first-epoch ratings on the frozen sentinel set.

    Attributes:
        sentinel_set_id: Content hash of the frozen sentinel set these
            ratings were captured on (see `compute_sentinel_set_id`).
        ensemble_config_id: Ensemble configuration in force when the
            baseline was captured.
        epoch_id: Epoch in force when the baseline was captured.
        votes: `vendor -> {record_id: baseline rating}`.
        recorded_at: When the baseline was captured.
    """

    sentinel_set_id: str
    ensemble_config_id: str
    epoch_id: str
    votes: dict[str, dict[str, int]]
    recorded_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentinel_set_id": self.sentinel_set_id,
            "ensemble_config_id": self.ensemble_config_id,
            "epoch_id": self.epoch_id,
            "votes": {vendor: dict(ratings) for vendor, ratings in self.votes.items()},
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SentinelBaseline:
        return cls(
            sentinel_set_id=payload["sentinel_set_id"],
            ensemble_config_id=payload["ensemble_config_id"],
            epoch_id=payload["epoch_id"],
            votes={vendor: dict(ratings) for vendor, ratings in payload["votes"].items()},
            recorded_at=datetime.fromisoformat(payload["recorded_at"]),
        )


def capture_baseline(
    sentinel_set_id: str,
    ensemble_config_id: str,
    epoch_id: str,
    votes_by_vendor: Mapping[str, Mapping[str, int]],
    *,
    recorded_at: datetime | None = None,
) -> SentinelBaseline:
    """Build a `SentinelBaseline` from a freshly collected set of votes.

    See `collect_sentinel_votes` for how `votes_by_vendor` is typically produced.
    """
    return SentinelBaseline(
        sentinel_set_id=sentinel_set_id,
        ensemble_config_id=ensemble_config_id,
        epoch_id=epoch_id,
        votes={vendor: dict(ratings) for vendor, ratings in votes_by_vendor.items()},
        recorded_at=recorded_at or datetime.now(UTC),
    )


@dataclass(frozen=True)
class VendorSentinelResult:
    """One vendor's baseline-vs-current sentinel comparison.

    Attributes:
        vendor: Vendor name.
        n_records: Number of sentinel records shared between baseline and
            current ratings (the comparison basis).
        raw_agreement: Proportion of records where baseline == current, per
            `attest.stats.agreement.raw_agreement`'s convention of never
            reporting alpha without it.
        alpha: Ordinal Krippendorff's alpha between the baseline and current
            rows, or None if undefined (e.g. every rating identical).
        polarity_crossings: Count of records where `baseline in {0, +1}` and
            `current == -1` -- the recall-critical, one-directional crossing.
        crossed_record_ids: Which record ids crossed, for provenance/debug.
        hard_trigger: Whether `polarity_crossings` meets or exceeds the
            configured threshold.
        advisory_flag: Whether `alpha` is defined and below the advisory
            threshold. Never epoch-opening on its own.
    """

    vendor: str
    n_records: int
    raw_agreement: float | None
    alpha: float | None
    polarity_crossings: int
    crossed_record_ids: tuple[str, ...]
    hard_trigger: bool
    advisory_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "n_records": self.n_records,
            "raw_agreement": self.raw_agreement,
            "alpha": self.alpha,
            "polarity_crossings": self.polarity_crossings,
            "crossed_record_ids": list(self.crossed_record_ids),
            "hard_trigger": self.hard_trigger,
            "advisory_flag": self.advisory_flag,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VendorSentinelResult:
        return cls(
            vendor=payload["vendor"],
            n_records=payload["n_records"],
            raw_agreement=payload.get("raw_agreement"),
            alpha=payload.get("alpha"),
            polarity_crossings=payload["polarity_crossings"],
            crossed_record_ids=tuple(payload.get("crossed_record_ids", ())),
            hard_trigger=payload["hard_trigger"],
            advisory_flag=payload["advisory_flag"],
        )


def evaluate_vendor(
    vendor: str,
    baseline_ratings: Mapping[str, int],
    current_ratings: Mapping[str, int],
    *,
    hard_trigger_crossings: int = DEFAULT_HARD_TRIGGER_CROSSINGS,
    advisory_alpha_threshold: float = DEFAULT_ADVISORY_ALPHA_THRESHOLD,
) -> VendorSentinelResult:
    """Compare one vendor's baseline and current ratings on the shared sentinel records.

    Args:
        vendor: Vendor name, for the result and error messages.
        baseline_ratings: `{record_id: baseline rating}`.
        current_ratings: `{record_id: current rating}`.
        hard_trigger_crossings: Minimum one-directional polarity crossings
            to hard-trigger.
        advisory_alpha_threshold: Alpha bound below which the advisory flag
            is set.

    Returns:
        This vendor's `VendorSentinelResult`.

    Raises:
        SentinelError: If `baseline_ratings` and `current_ratings` share no
            record id.
    """
    shared_ids = sorted(set(baseline_ratings) & set(current_ratings))
    if not shared_ids:
        raise SentinelError(
            f"vendor '{vendor}': no sentinel records shared between baseline and current ratings"
        )

    baseline_row = [float(baseline_ratings[record_id]) for record_id in shared_ids]
    current_row = [float(current_ratings[record_id]) for record_id in shared_ids]

    agreement = raw_agreement([baseline_row, current_row])
    try:
        alpha: float | None = krippendorff_alpha([baseline_row, current_row])
    except AgreementError:
        alpha = None

    crossed = tuple(
        record_id
        for record_id in shared_ids
        if baseline_ratings[record_id] in _CROSSING_FROM
        and current_ratings[record_id] == _CROSSING_TO
    )
    crossings = len(crossed)

    return VendorSentinelResult(
        vendor=vendor,
        n_records=len(shared_ids),
        raw_agreement=agreement,
        alpha=alpha,
        polarity_crossings=crossings,
        crossed_record_ids=crossed,
        hard_trigger=crossings >= hard_trigger_crossings,
        advisory_flag=alpha is not None and alpha < advisory_alpha_threshold,
    )


@dataclass(frozen=True)
class SentinelEvaluation:
    """A full sentinel run's results across every evaluated vendor.

    Attributes:
        sentinel_set_id: The frozen sentinel set this evaluation ran against.
        ensemble_config_id: The screening configuration in force.
        evaluated_at: When this evaluation ran.
        hard_trigger_crossings: The threshold that was in force.
        advisory_alpha_threshold: The advisory bound that was in force.
        per_vendor: Mapping of vendor name to its `VendorSentinelResult`.
    """

    sentinel_set_id: str
    ensemble_config_id: str
    evaluated_at: datetime
    hard_trigger_crossings: int
    advisory_alpha_threshold: float
    per_vendor: dict[str, VendorSentinelResult]

    @property
    def hard_trigger_vendors(self) -> tuple[str, ...]:
        """Vendors whose comparison hard-triggered, sorted."""
        return tuple(
            sorted(vendor for vendor, result in self.per_vendor.items() if result.hard_trigger)
        )

    @property
    def advisory_vendors(self) -> tuple[str, ...]:
        """Vendors flagged by the advisory-only alpha signal, sorted."""
        return tuple(
            sorted(vendor for vendor, result in self.per_vendor.items() if result.advisory_flag)
        )

    @property
    def triggered(self) -> bool:
        """Whether any vendor hard-triggered."""
        return bool(self.hard_trigger_vendors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentinel_set_id": self.sentinel_set_id,
            "ensemble_config_id": self.ensemble_config_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "hard_trigger_crossings": self.hard_trigger_crossings,
            "advisory_alpha_threshold": self.advisory_alpha_threshold,
            "per_vendor": {vendor: result.to_dict() for vendor, result in self.per_vendor.items()},
            "hard_trigger_vendors": list(self.hard_trigger_vendors),
            "advisory_vendors": list(self.advisory_vendors),
            "triggered": self.triggered,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SentinelEvaluation:
        return cls(
            sentinel_set_id=payload["sentinel_set_id"],
            ensemble_config_id=payload["ensemble_config_id"],
            evaluated_at=datetime.fromisoformat(payload["evaluated_at"]),
            hard_trigger_crossings=payload["hard_trigger_crossings"],
            advisory_alpha_threshold=payload["advisory_alpha_threshold"],
            per_vendor={
                vendor: VendorSentinelResult.from_dict(result)
                for vendor, result in payload["per_vendor"].items()
            },
        )


def evaluate_sentinel(
    baseline: SentinelBaseline,
    current_votes_by_vendor: Mapping[str, Mapping[str, int]],
    *,
    hard_trigger_crossings: int = DEFAULT_HARD_TRIGGER_CROSSINGS,
    advisory_alpha_threshold: float = DEFAULT_ADVISORY_ALPHA_THRESHOLD,
    evaluated_at: datetime | None = None,
) -> SentinelEvaluation:
    """Evaluate every vendor present in both the baseline and `current_votes_by_vendor`.

    Args:
        baseline: The frozen baseline to compare against.
        current_votes_by_vendor: This check's freshly collected votes (see
            `collect_sentinel_votes`), `vendor -> {record_id: rating}`.
        hard_trigger_crossings: Minimum one-directional polarity crossings
            to hard-trigger, evaluated independently per vendor.
        advisory_alpha_threshold: Alpha bound below which the advisory flag
            is set.
        evaluated_at: Timestamp to record; defaults to now (UTC).

    Returns:
        A `SentinelEvaluation` covering every vendor common to both inputs.

    Raises:
        SentinelError: If no vendor is common to both `baseline.votes` and
            `current_votes_by_vendor`.
    """
    per_vendor = {
        vendor: evaluate_vendor(
            vendor,
            baseline.votes[vendor],
            ratings,
            hard_trigger_crossings=hard_trigger_crossings,
            advisory_alpha_threshold=advisory_alpha_threshold,
        )
        for vendor, ratings in current_votes_by_vendor.items()
        if vendor in baseline.votes
    }
    if not per_vendor:
        raise SentinelError("no vendor in current_votes_by_vendor matches the baseline's vendors")
    return SentinelEvaluation(
        sentinel_set_id=baseline.sentinel_set_id,
        ensemble_config_id=baseline.ensemble_config_id,
        evaluated_at=evaluated_at or datetime.now(UTC),
        hard_trigger_crossings=hard_trigger_crossings,
        advisory_alpha_threshold=advisory_alpha_threshold,
        per_vendor=per_vendor,
    )


def open_epoch_for_hard_trigger(
    config: Config, evaluation: SentinelEvaluation, *, opened_at: datetime | None = None
) -> tuple[Epoch, ConfigChangeEvent]:
    """Open a new epoch and build its changelog event for a hard-triggered sentinel evaluation.

    Drift is a vendor *behavior* change, not necessarily a `Config` field
    change (see the module docstring and `docs/sentinel_drift_rule.md`'s
    "Wiring" section), so this always opens a genuinely new `Epoch` (a fresh
    id) rather than going through `attest.provenance.epochs.maybe_open_epoch`,
    which would see an unchanged `ensemble_config_id` and reuse the current
    epoch. `before == after` on the returned event, since the config's
    content hash is indeed unchanged -- only the epoch is new.

    The caller (typically the CLI) is responsible for persisting the
    returned epoch/event and for screening any further records for this
    configuration under a *new* run directory, exactly as an explicit config
    change already requires (`attest.io.store.RunStore` locks one run
    directory to one epoch).

    Args:
        config: The ensemble configuration in force (unchanged by drift).
        evaluation: A `SentinelEvaluation` with `triggered` True.
        opened_at: Timestamp for the new epoch; defaults to now (UTC).

    Returns:
        `(new_epoch, change_event)`.

    Raises:
        SentinelError: If `evaluation.triggered` is False.
    """
    if not evaluation.triggered:
        raise SentinelError("evaluation did not hard-trigger; no new epoch to open")
    new_epoch = open_epoch(config, opened_at=opened_at)
    ensemble_config_id = new_epoch.ensemble_config_id
    vendors = ", ".join(evaluation.hard_trigger_vendors)
    reason = (
        f"sentinel drift: vendor(s) {vendors}, >= {evaluation.hard_trigger_crossings} polarity "
        f"crossings on sentinel set '{evaluation.sentinel_set_id}'"
    )
    event = ConfigChangeEvent(
        timestamp=new_epoch.opened_at,
        before=ensemble_config_id,
        after=ensemble_config_id,
        reason=reason,
        change_type=CHANGE_TYPE_SENTINEL_DRIFT,
        changed_fields=(),
    )
    return new_epoch, event


__all__ = [
    "DEFAULT_ADVISORY_ALPHA_THRESHOLD",
    "DEFAULT_HARD_TRIGGER_CROSSINGS",
    "SentinelBaseline",
    "SentinelError",
    "SentinelEvaluation",
    "VendorSentinelResult",
    "capture_baseline",
    "collect_sentinel_votes",
    "compute_sentinel_set_id",
    "evaluate_sentinel",
    "evaluate_vendor",
    "open_epoch_for_hard_trigger",
]
