"""Validation-record contract v1.5: one self-validation record per stable ensemble epoch.

This is a stable, versioned interface. Changing the shape produced here
requires a version bump to ``SCHEMA_VERSION``, not an in-place edit. v1.1
added ``config.zero_policy`` (additive: existing v1.0 consumers ignoring
unknown fields see no other change). v1.2 added ``unresolved_escalations``
(additive): the count of escalated decisions omitted from ``confusion``,
``recall``, and ``prisma`` because they had no resolved human label when
this record was assembled -- always 0 unless the caller explicitly opted
into `assemble_validation_record`'s `allow_unresolved_escalations`. v1.3
changes (not additive) ``error_correlation.pairwise_fn_on_relevant``'s
per-pair value from a bare float to an object (``correlation``, ``n``,
``both``, ``only_a``, ``only_b``, ``neither``): a consumer reading last
version's bare number needs updating. This also stops omitting a pair
whose correlation is undefined -- every vendor pair present in the run is
now reported, with `n`/joint counts standing in when `correlation` is
`None`, rather than that pair silently disappearing. v1.4 added
``recall.exact_floor`` (additive): an exact, hypergeometric one-sided
recall floor (see `attest.stats.recall.hypergeometric_upper_bound`),
reported alongside `recall.floor`'s asymptotic rule-of-three/Wilson
approximation as a genuine design-based alternative, `None` until audited.
v1.5 added ``config.batch_size`` (additive): the request batch size `b_e`
this epoch's configuration declares (see manuscript Eq. 1 and
`attest.provenance.config.Config.batch_size`), hashed unconditionally into
``ensemble_config_id`` and now reported alongside the rest of the
configuration for the same reason `zero_policy` is. v1.6 added
``recall.tp_estimation_method`` and ``recall.estimated_true_positives``
(additive): which of the two ways TP was obtained -- ``"full_review"``
(the include-and-escalate set was reviewed in full; unchanged default
behavior) or ``"inclusion_audit"`` (TP was scaled up from a sampled
inclusion audit, see `attest.stats.recall.stratified_recall_with_audited_tp`
and `attest.planes.inclusion_audit`) -- and, in the latter case, the
audit-scaled point estimate that was used. This makes the manuscript's own
reporting requirement (Section 2.9: a recall floor must be reported
together with the audit budget that produced it) machine-checkable rather
than implicit in which CLI flags a run happened to use.

A validation record captures, for a given immutable ensemble configuration
(``ensemble_config_id``) and epoch: the configuration itself, inter-rater
agreement, error correlation, escalation rate, recall (with its rule-of-three
worst-case floor), the confusion matrix against gold labels, and the PRISMA
flow counts. Statistics, audit, and escalation fields are populated later by
the kernel; `build()` only seeds the fields known at prefilter/config time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from attest.ensemble.aggregate import ZERO_POLICY_ESCALATE
from attest.prefilter.framework import Prisma as PrefilterPrisma

SCHEMA_VERSION = "1.6"


@dataclass
class Config:
    """The immutable ensemble configuration under evaluation.

    Attributes:
        vendors: LLM vendor names participating in the ensemble.
        models: Mapping of vendor name to model identifier.
        prompts: Mapping of prompt role (e.g. "screening") to prompt text or id.
        aggregation: Name of the aggregation strategy across ensemble votes.
        tau: Decision threshold used by the aggregation strategy.
        batch_size: The request batch size `b_e` this epoch's configuration
            declares (see `attest.provenance.config.Config.batch_size`).
        x: Ensemble size (number of voting members) at this epoch.
        zero_policy: Disposition of a would-be `auto_label == 0` decision
            (see `attest.ensemble.aggregate.g`) that produced this record's
            recall and precision -- the audit trail for a chosen, not
            defaulted-into, uncertainty policy.
    """

    vendors: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0
    batch_size: int = 1
    x: int = 0
    zero_policy: str = ZERO_POLICY_ESCALATE

    def to_dict(self) -> dict[str, Any]:
        """Return this configuration as a plain dict."""
        return {
            "vendors": list(self.vendors),
            "models": dict(self.models),
            "prompts": dict(self.prompts),
            "aggregation": self.aggregation,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "x": self.x,
            "zero_policy": self.zero_policy,
        }


@dataclass
class Agreement:
    """Inter-rater agreement among ensemble members.

    Attributes:
        krippendorff_alpha: Overall agreement coefficient, or None if not
            yet computed for this epoch.
        pairwise: Mapping of "vendorA|vendorB" to a pairwise agreement score.
    """

    krippendorff_alpha: float | None = None
    pairwise: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this agreement summary as a plain dict."""
        return {"krippendorff_alpha": self.krippendorff_alpha, "pairwise": dict(self.pairwise)}


@dataclass(frozen=True)
class PairwiseFnCorrelation:
    """One vendor pair's conditional false-negative correlation, with its joint counts.

    Attributes:
        correlation: Pearson correlation coefficient of the two vendors'
            false-negative indicators, or `None` if undefined (fewer than
            two relevant records, or one vendor made either no false
            negatives or false-negatived on every relevant record in this
            stratum) -- never coerced to 0.0.
        n: Number of gold-relevant records this pair's correlation was
            computed over.
        both: Records where both vendors produced a false negative.
        only_a: Records where only the first-named vendor did.
        only_b: Records where only the second-named vendor did.
        neither: Records where neither vendor did.
    """

    correlation: float | None
    n: int
    both: int
    only_a: int
    only_b: int
    neither: int

    def to_dict(self) -> dict[str, Any]:
        """Return this pair's correlation summary as a plain dict."""
        return {
            "correlation": self.correlation,
            "n": self.n,
            "both": self.both,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "neither": self.neither,
        }


@dataclass
class ErrorCorrelation:
    """Correlation of ensemble member errors on the relevant class.

    Attributes:
        pairwise_fn_on_relevant: Mapping of "vendorA|vendorB" to that pair's
            `PairwiseFnCorrelation`, computed over records that are
            actually relevant (gold include). Every vendor pair present in
            the run is included, even when `correlation` is `None`: a
            sparse stratum's undefined correlation is reported with its
            `n` and joint counts rather than silently omitted.
    """

    pairwise_fn_on_relevant: dict[str, PairwiseFnCorrelation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this error-correlation summary as a plain dict."""
        return {
            "pairwise_fn_on_relevant": {
                key: result.to_dict() for key, result in self.pairwise_fn_on_relevant.items()
            }
        }


@dataclass
class Recall:
    """Screening recall (sensitivity), estimated only from the random audit plane.

    Recall is TP/(TP+FN). Because the audit sample is typically small, the
    point estimate must always be reported alongside a rule-of-three
    worst-case floor -- never the point estimate alone.

    Attributes:
        point: Point estimate of recall, or None if not yet audited.
        floor: Rule-of-three/Wilson worst-case floor for recall (an
            asymptotic approximation), or None if not yet audited.
        exact_floor: Exact, hypergeometric one-sided recall floor (see
            `attest.stats.recall.hypergeometric_upper_bound`), or None if
            not yet audited. A genuine finite-population design-based
            bound, reported alongside `floor` rather than replacing it.
            When `tp_estimation_method` is `"inclusion_audit"`, this is the
            *joint* floor from `stratified_recall_with_audited_tp` --
            valid at the requested confidence simultaneously across both
            the TP-side and FN-side bounds, not just the FN side.
        ci: Optional confidence interval as (low, high).
        audit_n: Number of records drawn in the random recall audit.
        audit_budget_note: Free-text note on the audit sampling budget.
        tp_estimation_method: How TP (the numerator of `point`/`floor`/
            `exact_floor`) was obtained: `"full_review"` if the
            include-and-escalate set was reviewed in full (TP known
            exactly), or `"inclusion_audit"` if TP was instead scaled up
            from a sampled inclusion audit (see
            `attest.planes.inclusion_audit`), in which case `exact_floor`
            carries the joint TP/FN propagation described above.
        estimated_true_positives: The audit-scaled point estimate of TP
            used, when `tp_estimation_method` is `"inclusion_audit"`.
            `None` when TP was fully reviewed (the count is then exact and
            already visible in `confusion["tp"]`, with no separate
            estimate to report).
    """

    point: float | None = None
    floor: float | None = None
    exact_floor: float | None = None
    ci: tuple[float, float] | None = None
    audit_n: int = 0
    audit_budget_note: str = ""
    tp_estimation_method: str = "full_review"
    estimated_true_positives: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this recall summary as a plain dict."""
        return {
            "point": self.point,
            "floor": self.floor,
            "exact_floor": self.exact_floor,
            "ci": list(self.ci) if self.ci is not None else None,
            "audit_n": self.audit_n,
            "audit_budget_note": self.audit_budget_note,
            "tp_estimation_method": self.tp_estimation_method,
            "estimated_true_positives": self.estimated_true_positives,
        }


@dataclass
class Prisma:
    """Full PRISMA flow counts, extending the prefilter's upstream counts through screening.

    Attributes:
        identified: Records identified before dedup (from the prefilter).
        duplicates_removed: Records collapsed by exact-key dedup (from the prefilter).
        after_dedup: Records remaining after dedup (from the prefilter).
        prefilter_excluded: Records excluded by deterministic prefilter rules.
        screened: Records passed to the ensemble for screening.
        screen_excluded: Records excluded by the ensemble.
        included: Records included after screening.
    """

    identified: int = 0
    duplicates_removed: int = 0
    after_dedup: int = 0
    prefilter_excluded: int = 0
    screened: int = 0
    screen_excluded: int = 0
    included: int = 0

    @classmethod
    def from_prefilter(cls, prefilter_prisma: PrefilterPrisma) -> Prisma:
        """Seed the upstream PRISMA counts from a completed prefilter run.

        Maps the prefilter's `passed` count onto `screened`, since records
        that pass the prefilter are exactly those handed to the ensemble.
        """
        return cls(
            identified=prefilter_prisma.identified,
            duplicates_removed=prefilter_prisma.duplicates_removed,
            after_dedup=prefilter_prisma.after_dedup,
            prefilter_excluded=prefilter_prisma.prefilter_excluded,
            screened=prefilter_prisma.passed,
        )

    def to_dict(self) -> dict[str, int]:
        """Return these PRISMA counts as a plain dict."""
        return {
            "identified": self.identified,
            "duplicates_removed": self.duplicates_removed,
            "after_dedup": self.after_dedup,
            "prefilter_excluded": self.prefilter_excluded,
            "screened": self.screened,
            "screen_excluded": self.screen_excluded,
            "included": self.included,
        }


@dataclass
class ValidationRecord:
    """One self-validation record for a stable ensemble configuration epoch."""

    ensemble_config_id: str
    epoch: str
    config: Config
    agreement: Agreement = field(default_factory=Agreement)
    error_correlation: ErrorCorrelation = field(default_factory=ErrorCorrelation)
    escalation_rate: float | None = None
    unresolved_escalations: int = 0
    recall: Recall = field(default_factory=Recall)
    confusion: dict[str, int] = field(default_factory=dict)
    prisma: Prisma = field(default_factory=Prisma)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return this validation record as a plain dict matching the wire shape."""
        return {
            "schema_version": self.schema_version,
            "ensemble_config_id": self.ensemble_config_id,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
            "agreement": self.agreement.to_dict(),
            "error_correlation": self.error_correlation.to_dict(),
            "escalation_rate": self.escalation_rate,
            "unresolved_escalations": self.unresolved_escalations,
            "recall": self.recall.to_dict(),
            "confusion": dict(self.confusion),
            "prisma": self.prisma.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize this validation record to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def build(
    *,
    ensemble_config_id: str,
    epoch: str,
    config: Config,
    prefilter_prisma: PrefilterPrisma | None = None,
) -> ValidationRecord:
    """Build a new validation record for an ensemble configuration epoch.

    Args:
        ensemble_config_id: Immutable identifier of the ensemble configuration.
        epoch: Identifier of the stable epoch this record reports on.
        config: The ensemble configuration under evaluation.
        prefilter_prisma: If given, seeds the upstream PRISMA counts
            (identified, duplicates_removed, after_dedup, prefilter_excluded,
            screened) from a completed prefilter run.

    Returns:
        A `ValidationRecord` with upstream fields seeded and downstream
        statistics, audit, and escalation fields left at their defaults for
        the kernel to fill in later.
    """
    prisma = Prisma.from_prefilter(prefilter_prisma) if prefilter_prisma is not None else Prisma()
    return ValidationRecord(
        ensemble_config_id=ensemble_config_id,
        epoch=epoch,
        config=config,
        prisma=prisma,
    )
