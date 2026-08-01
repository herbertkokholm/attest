"""Validation-record contract v1.0: one self-validation record per stable ensemble epoch.

This is a stable, versioned interface. Changing the shape produced here
requires a version bump to ``SCHEMA_VERSION``, not an in-place edit.

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

from attest.prefilter.framework import Prisma as PrefilterPrisma

SCHEMA_VERSION = "1.0"


@dataclass
class Config:
    """The immutable ensemble configuration under evaluation.

    Attributes:
        vendors: LLM vendor names participating in the ensemble.
        models: Mapping of vendor name to model identifier.
        prompts: Mapping of prompt role (e.g. "screening") to prompt text or id.
        aggregation: Name of the aggregation strategy across ensemble votes.
        tau: Decision threshold used by the aggregation strategy.
        x: Ensemble size (number of voting members) at this epoch.
    """

    vendors: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0
    x: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return this configuration as a plain dict."""
        return {
            "vendors": list(self.vendors),
            "models": dict(self.models),
            "prompts": dict(self.prompts),
            "aggregation": self.aggregation,
            "tau": self.tau,
            "x": self.x,
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


@dataclass
class ErrorCorrelation:
    """Correlation of ensemble member errors on the relevant class.

    Attributes:
        pairwise_fn_on_relevant: Mapping of "vendorA|vendorB" to the
            correlation of false negatives between the two members, computed
            over records that are actually relevant (gold include).
    """

    pairwise_fn_on_relevant: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this error-correlation summary as a plain dict."""
        return {"pairwise_fn_on_relevant": dict(self.pairwise_fn_on_relevant)}


@dataclass
class Recall:
    """Screening recall (sensitivity), estimated only from the random audit plane.

    Recall is TP/(TP+FN). Because the audit sample is typically small, the
    point estimate must always be reported alongside a rule-of-three
    worst-case floor -- never the point estimate alone.

    Attributes:
        point: Point estimate of recall, or None if not yet audited.
        floor: Rule-of-three worst-case floor for recall, or None if not yet audited.
        ci: Optional confidence interval as (low, high).
        audit_n: Number of records drawn in the random recall audit.
        audit_budget_note: Free-text note on the audit sampling budget.
    """

    point: float | None = None
    floor: float | None = None
    ci: tuple[float, float] | None = None
    audit_n: int = 0
    audit_budget_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this recall summary as a plain dict."""
        return {
            "point": self.point,
            "floor": self.floor,
            "ci": list(self.ci) if self.ci is not None else None,
            "audit_n": self.audit_n,
            "audit_budget_note": self.audit_budget_note,
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
