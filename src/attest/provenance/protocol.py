"""Validation protocol descriptor: versioned separately from the screening ensemble config.

The manuscript's design (see docs/sentinel_drift_rule.md and README) keeps three
provenance levels statistically and administratively distinct:

- **screening config** (`attest.provenance.config.Config`): vendors, models,
  prompts, aggregation, tau, zero policy -- the only thing `ensemble_config_id`
  hashes. This is unchanged by this module.
- **validation protocol** (this module): audit-strata/design, adjudication
  protocol, the sentinel-drift rule's thresholds, and the reporting/analysis
  spec. Pure analysis-plan choices -- never part of `ensemble_config_id`,
  since they do not change what a vendor samples or the ensemble's own
  aggregate decision.
- **run manifest** (`attest.provenance.manifest`): software version, input
  hash, seeds, and per-artifact content hashes for one concrete execution.

Keeping the protocol out of `ensemble_config_id` matters: two runs of the
same screening configuration under two different audit designs (e.g. a
600-record vs. a 3000-record audit budget) are the same measurement
instrument, not two different ones -- only a validation-protocol change, not
a new epoch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

STRATIFY_NONE = "none"
STRATIFY_TRACK = "track"
STRATIFY_CONFIDENCE = "confidence"

KNOWN_STRATIFICATIONS = (STRATIFY_NONE, STRATIFY_TRACK, STRATIFY_CONFIDENCE)

DEFAULT_HARD_TRIGGER_CROSSINGS = 2
DEFAULT_ADVISORY_ALPHA_THRESHOLD = 0.80


class ProtocolError(ValueError):
    """Raised when a validation-protocol descriptor violates its invariants."""


@dataclass(frozen=True)
class AuditDesign:
    """The random recall-audit's sampling design.

    Attributes:
        stratify_by: One of `KNOWN_STRATIFICATIONS` -- what `audit-draw`
            stratifies the random sample by, if anything.
        audit_size_policy: Free-text description of how the audit sample
            size was chosen (e.g. "rule-of-three floor <= 0.005: n=600"),
            not a re-derivation of the rule-of-three math itself (see
            `attest.stats.recall`).
        confidence_level: Two-sided confidence level used for the recall
            floor and interval (see `attest.stats.recall.stratified_recall`).
    """

    stratify_by: str = STRATIFY_NONE
    audit_size_policy: str = ""
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.stratify_by not in KNOWN_STRATIFICATIONS:
            raise ProtocolError(
                f"unknown stratify_by '{self.stratify_by}': expected one of {KNOWN_STRATIFICATIONS}"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise ProtocolError(
                f"confidence_level must be strictly between 0 and 1, got {self.confidence_level}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratify_by": self.stratify_by,
            "audit_size_policy": self.audit_size_policy,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditDesign:
        return cls(
            stratify_by=payload.get("stratify_by", STRATIFY_NONE),
            audit_size_policy=payload.get("audit_size_policy", ""),
            confidence_level=payload.get("confidence_level", 0.95),
        )


@dataclass(frozen=True)
class AdjudicationProtocol:
    """The human-adjudication protocol in force for escalated records.

    Attributes:
        protocol_version: Free-text/semver identifier for the adjudication
            protocol (instructions, training, escalation criteria) reviewers
            follow. Referenced by `attest.planes.adjudication.AdjudicationItem.protocol_id`
            so a resolved item is traceable to the protocol version that
            governed it.
        description: Free-text description of the protocol.
    """

    protocol_version: str = "1"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "description": self.description}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AdjudicationProtocol:
        return cls(
            protocol_version=payload.get("protocol_version", "1"),
            description=payload.get("description", ""),
        )


@dataclass(frozen=True)
class SentinelPolicy:
    """Thresholds for the latent-vendor-drift sentinel (see `attest.provenance.sentinel`).

    Attributes:
        hard_trigger_crossings: Minimum number of one-directional polarity
            crossings (`baseline in {0, +1} -> current == -1`) on the frozen
            sentinel set that opens a new epoch for a vendor. The
            `docs/sentinel_drift_rule.md` analysis recommends >= 2.
        advisory_alpha_threshold: Krippendorff's-alpha bound below which a
            vendor's baseline/current comparison is flagged for human review
            without opening an epoch on its own (advisory-only signal; see
            the doc's "Feinstein & Cicchetti paradox" section for why this
            is deliberately looser than a would-be hard threshold).
        cadence_note: Free-text description of how often the sentinel is
            run. Scheduling itself is an operational concern that belongs to
            the runbook, not the kernel -- this field only documents the
            cadence a given protocol assumes, it does not enforce or run it.
    """

    hard_trigger_crossings: int = DEFAULT_HARD_TRIGGER_CROSSINGS
    advisory_alpha_threshold: float = DEFAULT_ADVISORY_ALPHA_THRESHOLD
    cadence_note: str = ""

    def __post_init__(self) -> None:
        if self.hard_trigger_crossings < 1:
            raise ProtocolError(
                f"hard_trigger_crossings must be >= 1, got {self.hard_trigger_crossings}"
            )
        if not (0.0 <= self.advisory_alpha_threshold <= 1.0):
            raise ProtocolError(
                f"advisory_alpha_threshold must be in [0, 1], got {self.advisory_alpha_threshold}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_trigger_crossings": self.hard_trigger_crossings,
            "advisory_alpha_threshold": self.advisory_alpha_threshold,
            "cadence_note": self.cadence_note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SentinelPolicy:
        return cls(
            hard_trigger_crossings=payload.get(
                "hard_trigger_crossings", DEFAULT_HARD_TRIGGER_CROSSINGS
            ),
            advisory_alpha_threshold=payload.get(
                "advisory_alpha_threshold", DEFAULT_ADVISORY_ALPHA_THRESHOLD
            ),
            cadence_note=payload.get("cadence_note", ""),
        )


@dataclass(frozen=True)
class ReportingSpec:
    """The reporting/analysis specification a validation record is assembled against.

    Attributes:
        validation_record_schema_version: The
            `attest.contracts.validation_record.SCHEMA_VERSION` this
            protocol expects `attest validate` to produce.
        notes: Free-text notes on the analysis plan (e.g. which frameworks
            are followed -- PRISMA 2020, PRISMA-S, TRIPOD-LLM).
    """

    validation_record_schema_version: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_record_schema_version": self.validation_record_schema_version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReportingSpec:
        return cls(
            validation_record_schema_version=payload.get("validation_record_schema_version", ""),
            notes=payload.get("notes", ""),
        )


@dataclass
class ValidationProtocol:
    """A full validation-protocol descriptor, hashed separately from the screening config.

    Attributes:
        audit_design: The random recall-audit's sampling design.
        adjudication_protocol: The human-adjudication protocol in force.
        sentinel_policy: Thresholds for the latent-vendor-drift sentinel.
        reporting_spec: The reporting/analysis specification.
    """

    audit_design: AuditDesign = field(default_factory=AuditDesign)
    adjudication_protocol: AdjudicationProtocol = field(default_factory=AdjudicationProtocol)
    sentinel_policy: SentinelPolicy = field(default_factory=SentinelPolicy)
    reporting_spec: ReportingSpec = field(default_factory=ReportingSpec)

    def to_dict(self) -> dict[str, Any]:
        """Return this protocol as a plain dict, canonical for hashing and persistence."""
        return {
            "audit_design": self.audit_design.to_dict(),
            "adjudication_protocol": self.adjudication_protocol.to_dict(),
            "sentinel_policy": self.sentinel_policy.to_dict(),
            "reporting_spec": self.reporting_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationProtocol:
        """Reconstruct a protocol from a plain dict produced by `to_dict` (or `RunStore`)."""
        return cls(
            audit_design=AuditDesign.from_dict(payload.get("audit_design", {})),
            adjudication_protocol=AdjudicationProtocol.from_dict(
                payload.get("adjudication_protocol", {})
            ),
            sentinel_policy=SentinelPolicy.from_dict(payload.get("sentinel_policy", {})),
            reporting_spec=ReportingSpec.from_dict(payload.get("reporting_spec", {})),
        )


def compute_protocol_id(protocol: ValidationProtocol) -> str:
    """Derive a stable, content-based identifier for a validation protocol.

    Mirrors `attest.provenance.config.compute_ensemble_config_id`'s approach
    (canonical, key-sorted JSON, SHA-256) but is computed over a completely
    disjoint payload: a protocol change never touches, and is never touched
    by, `ensemble_config_id`.

    Args:
        protocol: The validation protocol to identify.

    Returns:
        A hex-encoded SHA-256 digest of the protocol's canonical form.
    """
    canonical = json.dumps(protocol.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_ADVISORY_ALPHA_THRESHOLD",
    "DEFAULT_HARD_TRIGGER_CROSSINGS",
    "KNOWN_STRATIFICATIONS",
    "STRATIFY_CONFIDENCE",
    "STRATIFY_NONE",
    "STRATIFY_TRACK",
    "AdjudicationProtocol",
    "AuditDesign",
    "ProtocolError",
    "ReportingSpec",
    "SentinelPolicy",
    "ValidationProtocol",
    "compute_protocol_id",
]
