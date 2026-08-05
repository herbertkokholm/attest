"""Immutable ensemble configuration and its content-derived ensemble_config_id.

An ensemble configuration is the full set of choices that determine how
records are screened: which vendors vote, which model and prompt version
each vendor uses, and how their votes are aggregated. The `ensemble_config_id`
is a content hash of that configuration, so two configurations with the same
content always resolve to the same id regardless of when or in what order
they were constructed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from attest.ensemble.aggregate import KNOWN_ZERO_POLICIES, ZERO_POLICY_ESCALATE
from attest.ensemble.confidence import DEFAULT_LOW_THRESHOLD

# Version of the kernel-owned output-format contract appended to every
# screening prompt (see `attest.vendors.base.OUTPUT_CONTRACT`). Defined here,
# not in `attest.vendors.base`, because `Config.to_dict` needs it for hashing
# and `attest.vendors.base` already imports from this module -- importing it
# back the other way would be circular. `attest.vendors.base` re-exports it.
OUTPUT_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class VendorSpec:
    """One ensemble member's model, prompt, and sampling versioning.

    Attributes:
        model: Model identifier used by this vendor (e.g. "gpt-4o").
        model_version: Version string of the model.
        prompt_version: Version identifier of the screening prompt used with
            this vendor.
        temperature: Sampling temperature used with this vendor. Versioned
            exactly like `model_version` and `prompt_version`: it is
            hash-sensitive, so changing it yields a different
            `ensemble_config_id` and opens a new epoch, the same as a model
            or prompt change would.
    """

    model: str
    model_version: str
    prompt_version: str
    temperature: float

    def to_dict(self) -> dict[str, Any]:
        """Return this vendor spec as a plain dict."""
        return {
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
        }


@dataclass
class Config:
    """An ensemble configuration descriptor.

    Attributes:
        vendors: Mapping of vendor name to that vendor's model and prompt
            versioning. The ensemble size `x` is derived from this mapping,
            never set independently.
        aggregation: Name of the aggregation rule applied to ensemble votes.
        tau: Decision threshold used by the aggregation rule.
        default_prompt: Screening prompt text sent to every vendor for a
            record whose track has no entry in `track_prompts`. `None`
            (the default) means every rater falls back to its own
            hardcoded `attest.vendors.base.DEFAULT_SCREENING_PROMPT`,
            exactly as before this field existed.
        track_prompts: Mapping of record track (stringified) to the
            screening prompt text to use for records on that track,
            overriding `default_prompt`. Lets one ensemble configuration
            screen records from several systematic reviews in a single run,
            each against its own published eligibility criteria, while
            still voting with the same vendors/models/aggregation on all of
            them.
        zero_policy: Disposition of a would-be `auto_label == 0` decision
            (see `attest.ensemble.aggregate.g`). `ZERO_POLICY_ESCALATE` (the
            default) routes it to a human via escalation; `ZERO_POLICY_INCLUDE`
            folds it into `+1`. Validated at construction; only these two
            values are accepted -- there is deliberately no "exclude" option.
        confidence_threshold: Default `low_threshold` (see
            `attest.ensemble.confidence.confidence_tier`) a confidence-
            stratified `audit-draw` uses when no `--confidence-threshold`
            override is given on that command's own invocation. Sourced
            from the same config file as `tau` for the same ergonomic
            reason -- a runbook setting the caller shouldn't have to retype
            identically on every invocation -- but, unlike `tau`,
            deliberately excluded from `to_dict()`/`ensemble_config_id`:
            see `attest.io.store._config_to_dict` for where it is instead
            persisted, and `docs/logprob_support.md` for why it must never
            be hash-versioned (it changes only how an already-fixed
            excluded population is stratified for audit, never what a
            vendor samples or the ensemble's own aggregate decision).
    """

    vendors: dict[str, VendorSpec] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0
    default_prompt: str | None = None
    track_prompts: dict[str, str] = field(default_factory=dict)
    zero_policy: str = ZERO_POLICY_ESCALATE
    confidence_threshold: float = DEFAULT_LOW_THRESHOLD

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.zero_policy not in KNOWN_ZERO_POLICIES:
            raise ValueError(
                f"unknown zero_policy '{self.zero_policy}': expected one of {KNOWN_ZERO_POLICIES}"
            )

    @property
    def x(self) -> int:
        """Ensemble size, derived from the number of participating vendors."""
        return len(self.vendors)

    def prompt_for_track(self, track: int | str) -> str | None:
        """Resolve the screening prompt to use for a record on `track`.

        Args:
            track: A record's `track` field (see `attest.contracts.input.Record`).

        Returns:
            `track_prompts[str(track)]` if set, else `default_prompt`, else
            `None` -- meaning the rater should fall back to its own
            hardcoded default prompt.
        """
        return self.track_prompts.get(str(track), self.default_prompt)

    def to_dict(self) -> dict[str, Any]:
        """Return this configuration as a plain dict, including the derived `x`.

        `default_prompt`/`track_prompts` are omitted when unset, so a
        configuration that does not use them serializes -- and hashes into
        `compute_ensemble_config_id` -- identically to one built before
        these fields existed. `output_contract_version` is always included,
        regardless: the kernel-appended output contract (see
        `attest.vendors.base.compose_system_prompt`) is appended to *every*
        composed prompt, including the no-criteria fallback, so its version
        is always sensitive to the text actually sent to a vendor on this
        config's behalf. Bumping `OUTPUT_CONTRACT_VERSION` therefore opens a
        new epoch for every config, not just those supplying criteria.
        `zero_policy` is omitted when it is the default `ZERO_POLICY_ESCALATE`,
        so a config that never sets it hashes
        identically to one built before the field existed; a config that
        opts into `ZERO_POLICY_INCLUDE` picks up a different id, correctly
        opening a new epoch for that real behavior change.
        `confidence_threshold` is never included here -- unlike every other
        field on this class, it is not part of the hashed/versioned
        contract at all (see its attribute docstring above), so it is
        persisted separately by `attest.io.store._config_to_dict` instead
        of through this method.
        """
        payload: dict[str, Any] = {
            "vendors": {name: spec.to_dict() for name, spec in self.vendors.items()},
            "aggregation": self.aggregation,
            "tau": self.tau,
            "x": self.x,
        }
        if self.default_prompt is not None:
            payload["default_prompt"] = self.default_prompt
        if self.track_prompts:
            payload["track_prompts"] = dict(self.track_prompts)
        payload["output_contract_version"] = OUTPUT_CONTRACT_VERSION
        if self.zero_policy != ZERO_POLICY_ESCALATE:
            payload["zero_policy"] = self.zero_policy
        return payload


def compute_ensemble_config_id(config: Config) -> str:
    """Derive a stable, content-based identifier for an ensemble configuration.

    The id is a SHA-256 hex digest of the configuration's canonical JSON
    serialization: keys sorted recursively and compactly separated, so the
    same configuration content always yields the same id regardless of
    construction order or Python dict iteration order, and any change to any
    field yields a different id.

    Args:
        config: The ensemble configuration to identify.

    Returns:
        A hex-encoded SHA-256 digest of the configuration's canonical form.
    """
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
