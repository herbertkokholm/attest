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


@dataclass(frozen=True)
class VendorSpec:
    """One ensemble member's model and prompt versioning.

    Attributes:
        model: Model identifier used by this vendor (e.g. "gpt-4o").
        model_version: Version string of the model.
        prompt_version: Version identifier of the screening prompt used with
            this vendor.
    """

    model: str
    model_version: str
    prompt_version: str

    def to_dict(self) -> dict[str, str]:
        """Return this vendor spec as a plain dict."""
        return {
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
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
    """

    vendors: dict[str, VendorSpec] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0
    default_prompt: str | None = None
    track_prompts: dict[str, str] = field(default_factory=dict)

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
        these fields existed.
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
