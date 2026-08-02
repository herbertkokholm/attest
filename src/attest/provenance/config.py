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
    """

    vendors: dict[str, VendorSpec] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0

    @property
    def x(self) -> int:
        """Ensemble size, derived from the number of participating vendors."""
        return len(self.vendors)

    def to_dict(self) -> dict[str, Any]:
        """Return this configuration as a plain dict, including the derived `x`."""
        return {
            "vendors": {name: spec.to_dict() for name, spec in self.vendors.items()},
            "aggregation": self.aggregation,
            "tau": self.tau,
            "x": self.x,
        }


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
