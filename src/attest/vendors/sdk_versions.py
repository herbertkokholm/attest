"""Best-effort provider SDK version lookup, for manifest provenance.

The scheme supports integrity of what a run's manifest recorded, not
deterministic reproducibility of a vendor's own API: knowing which version of
a vendor's client library was installed is still useful replay/audit
provenance (a library upgrade can change request shape, retry behavior, or
response parsing even when the wire API is unchanged), so it belongs in the
manifest alongside `attest.provenance.manifest.software_version`.

Versions are read from currently-installed package metadata, not captured at
the moment a vendor's API was actually called -- the same "at manifest-build
time" character every other manifest field already has (see
`attest.provenance.manifest.build_manifest`'s docstring: artifact hashes are
also a snapshot at build time, not at run time). Build the manifest promptly
after a run to keep this accurate, and treat it as advisory if the
environment was later reinstalled or upgraded.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata

# Vendor name (attest.vendors.registry's keys) -> the pip distribution that
# provides its client library. "openmodel" is deliberately absent: it speaks
# to a self-hosted OpenAI-compatible HTTP endpoint directly (see
# attest.vendors.providers.openmodel), with no vendor SDK dependency to version.
_DISTRIBUTION_BY_VENDOR: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google-genai",
    "mistral": "mistralai",
    "fireworks": "fireworks-ai",
    "together": "together",
}


def sdk_version(vendor: str) -> str | None:
    """Return the installed version of `vendor`'s provider SDK, or None.

    None means either `vendor` has no SDK dependency (e.g. "openmodel") or
    its package is not installed/importable right now -- both are ordinary,
    unremarkable states here, not errors.

    Args:
        vendor: A vendor name as used in `attest.provenance.config.Config.vendors`.
    """
    distribution = _DISTRIBUTION_BY_VENDOR.get(vendor)
    if distribution is None:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def sdk_versions(vendors: Iterable[str]) -> dict[str, str]:
    """Return `{vendor: version}` for every vendor in `vendors` with a resolvable SDK version.

    Args:
        vendors: Vendor names to look up, e.g. `config.vendors.keys()`.

    Returns:
        Only vendors `sdk_version` resolved a version for -- a vendor with no
        SDK dependency, or an uninstalled/unresolvable one, is simply absent
        rather than mapped to `None`, so the manifest field stays a clean
        `dict[str, str]`.
    """
    return {vendor: version for vendor in vendors if (version := sdk_version(vendor)) is not None}
