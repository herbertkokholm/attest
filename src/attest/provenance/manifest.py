"""Run manifest: software/input/seed provenance and per-artifact SHA-256 integrity hashes.

A `RunManifest` is the third and last provenance level (see
`attest.provenance.protocol` for the first two): one concrete execution's
software version, the hash of the input file it screened, every random seed
it used, and a content hash of each artifact file the run directory produced
at the time the manifest was built. `verify_artifacts` recomputes those
hashes later and reports exactly which artifact is missing or has changed,
so a completed run directory can be checked offline, without re-executing
anything or trusting the filesystem timestamps.

This module never decides *which* filenames belong in a manifest -- that
list is a `RunStore` concern (see `attest.io.store.RunStore.MANIFEST_ARTIFACTS`)
so this module stays agnostic to the store's file layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a run manifest or its integrity verification is malformed."""


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of `path`'s content, or None if it does not exist."""
    if not path.exists():
        return None
    return hash_bytes(path.read_bytes())


def software_version() -> str:
    """Best-effort `attest` package version, optionally suffixed with a source commit.

    The commit is never derived by shelling out to `git` -- the kernel does
    not spawn subprocesses for provenance -- but is read from the
    `ATTEST_SOURCE_COMMIT` environment variable when the caller (typically a
    CI job or the runbook) sets it.
    """
    try:
        from importlib.metadata import version

        pkg_version = version("attest")
    except Exception:
        pkg_version = "unknown"
    commit = os.environ.get("ATTEST_SOURCE_COMMIT")
    return f"attest {pkg_version}+{commit}" if commit else f"attest {pkg_version}"


@dataclass(frozen=True)
class RunManifest:
    """One execution's software/input/seed provenance and per-artifact content hashes.

    Attributes:
        manifest_id: Content hash of every field below except itself --
            change any of them and the manifest gets a different id.
        created_at: When this manifest was built.
        software_version: `attest` package version (see `software_version`).
        ensemble_config_id: The screening configuration this run executed under.
        protocol_id: The validation protocol this run executed under, or
            None if no protocol was recorded for this run directory.
        input_hash: SHA-256 hex digest of the raw input-contract file bytes
            this run screened, or None if not recorded.
        input_source: Free-text/path identifying the input file, purely
            informational (not part of integrity verification).
        seeds: Named random seeds used by this run (e.g.
            `{"screen_deterministic": 42, "audit_draw": 7}`).
        artifact_hashes: Mapping of run-directory filename to the SHA-256
            hex digest of its content at manifest-build time.
    """

    manifest_id: str
    created_at: datetime
    software_version: str
    ensemble_config_id: str
    protocol_id: str | None = None
    input_hash: str | None = None
    input_source: str | None = None
    seeds: dict[str, int] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this manifest as a plain, JSON-serializable dict."""
        return {
            "manifest_id": self.manifest_id,
            "created_at": self.created_at.isoformat(),
            "software_version": self.software_version,
            "ensemble_config_id": self.ensemble_config_id,
            "protocol_id": self.protocol_id,
            "input_hash": self.input_hash,
            "input_source": self.input_source,
            "seeds": dict(self.seeds),
            "artifact_hashes": dict(self.artifact_hashes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        """Reconstruct a manifest from a plain dict produced by `to_dict`."""
        return cls(
            manifest_id=payload["manifest_id"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            software_version=payload["software_version"],
            ensemble_config_id=payload["ensemble_config_id"],
            protocol_id=payload.get("protocol_id"),
            input_hash=payload.get("input_hash"),
            input_source=payload.get("input_source"),
            seeds=dict(payload.get("seeds", {})),
            artifact_hashes=dict(payload.get("artifact_hashes", {})),
        )


def _payload_for_id(
    *,
    created_at: datetime,
    sw_version: str,
    ensemble_config_id: str,
    protocol_id: str | None,
    input_hash: str | None,
    input_source: str | None,
    seeds: Mapping[str, int],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "created_at": created_at.isoformat(),
        "software_version": sw_version,
        "ensemble_config_id": ensemble_config_id,
        "protocol_id": protocol_id,
        "input_hash": input_hash,
        "input_source": input_source,
        "seeds": dict(seeds),
        "artifact_hashes": dict(artifact_hashes),
    }


def compute_artifact_hashes(root: Path, filenames: Sequence[str]) -> dict[str, str]:
    """Hash every existing artifact file in `filenames`, skipping any that don't exist.

    A skipped (not-yet-produced) artifact is not an error here -- a manifest
    built mid-run legitimately covers only what exists so far -- but every
    artifact it *does* cover becomes verifiable later.

    Args:
        root: The run directory.
        filenames: Candidate artifact filenames, relative to `root`.

    Returns:
        Mapping of filename to SHA-256 hex digest, for files that exist.
    """
    return {name: digest for name in filenames if (digest := hash_file(root / name)) is not None}


def build_manifest(
    *,
    root: Path,
    artifact_filenames: Sequence[str],
    ensemble_config_id: str,
    protocol_id: str | None = None,
    input_hash: str | None = None,
    input_source: str | None = None,
    seeds: Mapping[str, int] | None = None,
    created_at: datetime | None = None,
    sw_version: str | None = None,
) -> RunManifest:
    """Build a `RunManifest` by hashing every existing artifact in `artifact_filenames`.

    Args:
        root: The run directory to hash artifacts from.
        artifact_filenames: Candidate artifact filenames to include, e.g.
            `attest.io.store.RunStore.MANIFEST_ARTIFACTS`.
        ensemble_config_id: The screening configuration this run executed under.
        protocol_id: The validation protocol this run executed under, if any.
        input_hash: SHA-256 hex digest of the input file this run screened.
        input_source: Free-text/path identifying the input file.
        seeds: Named random seeds used by this run.
        created_at: Timestamp to record; defaults to now (UTC).
        sw_version: Software version string; defaults to `software_version()`.

    Returns:
        A `RunManifest` with `manifest_id` derived from every other field.
    """
    created = created_at or datetime.now(UTC)
    sw = sw_version or software_version()
    seeds_dict = dict(seeds or {})
    artifact_hashes = compute_artifact_hashes(root, artifact_filenames)
    payload = _payload_for_id(
        created_at=created,
        sw_version=sw,
        ensemble_config_id=ensemble_config_id,
        protocol_id=protocol_id,
        input_hash=input_hash,
        input_source=input_source,
        seeds=seeds_dict,
        artifact_hashes=artifact_hashes,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    manifest_id = hash_bytes(canonical.encode("utf-8"))
    return RunManifest(
        manifest_id=manifest_id,
        created_at=created,
        software_version=sw,
        ensemble_config_id=ensemble_config_id,
        protocol_id=protocol_id,
        input_hash=input_hash,
        input_source=input_source,
        seeds=seeds_dict,
        artifact_hashes=artifact_hashes,
    )


@dataclass(frozen=True)
class IntegrityProblem:
    """One artifact that failed offline integrity verification.

    Attributes:
        artifact: Filename of the affected artifact.
        kind: `"missing"` (the manifest recorded a hash but the file is now
            absent) or `"hash_mismatch"` (the file exists but its content
            hash no longer matches the manifest).
        detail: Human-readable explanation.
    """

    artifact: str
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"artifact": self.artifact, "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class IntegrityReport:
    """The result of offline-verifying a run directory's artifacts against its manifest.

    Attributes:
        ok: True iff every artifact the manifest recorded a hash for is
            present and unchanged.
        problems: Every missing or changed artifact found, empty when `ok`.
    """

    ok: bool
    problems: tuple[IntegrityProblem, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": [p.to_dict() for p in self.problems]}


def verify_artifacts(root: Path, manifest: RunManifest) -> IntegrityReport:
    """Recompute artifact hashes under `root` and compare them against `manifest`.

    Args:
        root: The run directory to verify.
        manifest: The previously built manifest to verify against.

    Returns:
        An `IntegrityReport` listing every artifact that is missing or whose
        content hash no longer matches; `ok` is True iff that list is empty.
    """
    problems: list[IntegrityProblem] = []
    for name, expected in sorted(manifest.artifact_hashes.items()):
        actual = hash_file(root / name)
        if actual is None:
            problems.append(
                IntegrityProblem(
                    artifact=name,
                    kind="missing",
                    detail=f"'{name}' is missing from the run directory",
                )
            )
        elif actual != expected:
            problems.append(
                IntegrityProblem(
                    artifact=name,
                    kind="hash_mismatch",
                    detail=f"'{name}' content hash does not match the manifest",
                )
            )
    return IntegrityReport(ok=not problems, problems=tuple(problems))


__all__ = [
    "IntegrityProblem",
    "IntegrityReport",
    "ManifestError",
    "RunManifest",
    "build_manifest",
    "compute_artifact_hashes",
    "hash_bytes",
    "hash_file",
    "software_version",
    "verify_artifacts",
]
