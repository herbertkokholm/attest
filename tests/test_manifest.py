"""Tests for attest.provenance.manifest: manifest building and offline integrity verification."""

from __future__ import annotations

from pathlib import Path

from attest.provenance.manifest import (
    build_manifest,
    compute_artifact_hashes,
    hash_bytes,
    hash_file,
    verify_artifacts,
)

_ARTIFACTS = ("config.json", "votes.json", "decisions.json")


def _write(root: Path, name: str, content: str) -> None:
    (root / name).write_text(content, encoding="utf-8")


def test_hash_file_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert hash_file(tmp_path / "missing.json") is None


def test_hash_file_matches_hash_bytes(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", '{"x": 1}')

    assert hash_file(tmp_path / "a.json") == hash_bytes(b'{"x": 1}')


def test_compute_artifact_hashes_skips_missing_files(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", "{}")

    hashes = compute_artifact_hashes(tmp_path, _ARTIFACTS)

    assert set(hashes) == {"config.json"}


def test_build_manifest_is_deterministic_given_the_same_inputs(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", '{"tau": 0.5}')

    from datetime import UTC, datetime

    fixed_time = datetime(2026, 1, 1, tzinfo=UTC)
    first = build_manifest(
        root=tmp_path,
        artifact_filenames=_ARTIFACTS,
        ensemble_config_id="cfg-1",
        created_at=fixed_time,
        sw_version="attest 0.0.0",
    )
    second = build_manifest(
        root=tmp_path,
        artifact_filenames=_ARTIFACTS,
        ensemble_config_id="cfg-1",
        created_at=fixed_time,
        sw_version="attest 0.0.0",
    )

    assert first.manifest_id == second.manifest_id


def test_build_manifest_id_changes_when_an_artifact_changes(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    fixed_time = datetime(2026, 1, 1, tzinfo=UTC)
    _write(tmp_path, "config.json", '{"tau": 0.5}')
    before = build_manifest(
        root=tmp_path,
        artifact_filenames=_ARTIFACTS,
        ensemble_config_id="cfg-1",
        created_at=fixed_time,
    )

    _write(tmp_path, "config.json", '{"tau": 0.6}')
    after = build_manifest(
        root=tmp_path,
        artifact_filenames=_ARTIFACTS,
        ensemble_config_id="cfg-1",
        created_at=fixed_time,
    )

    assert before.manifest_id != after.manifest_id


def test_verify_artifacts_ok_when_nothing_changed(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "votes.json", "{}")
    manifest = build_manifest(
        root=tmp_path, artifact_filenames=_ARTIFACTS, ensemble_config_id="cfg-1"
    )

    report = verify_artifacts(tmp_path, manifest)

    assert report.ok is True
    assert report.problems == ()


def test_verify_artifacts_detects_tampered_content(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", "{}")
    manifest = build_manifest(
        root=tmp_path, artifact_filenames=_ARTIFACTS, ensemble_config_id="cfg-1"
    )

    _write(tmp_path, "config.json", '{"tampered": true}')

    report = verify_artifacts(tmp_path, manifest)

    assert report.ok is False
    assert len(report.problems) == 1
    assert report.problems[0].artifact == "config.json"
    assert report.problems[0].kind == "hash_mismatch"


def test_verify_artifacts_detects_missing_file(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", "{}")
    manifest = build_manifest(
        root=tmp_path, artifact_filenames=_ARTIFACTS, ensemble_config_id="cfg-1"
    )

    (tmp_path / "config.json").unlink()

    report = verify_artifacts(tmp_path, manifest)

    assert report.ok is False
    assert report.problems[0].kind == "missing"


def test_manifest_round_trips_through_dict(tmp_path: Path) -> None:
    _write(tmp_path, "config.json", "{}")
    manifest = build_manifest(
        root=tmp_path,
        artifact_filenames=_ARTIFACTS,
        ensemble_config_id="cfg-1",
        protocol_id="proto-1",
        input_hash="deadbeef",
        input_source="data/gold.json",
        seeds={"screen": 42},
    )

    restored = type(manifest).from_dict(manifest.to_dict())

    assert restored == manifest
