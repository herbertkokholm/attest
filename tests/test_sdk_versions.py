"""Tests for attest.vendors.sdk_versions: best-effort provider SDK version lookup."""

from __future__ import annotations

from importlib import metadata

import pytest

from attest.vendors.sdk_versions import sdk_version, sdk_versions


def test_sdk_version_returns_installed_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "version", lambda dist: {"openai": "2.53.0"}[dist])

    assert sdk_version("openai") == "2.53.0"


def test_sdk_version_returns_none_for_a_vendor_with_no_sdk_dependency() -> None:
    assert sdk_version("openmodel") is None


def test_sdk_version_returns_none_for_an_unknown_vendor() -> None:
    assert sdk_version("not-a-real-vendor") is None


def test_sdk_version_returns_none_when_package_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(dist: str) -> str:
        raise metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(metadata, "version", _raise)

    assert sdk_version("anthropic") is None


def test_sdk_versions_omits_unresolvable_vendors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _version(dist: str) -> str:
        if dist == "openai":
            return "2.53.0"
        raise metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(metadata, "version", _version)

    result = sdk_versions(["openai", "anthropic", "openmodel"])

    assert result == {"openai": "2.53.0"}
