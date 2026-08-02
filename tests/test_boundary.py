"""Tests for tools/check_boundary.py: the kernel's import-boundary guard."""

from __future__ import annotations

from pathlib import Path

from check_boundary import check_tree


def test_real_kernel_has_no_boundary_violations() -> None:
    violations = check_tree()
    assert violations == [], [str(v) for v in violations]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_network_import_outside_vendors_is_a_violation(tmp_path: Path) -> None:
    kernel_root = tmp_path / "attest"
    _write(kernel_root / "prefilter" / "framework.py", "import urllib.request\n")

    violations = check_tree(kernel_root)

    assert len(violations) == 1
    assert violations[0].module == "urllib.request"
    assert "network-client" in violations[0].reason


def test_network_import_inside_vendors_is_allowed(tmp_path: Path) -> None:
    kernel_root = tmp_path / "attest"
    _write(kernel_root / "vendors" / "providers" / "openmodel.py", "import urllib.request\n")

    assert check_tree(kernel_root) == []


def test_storage_scheduler_and_web_imports_are_forbidden_everywhere(tmp_path: Path) -> None:
    kernel_root = tmp_path / "attest"
    _write(
        kernel_root / "io" / "store.py",
        "import sqlite3\nimport boto3\n",
    )
    _write(kernel_root / "vendors" / "registry.py", "import celery\n")
    _write(kernel_root / "cli.py", "from flask import Flask\n")

    violations = check_tree(kernel_root)
    modules = {v.module for v in violations}

    assert modules == {"sqlite3", "boto3", "celery", "flask"}


def test_relative_imports_are_ignored(tmp_path: Path) -> None:
    kernel_root = tmp_path / "attest"
    _write(kernel_root / "cli.py", "from . import io\nfrom .io import store\n")

    assert check_tree(kernel_root) == []
