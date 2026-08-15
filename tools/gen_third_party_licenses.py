"""Regenerate the dependency table in THIRD-PARTY-LICENSES.md.

Shells out to `pip-licenses` over the current environment and replaces the
content between the `BEGIN GENERATED TABLE` / `END GENERATED TABLE` markers
in THIRD-PARTY-LICENSES.md with its output. Run this after changing
`dependencies` or `optional-dependencies` in pyproject.toml, from an
environment with all extras installed (`pip install -e ".[all]" pip-licenses`),
so the table reflects every optional vendor SDK, not just whatever happens to
be installed locally.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
TARGET = REPO_ROOT / "THIRD-PARTY-LICENSES.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED TABLE -->"
END_MARKER = "<!-- END GENERATED TABLE -->"

PIP_LICENSES_ARGS = (
    "pip-licenses",
    "--format=markdown",
    "--with-urls",
    "--order=name",
    "--ignore-packages",
    "attest",
    "--from=classifier",
)


def _normalize(name: str) -> str:
    """PEP 503 distribution-name normalization, for name comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """Extract the bare package name from a requirement string like 'ruff>=0.16.0'."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise ValueError(f"cannot parse requirement: {requirement!r}")
    return match.group(0)


def check_no_dev_tools_installed() -> None:
    """Refuse to run if pyproject.toml's `dev` extra is installed in this environment.

    THIRD-PARTY-LICENSES.md documents what a user gets from `pip install
    attest[all]`, per pyproject.toml -- the authoritative dependency source
    -- not whatever else happens to be in the environment this script runs
    in. `pip-licenses` cannot distinguish the two: it lists every installed
    distribution. Running this from an ordinary dev venv (dev tools and
    vendor extras both installed) has silently pulled dev-only packages
    (e.g. `coverage`, `ast_serialize`) into the generated table before.

    Raises:
        SystemExit: If any package from pyproject.toml's `dev` extra is
            installed in the current environment.
    """
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_requirements = pyproject["project"]["optional-dependencies"]["dev"]
    dev_names = {_normalize(_requirement_name(r)) for r in dev_requirements}

    installed_names = {_normalize(dist.name) for dist in distributions() if dist.name}
    found = sorted(dev_names & installed_names)
    if found:
        raise SystemExit(
            "refusing to regenerate THIRD-PARTY-LICENSES.md: dev-only package(s) "
            f"{found} from pyproject.toml's [dev] extra are installed in this "
            "environment. Run from a clean venv with only "
            '`pip install -e ".[all]" pip-licenses` (no [dev]).'
        )


def render_table() -> str:
    """Run pip-licenses and return its markdown table output."""
    result = subprocess.run(PIP_LICENSES_ARGS, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def regenerate() -> str:
    """Replace the generated-table section of THIRD-PARTY-LICENSES.md.

    Returns:
        The updated file content, both written to disk and returned so
        callers (e.g. a CI check) can diff it against the prior content.
    """
    content = TARGET.read_text(encoding="utf-8")
    before, _, rest = content.partition(BEGIN_MARKER)
    _, _, after = rest.partition(END_MARKER)
    updated = f"{before}{BEGIN_MARKER}\n{render_table()}\n{END_MARKER}{after}"
    TARGET.write_text(updated, encoding="utf-8")
    return updated


def main() -> int:
    check_no_dev_tools_installed()
    regenerate()
    print(f"wrote {TARGET.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
