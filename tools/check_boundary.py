"""Import-boundary guard for the attest kernel.

Statically scans every module under `src/attest` and fails if a module
outside `attest.vendors` imports a network-client library, or if any kernel
module (including `attest.vendors`) imports a scheduler, database/object-storage,
or web-framework library. This is the mechanical enforcement of the rule
stated in README.md: the kernel may call LLM vendor APIs, only through
`attest.vendors`, and must never fetch candidate records, schedule, persist
to a database or object store, or serve HTTP/UI.

Run directly (`python tools/check_boundary.py`) or import `check_tree` /
`iter_violations` to run the same check programmatically, e.g. from a test.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_ROOT = REPO_ROOT / "src" / "attest"
VENDOR_PACKAGE = "vendors"

# Network-client libraries: permitted only inside attest.vendors, since that
# is the sole sanctioned path to an LLM vendor API.
NETWORK_MODULES = (
    "socket",
    "http.client",
    "http.server",
    "urllib.request",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "grpc",
    "websocket",
    "websockets",
    "ftplib",
    "smtplib",
    "telnetlib",
)

# Scheduling libraries: never permitted anywhere in the kernel. Scheduling
# candidate work belongs to the product shell, not attest.
SCHEDULER_MODULES = (
    "celery",
    "apscheduler",
    "rq",
    "dramatiq",
    "airflow",
    "prefect",
    "luigi",
    "huey",
)

# Database / object-storage libraries: never permitted anywhere in the
# kernel. attest persists only through attest.io.store's local JSON files.
STORAGE_MODULES = (
    "boto3",
    "botocore",
    "sqlite3",
    "sqlalchemy",
    "psycopg2",
    "psycopg",
    "pymongo",
    "redis",
    "s3fs",
    "google.cloud.storage",
    "azure.storage",
)

# Web-framework / server libraries: never permitted anywhere in the kernel.
# Serving HTTP or UI is the product shell's job.
WEB_MODULES = (
    "flask",
    "django",
    "fastapi",
    "starlette",
    "uvicorn",
    "tornado",
    "bottle",
    "gunicorn",
    "wsgiref",
    "asgiref",
)

_ALWAYS_FORBIDDEN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scheduler", SCHEDULER_MODULES),
    ("database/object-storage", STORAGE_MODULES),
    ("web-framework", WEB_MODULES),
)


@dataclass(frozen=True)
class Violation:
    """One import that crosses the kernel's boundary rule.

    Attributes:
        path: Source file the offending import was found in, relative to
            the repository root.
        line: Line number of the offending import statement.
        module: Dotted module name that was imported.
        reason: Human-readable explanation of which rule it broke.
    """

    path: Path
    line: int
    module: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: import of '{self.module}' {self.reason}"


def _imported_module_names(node: ast.stmt) -> Iterator[str]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module is not None:
            yield node.module


def _matches(module: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return None


def _is_vendor_module(path: Path, kernel_root: Path) -> bool:
    return VENDOR_PACKAGE in path.relative_to(kernel_root).parts


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _check_file(path: Path, kernel_root: Path) -> Iterator[Violation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    in_vendors = _is_vendor_module(path, kernel_root)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for module in _imported_module_names(node):
            if not in_vendors and _matches(module, NETWORK_MODULES):
                yield Violation(
                    path=_display_path(path),
                    line=node.lineno,
                    module=module,
                    reason="is a network-client library outside attest.vendors",
                )
                continue
            for category, prefixes in _ALWAYS_FORBIDDEN:
                if _matches(module, prefixes):
                    yield Violation(
                        path=_display_path(path),
                        line=node.lineno,
                        module=module,
                        reason=f"is a {category} library, which the kernel must never use",
                    )
                    break


def iter_violations(kernel_root: Path = KERNEL_ROOT) -> Iterator[Violation]:
    """Yield every boundary violation found under `kernel_root`.

    Args:
        kernel_root: Root package directory to scan, defaulting to
            `src/attest`. Overridable in tests to scan a synthetic tree.
    """
    for path in sorted(kernel_root.rglob("*.py")):
        yield from _check_file(path, kernel_root)


def check_tree(kernel_root: Path = KERNEL_ROOT) -> list[Violation]:
    """Return every boundary violation found under `kernel_root` as a list."""
    return list(iter_violations(kernel_root))


def main() -> int:
    """Run the boundary guard over the kernel and print any violations.

    Returns:
        0 if no violations were found, 1 otherwise.
    """
    violations = check_tree()
    if not violations:
        print("boundary guard: OK -- no forbidden imports found under src/attest")
        return 0
    print(f"boundary guard: {len(violations)} violation(s) found:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
