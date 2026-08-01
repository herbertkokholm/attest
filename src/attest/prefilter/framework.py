"""Deterministic prefilter framework.

The prefilter runs before the ensemble and owns the upstream PRISMA counts
(identified, duplicates removed, after dedup, prefilter-excluded, passed).
The framework here is source-agnostic: concrete deduplication heuristics and
exclusion rules are injected by the caller, since they are shaped by the
specific record sources (e.g. which fields are reliable, which id kinds are
present) rather than by the kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from attest.contracts.input import Record

_DEDUP_ID_PRIORITY = ("doi", "pmid", "arxiv")


@runtime_checkable
class PrefilterRule(Protocol):
    """A pure, stateless rule that either excludes a record or passes it through.

    Implementations must not mutate the record or depend on external state:
    given the same record, a rule must always return the same result.
    """

    code: str

    def __call__(self, record: Record) -> str | None:
        """Return an exclusion reason string to exclude `record`, or None to pass it."""
        ...


@dataclass
class FunctionRule:
    """A `PrefilterRule` built from a plain predicate function."""

    code: str
    predicate: Callable[[Record], str | None]

    def __call__(self, record: Record) -> str | None:
        return self.predicate(record)


def require_nonempty(field_name: str) -> FunctionRule:
    """Build a rule excluding records whose named text field is empty after stripping.

    Args:
        field_name: Name of the `Record` attribute to check (e.g. "abstract").
    """
    code = f"empty_{field_name}"

    def check(record: Record) -> str | None:
        value = getattr(record, field_name, "")
        if not isinstance(value, str) or not value.strip():
            return f"{field_name} is empty"
        return None

    return FunctionRule(code=code, predicate=check)


def default_dedup_key(record: Record) -> str:
    """Compute the default exact-match deduplication key for a record.

    Prefers a normalized external id, checked in priority order
    ("doi", "pmid", "arxiv"), formatted as "{kind}:{value.lower()}". Falls
    back to "id:{record.id.lower()}" when no known external id is present.

    This performs exact-key deduplication only. Fuzzy title/author matching
    is heavier, source-shaped, and left to a caller-injected key function
    (or a later dedup pass) rather than implemented here.
    """
    by_kind = {ext_id.kind: ext_id.value for ext_id in record.ids}
    for kind in _DEDUP_ID_PRIORITY:
        if kind in by_kind:
            return f"{kind}:{by_kind[kind].lower()}"
    return f"id:{record.id.lower()}"


@dataclass(frozen=True)
class Exclusion:
    """A record excluded by a prefilter rule."""

    record_id: str
    code: str
    reason: str


@dataclass(frozen=True)
class Duplicate:
    """A record collapsed into another record's dedup key."""

    record_id: str
    kept_id: str
    key: str


@dataclass(frozen=True)
class Prisma:
    """Upstream PRISMA flow counts owned by the prefilter."""

    identified: int
    duplicates_removed: int
    after_dedup: int
    prefilter_excluded: int
    passed: int

    def to_dict(self) -> dict[str, int]:
        """Return the PRISMA counts as a plain dict."""
        return {
            "identified": self.identified,
            "duplicates_removed": self.duplicates_removed,
            "after_dedup": self.after_dedup,
            "prefilter_excluded": self.prefilter_excluded,
            "passed": self.passed,
        }


@dataclass
class PrefilterOutcome:
    """The full result of running the prefilter over a batch of records."""

    kept: list[Record] = field(default_factory=list)
    excluded: list[Exclusion] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)
    prisma: Prisma = field(
        default_factory=lambda: Prisma(
            identified=0, duplicates_removed=0, after_dedup=0, prefilter_excluded=0, passed=0
        )
    )


@dataclass
class Prefilter:
    """Deterministic prefilter: exact-key dedup followed by injected exclusion rules.

    Args:
        rules: Ordered list of rules to apply to each surviving record. The
            first rule that returns a reason wins; later rules do not run.
        dedup_key: Function computing the exact-match dedup key for a record.
    """

    rules: list[PrefilterRule]
    dedup_key: Callable[[Record], str] = default_dedup_key

    def run(self, records: Iterable[Record]) -> PrefilterOutcome:
        """Run deduplication and then the injected rules over `records`."""
        records = list(records)
        identified = len(records)

        seen_keys: dict[str, str] = {}
        duplicates: list[Duplicate] = []
        deduped: list[Record] = []
        for record in records:
            key = self.dedup_key(record)
            if key in seen_keys:
                duplicates.append(Duplicate(record_id=record.id, kept_id=seen_keys[key], key=key))
                continue
            seen_keys[key] = record.id
            deduped.append(record)

        after_dedup = len(deduped)

        kept: list[Record] = []
        excluded: list[Exclusion] = []
        for record in deduped:
            reason = None
            code = ""
            for rule in self.rules:
                reason = rule(record)
                if reason is not None:
                    code = rule.code
                    break
            if reason is not None:
                excluded.append(Exclusion(record_id=record.id, code=code, reason=reason))
            else:
                kept.append(record)

        prisma = Prisma(
            identified=identified,
            duplicates_removed=len(duplicates),
            after_dedup=after_dedup,
            prefilter_excluded=len(excluded),
            passed=len(kept),
        )

        return PrefilterOutcome(kept=kept, excluded=excluded, duplicates=duplicates, prisma=prisma)
