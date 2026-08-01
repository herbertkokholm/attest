"""Tests for attest.prefilter.framework."""

from __future__ import annotations

from attest.contracts.input import ExternalId, Record
from attest.prefilter.framework import Prefilter, require_nonempty


def _record(
    record_id: str,
    abstract: str = "some abstract",
    doi: str | None = None,
) -> Record:
    ids = [ExternalId(kind="doi", value=doi)] if doi else []
    return Record(id=record_id, title="title", abstract=abstract, track=1, ids=ids)


def test_duplicate_doi_collapses_to_one_kept_record() -> None:
    records = [
        _record("rec-a", doi="10.1/x"),
        _record("rec-b", doi="10.1/x"),
    ]

    outcome = Prefilter(rules=[]).run(records)

    assert [r.id for r in outcome.kept] == ["rec-a"]
    assert len(outcome.duplicates) == 1
    assert outcome.duplicates[0].record_id == "rec-b"
    assert outcome.duplicates[0].kept_id == "rec-a"


def test_require_nonempty_excludes_empty_abstract() -> None:
    records = [
        _record("rec-a", abstract="has content"),
        _record("rec-b", abstract="   "),
    ]

    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)

    assert [r.id for r in outcome.kept] == ["rec-a"]
    assert len(outcome.excluded) == 1
    assert outcome.excluded[0].record_id == "rec-b"
    assert outcome.excluded[0].code == "empty_abstract"


def test_prisma_counts_are_consistent() -> None:
    records = [
        _record("rec-a", doi="10.1/x"),
        _record("rec-b", doi="10.1/x"),
        _record("rec-c", abstract=""),
        _record("rec-d"),
    ]

    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)
    prisma = outcome.prisma

    assert prisma.identified == prisma.duplicates_removed + prisma.after_dedup
    assert prisma.after_dedup == prisma.prefilter_excluded + prisma.passed
    assert prisma.identified == 4
    assert prisma.duplicates_removed == 1
    assert prisma.prefilter_excluded == 1
    assert prisma.passed == 2
