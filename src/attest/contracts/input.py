"""Input contract v1.0: the wire shape for records submitted to attest for screening.

This is a stable, versioned interface. Changing the shape validated here
requires a version bump to ``SCHEMA_VERSION`` (and, for breaking changes, to
``SUPPORTED_MAJOR_VERSIONS``), not an in-place edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR_VERSIONS = {"1"}
REQUIRED_RECORD_FIELDS = ("id", "title", "abstract", "track")
VALID_GOLD_LABELS = (-1, 0, 1)


class ContractError(ValueError):
    """Raised when a payload violates the input contract."""


@dataclass(frozen=True)
class ExternalId:
    """A normalized external identifier for a record, e.g. a DOI or PMID."""

    kind: str
    value: str


@dataclass
class Record:
    """A single record submitted for screening.

    Attributes:
        id: Stable, producer-assigned deduplication key for this record.
        title: Title text. May not be empty.
        abstract: Abstract text. May be empty: the prefilter decides what to
            do with empty abstracts, not the contract.
        track: The screening track this record belongs to (int or string).
        ids: Known external identifiers (DOI, arXiv, PMID, ...) for this
            record, normalized and used for exact-key deduplication.
        gold_label: Optional ground-truth label for validation purposes, one
            of -1 (exclude), 0 (related/uncertain), or 1 (include).
    """

    id: str
    title: str
    abstract: str
    track: int | str
    ids: list[ExternalId] = field(default_factory=list)
    gold_label: int | None = None

    @property
    def has_gold(self) -> bool:
        """Whether this record carries a ground-truth label."""
        return self.gold_label is not None


@dataclass
class NormalizedInput:
    """A validated, normalized input payload ready for the prefilter and ensemble."""

    schema_version: str
    project: str
    records: list[Record]

    @property
    def gold_records(self) -> list[Record]:
        """Records that carry a ground-truth label."""
        return [r for r in self.records if r.has_gold]


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{what} must be a string, got {type(value).__name__}")
    return value


def _normalize_ids(raw_ids: Any, record_id: str) -> list[ExternalId]:
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list):
        raise ContractError(f"record '{record_id}': ids must be a list")
    normalized: list[ExternalId] = []
    for entry in raw_ids:
        if not isinstance(entry, dict) or "kind" not in entry or "value" not in entry:
            raise ContractError(
                f"record '{record_id}': each id must be an object with 'kind' and 'value'"
            )
        kind = _require_str(entry["kind"], f"record '{record_id}' id kind").strip().lower()
        value = _require_str(entry["value"], f"record '{record_id}' id value").strip().lower()
        if not kind or not value:
            raise ContractError(f"record '{record_id}': id kind and value must be non-empty")
        normalized.append(ExternalId(kind=kind, value=value))
    return normalized


def _normalize_record(raw: Any, seen_ids: set[str]) -> Record:
    if not isinstance(raw, dict):
        raise ContractError(f"each record must be an object, got {type(raw).__name__}")

    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in raw]
    if missing:
        raise ContractError(f"record missing required field(s): {', '.join(missing)}")

    record_id = _require_str(raw["id"], "record id").strip()
    if not record_id:
        raise ContractError("record id must be non-empty")
    if record_id in seen_ids:
        raise ContractError(f"duplicate record id: '{record_id}'")
    seen_ids.add(record_id)

    title = _require_str(raw["title"], f"record '{record_id}' title")
    abstract = _require_str(raw["abstract"], f"record '{record_id}' abstract")

    track = raw["track"]
    if isinstance(track, bool) or not isinstance(track, (int, str)):
        raise ContractError(f"record '{record_id}': track must be an int or string")
    if isinstance(track, str) and not track.strip():
        raise ContractError(f"record '{record_id}': track must be non-empty")

    ids = _normalize_ids(raw.get("ids"), record_id)

    gold_label: int | None = None
    if "gold_label" in raw and raw["gold_label"] is not None:
        gold_label = raw["gold_label"]
        if isinstance(gold_label, bool) or not isinstance(gold_label, int):
            raise ContractError(f"record '{record_id}': gold_label must be an int")
        if gold_label not in VALID_GOLD_LABELS:
            raise ContractError(
                f"record '{record_id}': gold_label must be one of {VALID_GOLD_LABELS}"
            )

    return Record(
        id=record_id,
        title=title,
        abstract=abstract,
        track=track,
        ids=ids,
        gold_label=gold_label,
    )


def validate_and_normalize(payload: Any) -> NormalizedInput:
    """Validate a raw input payload against the input contract and normalize it.

    Args:
        payload: A JSON-decoded object matching the input contract wire shape.

    Returns:
        A `NormalizedInput` with validated, normalized records.

    Raises:
        ContractError: On any violation of the input contract, with a message
            specific enough to locate the offending field or record.
    """
    if not isinstance(payload, dict):
        raise ContractError(f"payload must be an object, got {type(payload).__name__}")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ContractError("schema_version is required and must be a non-empty string")
    major = schema_version.split(".", 1)[0]
    if major not in SUPPORTED_MAJOR_VERSIONS:
        raise ContractError(
            f"unsupported schema_version '{schema_version}': "
            f"supported major versions are {sorted(SUPPORTED_MAJOR_VERSIONS)}"
        )

    project = payload.get("project")
    if not isinstance(project, str) or not project.strip():
        raise ContractError("project is required and must be a non-empty string")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ContractError("records is required and must be a non-empty list")

    seen_ids: set[str] = set()
    records = [_normalize_record(raw, seen_ids) for raw in raw_records]

    return NormalizedInput(schema_version=schema_version, project=project, records=records)
