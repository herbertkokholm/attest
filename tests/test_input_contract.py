"""Tests for attest.contracts.input."""

from __future__ import annotations

import copy

import pytest

from attest.contracts.input import ContractError, ExternalId, validate_and_normalize

VALID_PAYLOAD = {
    "schema_version": "1.0",
    "project": "demo",
    "records": [
        {
            "id": "rec-1",
            "title": "Title one",
            "abstract": "Abstract one",
            "track": 1,
            "ids": [{"kind": "DOI", "value": " 10.1/ABC "}],
            "gold_label": 1,
        },
        {
            "id": "rec-2",
            "title": "Title two",
            "abstract": "",
            "track": "track-b",
        },
    ],
}


def test_valid_payload_round_trips() -> None:
    normalized = validate_and_normalize(VALID_PAYLOAD)

    assert normalized.schema_version == "1.0"
    assert normalized.project == "demo"
    assert len(normalized.records) == 2

    first = normalized.records[0]
    assert first.id == "rec-1"
    assert first.has_gold is True
    assert first.ids == [ExternalId(kind="doi", value="10.1/abc")]

    second = normalized.records[1]
    assert second.abstract == ""
    assert second.has_gold is False
    assert normalized.gold_records == [first]


def test_missing_required_field_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    del payload["records"][0]["title"]

    with pytest.raises(ContractError, match="missing required field"):
        validate_and_normalize(payload)


def test_empty_id_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["records"][0]["id"] = "   "

    with pytest.raises(ContractError, match="non-empty"):
        validate_and_normalize(payload)


def test_bad_gold_label_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["records"][0]["gold_label"] = 2

    with pytest.raises(ContractError, match="gold_label"):
        validate_and_normalize(payload)


def test_duplicate_id_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["records"][1]["id"] = "rec-1"

    with pytest.raises(ContractError, match="duplicate record id"):
        validate_and_normalize(payload)


def test_unsupported_major_version_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["schema_version"] = "2.0"

    with pytest.raises(ContractError, match="unsupported schema_version"):
        validate_and_normalize(payload)


def test_non_list_records_raises() -> None:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["records"] = "not-a-list"

    with pytest.raises(ContractError, match="non-empty list"):
        validate_and_normalize(payload)
