"""Tests for attest.contracts.validation_record."""

from __future__ import annotations

import json

from attest.contracts.input import ExternalId, Record
from attest.contracts.validation_record import Config, build
from attest.prefilter.framework import Prefilter, require_nonempty


def test_build_seeds_prisma_from_prefilter_outcome() -> None:
    records = [
        Record(id="rec-a", title="t", abstract="a", track=1, ids=[ExternalId("doi", "x")]),
        Record(id="rec-b", title="t", abstract="a", track=1, ids=[ExternalId("doi", "x")]),
        Record(id="rec-c", title="t", abstract="", track=1),
    ]
    outcome = Prefilter(rules=[require_nonempty("abstract")]).run(records)

    record = build(
        ensemble_config_id="cfg-1",
        epoch="epoch-1",
        config=Config(vendors=["a", "b"], x=2),
        prefilter_prisma=outcome.prisma,
    )

    assert record.prisma.identified == 3
    assert record.prisma.duplicates_removed == 1
    assert record.prisma.after_dedup == 2
    assert record.prisma.prefilter_excluded == 1
    assert record.prisma.screened == 1
    assert record.prisma.screen_excluded == 0
    assert record.prisma.included == 0


def test_to_json_contains_point_and_floor_keys() -> None:
    record = build(
        ensemble_config_id="cfg-1",
        epoch="epoch-1",
        config=Config(vendors=["a"], x=1),
    )

    payload = json.loads(record.to_json())

    assert "point" in payload["recall"]
    assert "floor" in payload["recall"]
    assert payload["schema_version"] == "1.0"
    assert payload["ensemble_config_id"] == "cfg-1"
