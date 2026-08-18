"""Tests for attest.contracts.validation_record."""

from __future__ import annotations

import json

from attest.contracts.input import ExternalId, Record
from attest.contracts.validation_record import (
    Config,
    ErrorCorrelation,
    PairwiseFnCorrelation,
    build,
)
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
    assert payload["schema_version"] == "1.3"
    assert payload["ensemble_config_id"] == "cfg-1"
    assert payload["config"]["zero_policy"] == "escalate"


def test_error_correlation_to_dict_reports_undefined_pair_with_joint_counts() -> None:
    error_correlation = ErrorCorrelation(
        pairwise_fn_on_relevant={
            "a|b": PairwiseFnCorrelation(
                correlation=None, n=3, both=2, only_a=1, only_b=0, neither=0
            )
        }
    )

    payload = error_correlation.to_dict()

    # The pair survives in the output even though correlation is undefined --
    # this is exactly what "no longer silently dropped" means.
    assert payload["pairwise_fn_on_relevant"]["a|b"] == {
        "correlation": None,
        "n": 3,
        "both": 2,
        "only_a": 1,
        "only_b": 0,
        "neither": 0,
    }
