"""Tests for attest.vendors: rater protocol, deterministic rater, registry."""

from __future__ import annotations

import importlib

from attest.contracts.input import Record
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.vendors.base import DeterministicRater, run_ensemble
from attest.vendors.registry import build_raters


def _record(record_id: str) -> Record:
    return Record(id=record_id, title="title", abstract="abstract", track=1)


def _config() -> Config:
    return Config(
        vendors={
            "vendor-a": VendorSpec(model="model-a", model_version="v1", prompt_version="p1"),
            "vendor-b": VendorSpec(model="model-b", model_version="v1", prompt_version="p1"),
        },
        aggregation="majority",
        tau=0.5,
    )


def test_deterministic_rater_reproducible_for_fixed_seed() -> None:
    record = _record("rec-1")
    rater_a = DeterministicRater(vendor="vendor-a", seed=42)
    rater_b = DeterministicRater(vendor="vendor-a", seed=42)

    ordinal_a, raw_a = rater_a.rate(record)
    ordinal_b, raw_b = rater_b.rate(record)

    assert ordinal_a == ordinal_b
    assert ordinal_a in (-1, 0, 1)
    assert raw_a == raw_b


def test_deterministic_rater_differs_across_seeds_or_vendors() -> None:
    record = _record("rec-1")
    ratings = {DeterministicRater(vendor="vendor-a", seed=s).rate(record)[0] for s in range(20)}

    # Not every seed need differ, but 20 seeds should not all collapse to one rating.
    assert len(ratings) > 1


def test_run_ensemble_produces_one_vote_per_rater_per_record_with_config_stamp() -> None:
    config = _config()
    expected_config_id = compute_ensemble_config_id(config)
    records = [_record("rec-1"), _record("rec-2"), _record("rec-3")]
    raters = [
        DeterministicRater(vendor="vendor-a", seed=1),
        DeterministicRater(vendor="vendor-b", seed=2),
    ]

    result = run_ensemble(records, raters, config)

    assert [v.record_id for v in result.votes] == ["rec-1", "rec-2", "rec-3"]
    for vote_vector in result.votes:
        assert vote_vector.ensemble_config_id == expected_config_id
        assert [vote.vendor for vote in vote_vector.votes] == ["vendor-a", "vendor-b"]
        assert len(vote_vector.votes) == len(raters)

    assert set(result.raw_responses) == {"rec-1", "rec-2", "rec-3"}
    for per_vendor in result.raw_responses.values():
        assert set(per_vendor) == {"vendor-a", "vendor-b"}


def test_registry_builds_raters_from_config() -> None:
    config = _config()

    raters = build_raters(
        config,
        factories={
            "vendor-a": lambda model: DeterministicRater(vendor="vendor-a", model=model, seed=1),
            "vendor-b": lambda model: DeterministicRater(vendor="vendor-b", model=model, seed=2),
        },
    )

    assert [(r.vendor, r.model) for r in raters] == [
        ("vendor-a", "model-a"),
        ("vendor-b", "model-b"),
    ]


def test_registry_raises_for_unknown_vendor() -> None:
    config = Config(
        vendors={"mystery": VendorSpec(model="m", model_version="v1", prompt_version="p1")}
    )

    try:
        build_raters(config)
    except KeyError as exc:
        assert "mystery" in str(exc)
    else:
        raise AssertionError("expected KeyError for unregistered vendor")


def test_import_attest_with_no_extras_installed_still_succeeds() -> None:
    for module_name in (
        "attest",
        "attest.vendors.base",
        "attest.vendors.registry",
        "attest.vendors.providers.anthropic",
        "attest.vendors.providers.openai",
        "attest.vendors.providers.google",
        "attest.vendors.providers.openmodel",
    ):
        importlib.import_module(module_name)
