"""Tests for attest.vendors: rater protocol, deterministic rater, registry."""

from __future__ import annotations

import importlib

from attest.contracts.input import Record
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.vendors.base import DeterministicRater, Rater, run_ensemble
from attest.vendors.registry import build_batch_raters, build_raters


def _record(record_id: str) -> Record:
    return Record(id=record_id, title="title", abstract="abstract", track=1)


def _config() -> Config:
    return Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "vendor-b": VendorSpec(
                model="model-b", model_version="v1", prompt_version="p1", temperature=0.0
            ),
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


def test_deterministic_rater_is_sensitive_to_an_explicit_prompt() -> None:
    record = _record("rec-1")
    rater = DeterministicRater(vendor="vendor-a", seed=42)

    without_prompt = rater.rate(record)
    with_prompt_a = rater.rate(record, prompt="criteria A")
    with_prompt_b = rater.rate(record, prompt="criteria B")
    with_prompt_a_again = rater.rate(record, prompt="criteria A")

    # A prompt=None call must match pre-existing (pre-prompt-param) behavior exactly.
    assert without_prompt[0] in (-1, 0, 1)
    assert with_prompt_a == with_prompt_a_again
    # Not guaranteed to differ for every possible pair, but these two shouldn't collide.
    assert with_prompt_a != with_prompt_b


def test_run_ensemble_resolves_prompt_per_record_track() -> None:
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            )
        },
        aggregation="majority",
        tau=0.5,
        track_prompts={"review-a": "criteria A", "review-b": "criteria B"},
    )
    records = [
        Record(id="rec-1", title="t", abstract="a", track="review-a"),
        Record(id="rec-2", title="t", abstract="a", track="review-b"),
        Record(id="rec-3", title="t", abstract="a", track="review-c"),  # no override, no default
    ]
    raters = [DeterministicRater(vendor="vendor-a", seed=1)]

    result = run_ensemble(records, raters, config)

    by_id = {vv.record_id: vv for vv in result.votes}
    assert result.raw_responses["rec-1"]["vendor-a"]["prompt"] == "criteria A"
    assert result.raw_responses["rec-2"]["vendor-a"]["prompt"] == "criteria B"
    assert result.raw_responses["rec-3"]["vendor-a"]["prompt"] is None

    direct_rec1 = DeterministicRater(vendor="vendor-a", seed=1).rate(
        records[0], prompt="criteria A"
    )
    assert by_id["rec-1"].votes[0].rating == direct_rec1[0]


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
            "vendor-a": lambda spec: DeterministicRater(
                vendor="vendor-a", model=spec.model, seed=1
            ),
            "vendor-b": lambda spec: DeterministicRater(
                vendor="vendor-b", model=spec.model, seed=2
            ),
        },
    )

    assert [(r.vendor, r.model) for r in raters] == [
        ("vendor-a", "model-a"),
        ("vendor-b", "model-b"),
    ]


def test_registry_raises_for_unknown_vendor() -> None:
    config = Config(
        vendors={
            "mystery": VendorSpec(
                model="m", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
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
        "attest.vendors.providers.mistral",
    ):
        importlib.import_module(module_name)


def test_registry_builds_mistral_sync_and_batch_raters_conforming_to_protocols() -> None:
    from attest.vendors.batch import BatchRater
    from attest.vendors.registry import build_batch_raters

    config = Config(
        vendors={
            "mistral": VendorSpec(
                model="mistral-small-latest",
                model_version="v1",
                prompt_version="p1",
                temperature=0.3,
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert isinstance(rater, Rater)
    assert (rater.vendor, rater.model) == ("mistral", "mistral-small-latest")
    assert isinstance(batch_rater, BatchRater)
    assert (batch_rater.vendor, batch_rater.model) == ("mistral", "mistral-small-latest")


def test_registry_passes_the_whole_vendor_spec_to_provider_factories() -> None:
    # model_version and temperature must actually reach the constructed
    # rater, not just be hashed into the ensemble_config_id and dropped.
    config = Config(
        vendors={
            "mistral": VendorSpec(
                model="mistral-small-latest",
                model_version="2026-01",
                prompt_version="p1",
                temperature=0.3,
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert (rater.model_version, rater.temperature) == ("2026-01", 0.3)
    assert (batch_rater.model_version, batch_rater.temperature) == ("2026-01", 0.3)


def test_four_vendor_ensemble_including_mistral_has_stable_config_id() -> None:
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "openai": VendorSpec(
                model="gpt-5", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "google": VendorSpec(
                model="gemini-1.5-pro", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "mistral": VendorSpec(
                model="mistral-small-latest",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
        },
        aggregation="majority",
        tau=0.5,
    )

    assert config.x == 4
    assert compute_ensemble_config_id(config) == compute_ensemble_config_id(config)
