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


def test_deterministic_rater_omits_logprobs_by_default() -> None:
    record = _record("rec-1")
    rater = DeterministicRater(vendor="openai", seed=42)

    _ordinal, raw = rater.rate(record)

    assert "logprobs" not in raw


def test_deterministic_rater_request_logprobs_is_deterministic_and_openai_shaped() -> None:
    from attest.ensemble.confidence import vote_confidence

    record = _record("rec-1")
    rater_a = DeterministicRater(vendor="openai", seed=42, request_logprobs=True)
    rater_b = DeterministicRater(vendor="openai", seed=42, request_logprobs=True)

    ordinal, raw_a = rater_a.rate(record)
    _ordinal_b, raw_b = rater_b.rate(record)

    assert raw_a == raw_b
    logprob = raw_a["logprobs"]["content"][0]["logprob"]
    assert -3.0 <= logprob <= 0.0
    assert raw_a["logprobs"]["content"][0]["token"] == str(ordinal)
    # Not a real vendor response, but shaped so attest.ensemble.confidence's
    # shared OpenAI-compatible extractor parses it exactly as a live
    # openai/mistral/fireworks/together response would.
    probability = vote_confidence("openai", raw_a)
    assert probability is not None
    assert 0.0 < probability <= 1.0


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
        "attest.vendors.providers.fireworks",
        "attest.vendors.providers.together",
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


def test_registry_builds_fireworks_sync_and_batch_raters_conforming_to_protocols() -> None:
    from attest.vendors.batch import BatchRater
    from attest.vendors.registry import build_batch_raters

    config = Config(
        vendors={
            "fireworks": VendorSpec(
                model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                model_version="v1",
                prompt_version="p1",
                temperature=0.3,
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert isinstance(rater, Rater)
    assert (rater.vendor, rater.model) == (
        "fireworks",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    )
    assert isinstance(batch_rater, BatchRater)
    assert (batch_rater.vendor, batch_rater.model) == (
        "fireworks",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    )


def test_registry_builds_together_sync_and_batch_raters_conforming_to_protocols() -> None:
    from attest.vendors.batch import BatchRater
    from attest.vendors.registry import build_batch_raters

    config = Config(
        vendors={
            "together": VendorSpec(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_version="v1",
                prompt_version="p1",
                temperature=0.3,
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert isinstance(rater, Rater)
    assert (rater.vendor, rater.model) == ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert isinstance(batch_rater, BatchRater)
    assert (batch_rater.vendor, batch_rater.model) == (
        "together",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )


def test_registry_passes_the_whole_vendor_spec_to_fireworks_and_together_factories() -> None:
    # model_version and temperature must actually reach the constructed
    # rater, not just be hashed into the ensemble_config_id and dropped.
    config = Config(
        vendors={
            "fireworks": VendorSpec(
                model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                model_version="2026-01",
                prompt_version="p1",
                temperature=0.3,
            ),
            "together": VendorSpec(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_version="2026-01",
                prompt_version="p1",
                temperature=0.4,
            ),
        }
    )

    raters = {r.vendor: r for r in build_raters(config)}
    batch_raters = {r.vendor: r for r in build_batch_raters(config)}

    assert (raters["fireworks"].model_version, raters["fireworks"].temperature) == ("2026-01", 0.3)
    assert (raters["together"].model_version, raters["together"].temperature) == ("2026-01", 0.4)
    assert (
        batch_raters["fireworks"].model_version,
        batch_raters["fireworks"].temperature,
    ) == ("2026-01", 0.3)
    assert (
        batch_raters["together"].model_version,
        batch_raters["together"].temperature,
    ) == ("2026-01", 0.4)


def test_build_raters_forwards_request_logprobs_to_supporting_vendors() -> None:
    config = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-5", model_version="v1", prompt_version="p1", temperature=0.0
            ),
            "mistral": VendorSpec(
                model="mistral-small-latest",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
            "google": VendorSpec(
                model="gemini-1.5-pro", model_version="v1", prompt_version="p1", temperature=0.0
            ),
        }
    )

    raters = build_raters(config, request_logprobs=True)
    batch_raters = build_batch_raters(config, request_logprobs=True)

    assert all(r.request_logprobs is True for r in raters)  # type: ignore[attr-defined]
    assert all(r.request_logprobs is True for r in batch_raters)  # type: ignore[attr-defined]


def test_build_raters_request_logprobs_defaults_to_false() -> None:
    config = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-5", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.request_logprobs is False  # type: ignore[attr-defined]


def test_build_raters_accepts_request_logprobs_for_anthropic_without_adding_a_field() -> None:
    # Anthropic's factory must tolerate request_logprobs=True (signature
    # parity with the other factories) without crashing, but the Messages
    # API has no logprobs equivalent, so AnthropicRater/AnthropicBatchRater
    # must not gain a field for it -- a field that can never be applied is
    # exactly the decorative-config-field pattern this codebase avoids.
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config, request_logprobs=True)
    [batch_rater] = build_batch_raters(config, request_logprobs=True)

    assert not hasattr(rater, "request_logprobs")
    assert not hasattr(batch_rater, "request_logprobs")


def test_build_raters_forwards_request_logprobs_to_fireworks_and_together() -> None:
    # Both are OpenAI-compatible on the sync path, and Together's batch API
    # mirrors OpenAI's, so both get the same request_logprobs treatment as
    # openai/mistral/google -- see docs/logprob_support.md.
    config = Config(
        vendors={
            "fireworks": VendorSpec(
                model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
            "together": VendorSpec(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
        }
    )

    raters = build_raters(config, request_logprobs=True)

    assert all(r.request_logprobs is True for r in raters)  # type: ignore[attr-defined]


def test_build_batch_raters_forwards_request_logprobs_to_together_not_fireworks() -> None:
    # TogetherBatchRater mirrors OpenAI's batch shape and gets the field.
    # FireworksBatchRater's own row schema is still unconfirmed (see its
    # module docstring), so it deliberately has no request_logprobs field
    # yet -- request_logprobs=True must not crash building it, just be a
    # no-op, the same signature-parity pattern Anthropic's factories use.
    config = Config(
        vendors={
            "fireworks": VendorSpec(
                model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
            "together": VendorSpec(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
            ),
        }
    )

    batch_raters = {r.vendor: r for r in build_batch_raters(config, request_logprobs=True)}

    assert batch_raters["together"].request_logprobs is True  # type: ignore[attr-defined]
    assert not hasattr(batch_raters["fireworks"], "request_logprobs")


def test_anthropic_rater_constructor_rejects_request_logprobs() -> None:
    from attest.vendors.providers.anthropic import AnthropicBatchRater, AnthropicRater

    common = {"model": "claude-sonnet-5", "model_version": "v1", "temperature": 0.0}
    try:
        AnthropicRater(**common, request_logprobs=True)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("AnthropicRater unexpectedly accepted request_logprobs")

    try:
        AnthropicBatchRater(**common, request_logprobs=True)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("AnthropicBatchRater unexpectedly accepted request_logprobs")


def test_fireworks_batch_rater_constructor_rejects_request_logprobs() -> None:
    # Locks in that FireworksBatchRater truly has no request_logprobs field
    # -- not just that the registry happens not to pass one -- since its
    # batch row schema is itself unconfirmed (see its module docstring).
    from attest.vendors.providers.fireworks import FireworksBatchRater

    try:
        FireworksBatchRater(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            model_version="v1",
            temperature=0.0,
            request_logprobs=True,  # type: ignore[call-arg]
        )
    except TypeError:
        pass
    else:
        raise AssertionError("FireworksBatchRater unexpectedly accepted request_logprobs")


def test_build_raters_with_custom_factories_stays_backward_compatible() -> None:
    # A custom factories override taking only (spec) -- as every factory did
    # before request_logprobs existed -- must keep working at the default
    # (request_logprobs left False), since build_raters only forwards the
    # kwarg when the caller explicitly asks for it.
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(
        config,
        factories={
            "vendor-a": lambda spec: DeterministicRater(vendor="vendor-a", model=spec.model)
        },
    )

    assert rater.vendor == "vendor-a"


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
