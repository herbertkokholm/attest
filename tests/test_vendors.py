"""Tests for attest.vendors: rater protocol, deterministic rater, registry."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import pytest

from attest.contracts.input import Record
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.vendors.base import (
    DeterministicRater,
    Rater,
    SingleRecordOnlyRateMany,
    VendorResponseError,
    chunk_records,
    parse_batch_response,
    run_ensemble,
)
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
    expected_token = {-1: "E", 0: "U", 1: "I"}[ordinal]
    assert raw_a["logprobs"]["content"][0]["token"] == expected_token
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


def test_build_raters_forwards_reasoning_effort_to_openai() -> None:
    config = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-5.6-terra",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                reasoning_effort="none",
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert rater.reasoning_effort == "none"  # type: ignore[attr-defined]
    assert batch_rater.reasoning_effort == "none"  # type: ignore[attr-defined]


def test_build_raters_reasoning_effort_defaults_to_none() -> None:
    config = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-5", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.reasoning_effort is None  # type: ignore[attr-defined]


def test_build_raters_forwards_send_temperature_to_anthropic() -> None:
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                send_temperature=False,
            )
        }
    )

    [rater] = build_raters(config)
    [batch_rater] = build_batch_raters(config)

    assert rater.send_temperature is False  # type: ignore[attr-defined]
    assert batch_rater.send_temperature is False  # type: ignore[attr-defined]


def test_build_raters_send_temperature_defaults_to_true() -> None:
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-4-6", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.send_temperature is True  # type: ignore[attr-defined]


def test_build_raters_forwards_base_url_to_openmodel() -> None:
    config = Config(
        vendors={
            "openmodel": VendorSpec(
                model="qwen3.5-397b",
                model_version="qwen3.5-397b",
                prompt_version="p1",
                temperature=0.0,
                base_url="https://inference.alexandra.dk/v1",
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.base_url == "https://inference.alexandra.dk/v1"  # type: ignore[attr-defined]


def test_build_raters_openmodel_base_url_defaults_to_local_server() -> None:
    from attest.vendors.providers.openmodel import DEFAULT_BASE_URL

    config = Config(
        vendors={
            "openmodel": VendorSpec(
                model="local-model", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.base_url == DEFAULT_BASE_URL  # type: ignore[attr-defined]


def test_build_raters_resolves_api_key_env_for_openmodel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALEX_API_KEY", "secret-value")
    config = Config(
        vendors={
            "openmodel": VendorSpec(
                model="qwen3.5-397b",
                model_version="qwen3.5-397b",
                prompt_version="p1",
                temperature=0.0,
                api_key_env="ALEX_API_KEY",
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.api_key == "secret-value"  # type: ignore[attr-defined]


def test_build_raters_openmodel_api_key_defaults_to_none_without_api_key_env() -> None:
    config = Config(
        vendors={
            "openmodel": VendorSpec(
                model="local-model", model_version="v1", prompt_version="p1", temperature=0.0
            )
        }
    )

    [rater] = build_raters(config)

    assert rater.api_key is None  # type: ignore[attr-defined]


def test_build_raters_rejects_base_url_on_a_vendor_that_does_not_consume_it() -> None:
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                base_url="https://inference.alexandra.dk/v1",
            )
        }
    )

    with pytest.raises(ValueError, match="anthropic.*base_url"):
        build_raters(config)


def test_build_raters_rejects_api_key_env_on_a_vendor_that_does_not_consume_it() -> None:
    config = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-5",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                api_key_env="ALEX_API_KEY",
            )
        }
    )

    with pytest.raises(ValueError, match="openai.*api_key_env"):
        build_raters(config)


def test_build_batch_raters_rejects_base_url_on_any_vendor() -> None:
    config = Config(
        vendors={
            "together": VendorSpec(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                base_url="https://inference.alexandra.dk/v1",
            )
        }
    )

    with pytest.raises(ValueError, match="together.*base_url"):
        build_batch_raters(config)


def test_build_raters_allows_base_url_via_a_custom_factories_override() -> None:
    # A caller-supplied factories override takes on full responsibility for
    # whatever VendorSpec fields it reads -- the guard must not second-guess it.
    config = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5",
                model_version="v1",
                prompt_version="p1",
                temperature=0.0,
                base_url="https://inference.alexandra.dk/v1",
            )
        }
    )

    def _custom_factory(spec: VendorSpec, **_: Any) -> DeterministicRater:
        return DeterministicRater(vendor="anthropic", seed=1)

    [rater] = build_raters(config, factories={"anthropic": _custom_factory})

    assert isinstance(rater, DeterministicRater)


def test_openai_batch_rater_request_line_omits_reasoning_effort_by_default() -> None:
    from attest.vendors.providers.openai import OpenAIBatchRater

    rater = OpenAIBatchRater(model="gpt-4o", model_version="v1", temperature=0.0)

    line = rater._request_line(_record("r1"), "item-0", "criteria")

    assert "reasoning_effort" not in line["body"]


def test_openai_batch_rater_request_line_includes_reasoning_effort_when_set() -> None:
    from attest.vendors.providers.openai import OpenAIBatchRater

    rater = OpenAIBatchRater(
        model="gpt-5.6-terra", model_version="v1", temperature=0.0, reasoning_effort="none"
    )

    line = rater._request_line(_record("r1"), "item-0", "criteria")
    batch_line = rater._batch_request_line([_record("r1"), _record("r2")], "item-0", "criteria")

    assert line["body"]["reasoning_effort"] == "none"
    assert line["body"]["temperature"] == 0.0
    assert batch_line["body"]["reasoning_effort"] == "none"


def test_anthropic_batch_rater_request_omits_temperature_when_disabled() -> None:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    rater = AnthropicBatchRater(
        model="claude-sonnet-5", model_version="v1", temperature=0.0, send_temperature=False
    )

    request = rater._request(_record("r1"), "item-0", "criteria")
    batch_request = rater._batch_request([_record("r1"), _record("r2")], "item-0", "criteria")

    assert "temperature" not in request["params"]
    assert "temperature" not in batch_request["params"]


def test_anthropic_batch_rater_request_includes_temperature_by_default() -> None:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    rater = AnthropicBatchRater(model="claude-sonnet-4-6", model_version="v1", temperature=0.0)

    request = rater._request(_record("r1"), "item-0", "criteria")

    assert request["params"]["temperature"] == 0.0


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


# --- google provider: _serialize_logprobs's best-effort shape handling ---------
#
# `_serialize_logprobs` (attest.vendors.providers.google) probes for
# `to_dict`/`model_dump` and falls back to `str()`. `google-genai`'s response
# objects are uniformly pydantic (`model_dump` always applies), but the probe
# stays duck-typed and defensive rather than assuming that shape outright. It
# is a pure function, exercisable offline with plain stub objects, without
# either the `google` extra or a live API call -- unlike the rest of the
# provider's `rate`/`fetch` methods, which do require the SDK and are covered
# only by `tools/vendor_logprob_probe.py`'s manual, live verification (see
# `docs/logprob_support.md`).


def test_serialize_logprobs_uses_to_dict_when_available() -> None:
    from attest.vendors.providers.google import _serialize_logprobs

    class _WithToDict:
        def to_dict(self) -> dict[str, str]:
            return {"shape": "to_dict"}

    assert _serialize_logprobs(_WithToDict()) == {"shape": "to_dict"}


def test_serialize_logprobs_uses_model_dump_when_to_dict_is_absent() -> None:
    from attest.vendors.providers.google import _serialize_logprobs

    class _WithModelDump:
        def model_dump(self) -> dict[str, str]:
            return {"shape": "model_dump"}

    assert _serialize_logprobs(_WithModelDump()) == {"shape": "model_dump"}


def test_serialize_logprobs_prefers_to_dict_over_model_dump() -> None:
    from attest.vendors.providers.google import _serialize_logprobs

    class _WithBoth:
        def to_dict(self) -> dict[str, str]:
            return {"shape": "to_dict"}

        def model_dump(self) -> dict[str, str]:
            return {"shape": "model_dump"}

    assert _serialize_logprobs(_WithBoth()) == {"shape": "to_dict"}


def test_serialize_logprobs_falls_back_to_str_when_neither_method_exists() -> None:
    from attest.vendors.providers.google import _serialize_logprobs

    class _Bare:
        def __str__(self) -> str:
            return "bare-repr"

    assert _serialize_logprobs(_Bare()) == "bare-repr"


def test_serialize_logprobs_ignores_a_non_callable_to_dict_attribute() -> None:
    from attest.vendors.providers.google import _serialize_logprobs

    class _NonCallableAttr:
        to_dict = "not a method"

        def __str__(self) -> str:
            return "fallback"

    assert _serialize_logprobs(_NonCallableAttr()) == "fallback"


# --- batch_size request packing: chunking, rate_many, and failure policy ---------


class _CountingRater:
    """A `Rater` that records every request it was asked to make, without a network call."""

    def __init__(self, vendor: str = "vendor-a", model: str = "model-a") -> None:
        self.vendor = vendor
        self.model = model
        self.requests: list[list[str]] = []  # one entry per request; multi-id => rate_many

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        self.requests.append([record.id])
        return (1, {"record_id": record.id, "prompt": prompt})

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        self.requests.append([r.id for r in records])
        return [(1, {"record_id": r.id, "prompt": prompt}) for r in records]


def test_run_ensemble_issues_ceil_division_requests_per_prompt_group() -> None:
    # One prompt group of 5 records at batch_size=2: chunks of [2, 2, 1] --
    # ceil(5/2) == 3 requests, the last one undersized (not an error).
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            )
        },
        aggregation="majority",
        tau=0.5,
        batch_size=2,
    )
    records = [_record(f"rec-{i}") for i in range(5)]
    rater = _CountingRater()

    result = run_ensemble(records, [rater], config)

    assert [len(chunk) for chunk in rater.requests] == [2, 2, 1]
    assert sum(len(chunk) for chunk in rater.requests) == 5
    assert [vv.record_id for vv in result.votes] == [r.id for r in records]


def test_run_ensemble_accepts_batch_size_larger_than_the_group() -> None:
    # batch_size larger than the whole prompt group collapses to one request
    # covering every record in it -- not an error.
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            )
        },
        aggregation="majority",
        tau=0.5,
        batch_size=100,
    )
    records = [_record(f"rec-{i}") for i in range(3)]
    rater = _CountingRater()

    run_ensemble(records, [rater], config)

    assert [len(chunk) for chunk in rater.requests] == [3]


def test_run_ensemble_never_co_batches_records_on_different_resolved_prompts() -> None:
    config = Config(
        vendors={
            "vendor-a": VendorSpec(
                model="model-a", model_version="v1", prompt_version="p1", temperature=0.0
            )
        },
        aggregation="majority",
        tau=0.5,
        batch_size=10,
        track_prompts={"review-a": "criteria A", "review-b": "criteria B"},
    )
    records = [
        Record(id="rec-1", title="t", abstract="a", track="review-a"),
        Record(id="rec-2", title="t", abstract="a", track="review-b"),
        Record(id="rec-3", title="t", abstract="a", track="review-a"),
        Record(id="rec-4", title="t", abstract="a", track="review-b"),
    ]
    rater = _CountingRater()

    run_ensemble(records, [rater], config)

    ids_by_chunk = {frozenset(chunk) for chunk in rater.requests}
    assert ids_by_chunk == {frozenset({"rec-1", "rec-3"}), frozenset({"rec-2", "rec-4"})}


def test_chunk_records_splits_final_undersized_group() -> None:
    records = [_record(f"rec-{i}") for i in range(5)]

    chunks = chunk_records(records, lambda r: None, batch_size=2)

    assert [[r.id for r in chunk] for chunk in chunks] == [
        ["rec-0", "rec-1"],
        ["rec-2", "rec-3"],
        ["rec-4"],
    ]


# --- DeterministicRater.rate_many: chunk-peer sensitivity -----------------------


def test_deterministic_rater_rate_many_matches_rate_for_a_singleton_chunk() -> None:
    record = _record("rec-1")
    rater = DeterministicRater(vendor="vendor-a", seed=42)

    [alone] = rater.rate_many([record])
    direct = rater.rate(record)

    assert alone == direct


def test_deterministic_rater_rate_many_is_sensitive_to_chunk_peers() -> None:
    rater = DeterministicRater(vendor="vendor-a", seed=42)
    a, b, c = _record("rec-a"), _record("rec-b"), _record("rec-c")

    with_bc = rater.rate_many([a, b, c])[0]
    alone = rater.rate(a)

    assert with_bc != alone

    # Peer set (not just peer count) matters: swapping one peer for another
    # changes the digest even though the chunk size is unchanged.
    d = _record("rec-d")
    with_bd = rater.rate_many([a, b, d])[0]
    assert with_bc != with_bd

    # But peer order within the chunk does not: peers are sorted before
    # entering the digest, so the two permutations below rate 'a' identically.
    with_bc_reordered = rater.rate_many([a, c, b])[0]
    assert with_bc == with_bc_reordered


def test_deterministic_rater_rate_many_returns_one_result_per_record_in_order() -> None:
    rater = DeterministicRater(vendor="vendor-a", seed=1)
    records = [_record(f"rec-{i}") for i in range(4)]

    results = rater.rate_many(records)

    assert len(results) == len(records)
    for record, (ordinal, raw) in zip(records, results, strict=True):
        assert raw["record_id"] == record.id
        assert ordinal in (-1, 0, 1)


def test_deterministic_rater_rate_regression_fixture_pins_batch_size_one_bytes() -> None:
    # Pins DeterministicRater.rate()'s exact digest/ordinal/raw_response for
    # a fixed (seed, vendor, model, record_id) -- the batch_size == 1 path,
    # untouched by rate_many's chunk-peer digest suffix (peers=() is a no-op
    # -- see DeterministicRater._rate_one). A change to this value means the
    # batch_size == 1 path stopped being byte-identical to before batch_size
    # packing existed, which must never happen (see Config.batch_size).
    rater = DeterministicRater(vendor="openai", model="gpt-5", seed=7)
    record = Record(id="rec-42", title="t", abstract="a", track=1)

    ordinal, raw = rater.rate(record)

    assert ordinal == -1
    assert raw == {
        "vendor": "openai",
        "model": "gpt-5",
        "seed": 7,
        "record_id": "rec-42",
        "digest": "7591e207f971da8cd28604cbccb611cc85709e217b22b5d2211b6747d752017f",
        "prompt": None,
    }


# --- SingleRecordOnlyRateMany mixin: reserved for a future not-yet-converted provider ---


def test_single_record_only_rate_many_supports_a_singleton_chunk() -> None:
    class _StubRater(SingleRecordOnlyRateMany):
        vendor = "stub"
        model = "stub-model"

        def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
            return (1, {"record_id": record.id, "prompt": prompt})

    rater = _StubRater()
    record = _record("rec-1")

    [result] = rater.rate_many([record], prompt="criteria")

    assert result == rater.rate(record, prompt="criteria")


@pytest.mark.parametrize(
    "provider_module,class_name",
    [
        ("attest.vendors.providers.mistral", "MistralRater"),
        ("attest.vendors.providers.google", "GoogleRater"),
        ("attest.vendors.providers.fireworks", "FireworksRater"),
        ("attest.vendors.providers.together", "TogetherRater"),
        ("attest.vendors.providers.openmodel", "OpenModelRater"),
        ("attest.vendors.providers.openai", "OpenAIRater"),
        ("attest.vendors.providers.anthropic", "AnthropicRater"),
    ],
)
def test_every_rater_provides_a_real_rate_many_not_the_single_record_mixin(
    provider_module: str, class_name: str
) -> None:
    # Every sync Rater provider has been converted to true multi-record
    # packing (see Config.batch_size) -- none of them should still be using
    # SingleRecordOnlyRateMany, which exists only for a future not-yet-
    # converted provider.
    module = importlib.import_module(provider_module)
    rater_cls = getattr(module, class_name)

    assert not issubclass(rater_cls, SingleRecordOnlyRateMany)
    assert "rate_many" in rater_cls.__dict__


@pytest.mark.parametrize(
    "provider_module,class_name",
    [
        ("attest.vendors.providers.mistral", "MistralBatchRater"),
        ("attest.vendors.providers.google", "GoogleBatchRater"),
        ("attest.vendors.providers.fireworks", "FireworksBatchRater"),
        ("attest.vendors.providers.together", "TogetherBatchRater"),
        ("attest.vendors.providers.openai", "OpenAIBatchRater"),
        ("attest.vendors.providers.anthropic", "AnthropicBatchRater"),
    ],
)
def test_every_batch_rater_submit_accepts_a_batch_size_keyword(
    provider_module: str, class_name: str
) -> None:
    import inspect

    module = importlib.import_module(provider_module)
    rater_cls = getattr(module, class_name)
    signature = inspect.signature(rater_cls.submit)
    assert "batch_size" in signature.parameters
    assert signature.parameters["batch_size"].default == 1


# --- parse_batch_response: id-keyed parsing, failure policy, robustness ---------


def test_parse_batch_response_parses_every_requested_id() -> None:
    text = '{"1": "I", "2": "E", "3": "U"}'

    ratings = parse_batch_response(text, ["1", "2", "3"])

    assert ratings == {"1": 1, "2": -1, "3": 0}


def test_parse_batch_response_is_order_independent() -> None:
    text = '{"3": "U", "1": "I", "2": "E"}'

    ratings = parse_batch_response(text, ["1", "2", "3"])

    assert ratings == {"1": 1, "2": -1, "3": 0}


def test_parse_batch_response_ignores_extra_and_hallucinated_ids() -> None:
    text = '{"1": "I", "2": "E", "hallucinated-id": "I"}'

    ratings = parse_batch_response(text, ["1", "2"])

    assert ratings == {"1": 1, "2": -1}
    assert "hallucinated-id" not in ratings


def test_parse_batch_response_raises_for_a_missing_id() -> None:
    text = '{"1": "I"}'

    try:
        parse_batch_response(text, ["1", "2"])
    except VendorResponseError as exc:
        assert "2" in str(exc)
    else:
        raise AssertionError("expected VendorResponseError for missing id '2'")


def test_parse_batch_response_raises_for_an_unparseable_value() -> None:
    text = '{"1": "not-a-rating"}'

    try:
        parse_batch_response(text, ["1"])
    except VendorResponseError:
        pass
    else:
        raise AssertionError("expected VendorResponseError for an unparseable rating")


def test_parse_batch_response_raises_for_invalid_json() -> None:
    try:
        parse_batch_response("not json", ["1"])
    except VendorResponseError:
        pass
    else:
        raise AssertionError("expected VendorResponseError for invalid JSON")


def test_parse_batch_response_raises_for_a_non_object_json_value() -> None:
    try:
        parse_batch_response("[1, 2, 3]", ["1"])
    except VendorResponseError:
        pass
    else:
        raise AssertionError("expected VendorResponseError for a JSON array")


def test_parse_batch_response_accepts_integer_ratings() -> None:
    text = '{"1": 1, "2": -1, "3": 0}'

    ratings = parse_batch_response(text, ["1", "2", "3"])

    assert ratings == {"1": 1, "2": -1, "3": 0}
