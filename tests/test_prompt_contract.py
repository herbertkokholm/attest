"""Tests for Workstream A: the kernel-owned output contract.

Covers `attest.vendors.base.compose_system_prompt`, the hardened
`parse_ordinal_response`, every provider's routing of criteria through
`compose_system_prompt`, and `OUTPUT_CONTRACT_VERSION`'s effect on
`compute_ensemble_config_id`.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from attest.contracts.input import Record
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.vendors.base import (
    OUTPUT_CONTRACT,
    OUTPUT_CONTRACT_VERSION,
    SCREENING_TASK_PREAMBLE,
    DeterministicRater,
    VendorResponseError,
    compose_system_prompt,
    parse_ordinal_response,
)
from attest.vendors.batch import BatchHandle


def _record(record_id: str = "rec-1") -> Record:
    return Record(id=record_id, title="t", abstract="a", track=1)


# --- compose_system_prompt -----------------------------------------------------


def test_compose_system_prompt_falls_back_to_generic_preamble_when_criteria_is_none() -> None:
    composed = compose_system_prompt(None)

    assert composed == f"{SCREENING_TASK_PREAMBLE}\n\n{OUTPUT_CONTRACT}"


def test_compose_system_prompt_appends_contract_to_supplied_criteria() -> None:
    composed = compose_system_prompt("Include only randomized controlled trials.")

    assert composed == f"Include only randomized controlled trials.\n\n{OUTPUT_CONTRACT}"


def test_compose_system_prompt_warns_when_criteria_already_contains_the_contract() -> None:
    criteria = f"Some criteria.\n\n{OUTPUT_CONTRACT}"

    with pytest.warns(UserWarning, match="already contains the output-contract"):
        composed = compose_system_prompt(criteria)

    # The contract still gets appended once more by the composition point --
    # the warning tells the caller to fix their criteria, it does not
    # silently deduplicate for them.
    assert composed == f"{criteria}\n\n{OUTPUT_CONTRACT}"


def test_compose_system_prompt_does_not_warn_for_ordinary_criteria() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compose_system_prompt("Include only studies published after 2015.")


# --- parse_ordinal_response: strict match --------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I", 1),
        ("E", -1),
        ("U", 0),
        ("i", 1),
        ("e", -1),
        ("u", 0),
        (" I ", 1),
        ("I.", 1),
        ("(E)", -1),
    ],
)
def test_parse_ordinal_response_strict_whole_reply(text: str, expected: int) -> None:
    assert parse_ordinal_response(text) == expected


def test_parse_ordinal_response_strict_last_non_empty_line() -> None:
    text = "Reasoning: this study meets the inclusion criteria.\n\nFinal answer:\nI"

    assert parse_ordinal_response(text) == 1


def test_parse_ordinal_response_strict_last_line_avoids_false_ambiguity() -> None:
    # An earlier line mentions "I", but the strict last-line match short-circuits
    # before the fallback scan ever sees it, so this does not raise as ambiguous.
    text = "Initial leaning was I, but on further reflection:\nE"

    assert parse_ordinal_response(text) == -1


# --- parse_ordinal_response: fallback scan -------------------------------------


def test_parse_ordinal_response_falls_back_to_first_token_scan_for_prose() -> None:
    assert parse_ordinal_response("Decision: E, excluded due to wrong population.") == -1


def test_parse_ordinal_response_fallback_tolerates_repeated_equal_ratings() -> None:
    # Both mentions spell the same rating, so this is not ambiguous.
    assert parse_ordinal_response("First pass: I. Second pass: I. Both agree.") == 1


# --- parse_ordinal_response: ambiguity and failure ------------------------------


def test_parse_ordinal_response_raises_on_genuinely_ambiguous_reply() -> None:
    # Two distinct ratings appear as tokens outside of any strict match.
    text = "Leaning E but could also see an argument for I here."

    with pytest.raises(VendorResponseError, match="ambiguous"):
        parse_ordinal_response(text)


def test_parse_ordinal_response_raises_when_no_ordinal_token_found() -> None:
    with pytest.raises(VendorResponseError):
        parse_ordinal_response("Unable to reach a determination for this record.")


def test_parse_ordinal_response_criteria_echoing_a_number_does_not_misparse() -> None:
    # A criteria prompt mentioning a roman-numeral-labeled criterion the model
    # echoes back must not be silently taken as the rating when a real, distinct
    # rating is also present.
    text = "Per criterion I this fails eligibility, so the rating is E."

    with pytest.raises(VendorResponseError, match="ambiguous"):
        parse_ordinal_response(text)


# --- DeterministicRater: digest sensitivity to the composed prompt -------------


def test_deterministic_rater_digest_reflects_composed_prompt_not_raw_criteria() -> None:
    rater_a = DeterministicRater(vendor="v", seed=1)
    rater_b = DeterministicRater(vendor="v", seed=1)
    record = _record()

    # Two different criteria strings that compose to the same final message
    # (impossible in practice since compose_system_prompt is injective on
    # distinct criteria, but same criteria across two instances) must match.
    result_a = rater_a.rate(record, prompt="criteria X")
    result_b = rater_b.rate(record, prompt="criteria X")

    assert result_a == result_b


def test_deterministic_rater_rate_warns_when_prompt_contains_the_contract() -> None:
    rater = DeterministicRater(vendor="v", seed=1)

    with pytest.warns(UserWarning, match="already contains the output-contract"):
        rater.rate(_record(), prompt=f"criteria\n\n{OUTPUT_CONTRACT}")


# --- provenance.config: OUTPUT_CONTRACT_VERSION and the config hash -----------


def _base_config(**overrides: Any) -> Config:
    defaults: dict[str, Any] = {
        "vendors": {
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p", temperature=0.0),
        },
        "aggregation": "boundary_dispersion",
        "tau": 0.5,
    }
    defaults.update(overrides)
    return Config(**defaults)


def test_output_contract_version_included_even_when_no_prompt_fields_set() -> None:
    # The kernel-owned output contract is appended to every composed prompt,
    # including the no-criteria fallback (see compose_system_prompt), so its
    # version is always config-hash-sensitive -- not just for configs that
    # supply criteria.
    payload = _base_config().to_dict()

    assert payload["output_contract_version"] == OUTPUT_CONTRACT_VERSION


def test_config_hash_stable_for_default_contract_config_with_no_prompt_fields() -> None:
    # Pinned hash of the current to_dict() shape (including
    # "output_contract_version", unconditionally), computed independently of
    # compute_ensemble_config_id, so this test actually catches a regression
    # rather than just re-deriving the same formula. This pin was retired and
    # recomputed once, deliberately, when output_contract_version became
    # unconditional: a one-time id change for configs supplying no criteria,
    # since their composed prompt has always contained the contract -- only
    # the hash was previously blind to it. Retired and recomputed a second
    # time when `VendorSpec.temperature` was added, unconditionally included
    # in `VendorSpec.to_dict()` alongside `model_version`/`prompt_version`.
    # Retired and recomputed a third time when `Config.batch_size` was
    # added: unconditionally included, on par with
    # `vendors`/`aggregation`/`tau`/`x` (see `Config.batch_size`).
    # Retired and recomputed a fourth time when `Config.batch_size`'s default
    # changed from `0` (a placeholder value that was always checked against
    # the corpus size, never actually applied) to `1` (a real request-packing
    # width, applied whether or not a config sets it) -- a config built
    # before the field existed now picks up the honest one-record-per-request
    # default instead of the old placeholder.
    config = _base_config()
    payload = config.to_dict()
    expected_payload = {
        "vendors": {
            "v1": {
                "model": "m",
                "model_version": "1",
                "prompt_version": "p",
                "temperature": 0.0,
            }
        },
        "aggregation": "boundary_dispersion",
        "tau": 0.5,
        "batch_size": 1,
        "x": 1,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
    }
    pinned_hash = "58607addf7931faa3ef147488a05c023be438d3de0e29be2edc81a2a54a3dbd2"

    assert payload == expected_payload
    assert compute_ensemble_config_id(config) == pinned_hash


def test_output_contract_version_included_when_default_prompt_set() -> None:
    payload = _base_config(default_prompt="criteria").to_dict()

    assert payload["output_contract_version"] == OUTPUT_CONTRACT_VERSION


def test_output_contract_version_included_when_track_prompts_set() -> None:
    payload = _base_config(track_prompts={"a": "criteria A"}).to_dict()

    assert payload["output_contract_version"] == OUTPUT_CONTRACT_VERSION


def test_config_id_deterministic_for_configs_with_prompts_unset() -> None:
    # output_contract_version is now unconditional in to_dict(), so two
    # configs that both leave prompt fields unset must still collide.
    assert compute_ensemble_config_id(_base_config()) == compute_ensemble_config_id(_base_config())


# --- providers: sync raters route criteria through compose_system_prompt -------


def test_anthropic_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.anthropic import AnthropicRater

    captured: dict[str, Any] = {}

    class _FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = SimpleNamespace(type="text", text="I")
            return SimpleNamespace(content=[block], id="resp-1", model="v1")

    class _FakeClient:
        messages = _FakeMessages()

    rater = AnthropicRater(model="claude-sonnet-5", model_version="v1", temperature=0.2)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["system"] == compose_system_prompt(None)

    rater.rate(_record(), prompt="custom criteria")
    assert captured["system"] == compose_system_prompt("custom criteria")


def test_anthropic_rater_passes_temperature_to_the_api_call() -> None:
    from attest.vendors.providers.anthropic import AnthropicRater

    captured: dict[str, Any] = {}

    class _FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = SimpleNamespace(type="text", text="I")
            return SimpleNamespace(content=[block], id="resp-1", model="v1")

    class _FakeClient:
        messages = _FakeMessages()

    rater = AnthropicRater(model="claude-sonnet-5", model_version="v1", temperature=0.7)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())

    assert captured["temperature"] == 0.7


def test_anthropic_rater_raises_on_model_version_drift() -> None:
    from attest.vendors.base import ModelVersionDriftError
    from attest.vendors.providers.anthropic import AnthropicRater

    class _FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            block = SimpleNamespace(type="text", text="I")
            return SimpleNamespace(content=[block], id="resp-1", model="v2-snapshot")

    class _FakeClient:
        messages = _FakeMessages()

    rater = AnthropicRater(model="claude-sonnet-5", model_version="v1", temperature=0.0)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    with pytest.raises(ModelVersionDriftError, match="v2-snapshot"):
        rater.rate(_record())


def test_openai_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.openai import OpenAIRater

    captured: dict[str, Any] = {}

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v1"
            )

    class _FakeClient:
        chat = SimpleNamespace(completions=_FakeCompletions())

    rater = OpenAIRater(model="gpt-5", model_version="v1", temperature=0.2)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["messages"][0] == {"role": "system", "content": compose_system_prompt(None)}

    rater.rate(_record(), prompt="custom criteria")
    assert captured["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }


def test_openai_rater_passes_temperature_to_the_api_call() -> None:
    from attest.vendors.providers.openai import OpenAIRater

    captured: dict[str, Any] = {}

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v1"
            )

    class _FakeClient:
        chat = SimpleNamespace(completions=_FakeCompletions())

    rater = OpenAIRater(model="gpt-5", model_version="v1", temperature=0.7)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())

    assert captured["temperature"] == 0.7


def test_openai_rater_raises_on_model_version_drift() -> None:
    from attest.vendors.base import ModelVersionDriftError
    from attest.vendors.providers.openai import OpenAIRater

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v2-snapshot"
            )

    class _FakeClient:
        chat = SimpleNamespace(completions=_FakeCompletions())

    rater = OpenAIRater(model="gpt-5", model_version="v1", temperature=0.0)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    with pytest.raises(ModelVersionDriftError, match="v2-snapshot"):
        rater.rate(_record())


def test_google_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.google import GoogleRater

    captured: list[Any] = []

    class _FakeModels:
        def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs["config"]["system_instruction"])
            return SimpleNamespace(text="I")

    class _FakeClient:
        models = _FakeModels()

    rater = GoogleRater(model="gemini-1.5-pro", model_version="v1", temperature=0.2)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    rater.rate(_record(), prompt="custom criteria")

    assert captured == [compose_system_prompt(None), compose_system_prompt("custom criteria")]


def test_google_rater_passes_temperature_to_generation_config() -> None:
    from attest.vendors.providers.google import GoogleRater

    captured: dict[str, Any] = {}

    class _FakeModels:
        def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(text="I")

    class _FakeClient:
        models = _FakeModels()

    rater = GoogleRater(model="gemini-1.5-pro", model_version="v1", temperature=0.7)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())

    assert captured["config"]["temperature"] == 0.7


def test_mistral_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.mistral import MistralRater

    captured: dict[str, Any] = {}

    class _FakeChat:
        def complete(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v1"
            )

    class _FakeClient:
        chat = _FakeChat()

    rater = MistralRater(model="mistral-small-latest", model_version="v1", temperature=0.2)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["messages"][0] == {"role": "system", "content": compose_system_prompt(None)}

    rater.rate(_record(), prompt="custom criteria")
    assert captured["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }


def test_mistral_rater_passes_temperature_to_the_api_call() -> None:
    from attest.vendors.providers.mistral import MistralRater

    captured: dict[str, Any] = {}

    class _FakeChat:
        def complete(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v1"
            )

    class _FakeClient:
        chat = _FakeChat()

    rater = MistralRater(model="mistral-small-latest", model_version="v1", temperature=0.7)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())

    assert captured["temperature"] == 0.7


def test_mistral_rater_raises_on_model_version_drift() -> None:
    from attest.vendors.base import ModelVersionDriftError
    from attest.vendors.providers.mistral import MistralRater

    class _FakeChat:
        def complete(self, **kwargs: Any) -> Any:
            message = SimpleNamespace(content="I")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], id="resp-1", model="v2-snapshot"
            )

    class _FakeClient:
        chat = _FakeChat()

    rater = MistralRater(model="mistral-small-latest", model_version="v1", temperature=0.0)
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    with pytest.raises(ModelVersionDriftError, match="v2-snapshot"):
        rater.rate(_record())


def test_openmodel_rater_composes_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from attest.vendors.providers import openmodel

    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"model": "v1", "choices": [{"message": {"content": "I"}}]}).encode(
                "utf-8"
            )

    def _fake_urlopen(request: Any, timeout: float) -> Any:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(openmodel.urllib.request, "urlopen", _fake_urlopen)

    rater = openmodel.OpenModelRater(model="local-model", model_version="v1", temperature=0.2)
    rater.rate(_record())
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt(None),
    }

    rater.rate(_record(), prompt="custom criteria")
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }


def test_openmodel_rater_passes_temperature_in_the_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from attest.vendors.providers import openmodel

    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"model": "v1", "choices": [{"message": {"content": "I"}}]}).encode(
                "utf-8"
            )

    def _fake_urlopen(request: Any, timeout: float) -> Any:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(openmodel.urllib.request, "urlopen", _fake_urlopen)

    rater = openmodel.OpenModelRater(model="local-model", model_version="v1", temperature=0.7)
    rater.rate(_record())

    assert captured["body"]["temperature"] == 0.7


def test_openmodel_rater_raises_on_model_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from attest.vendors.base import ModelVersionDriftError
    from attest.vendors.providers import openmodel

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"model": "v2-snapshot", "choices": [{"message": {"content": "I"}}]}
            ).encode("utf-8")

    def _fake_urlopen(request: Any, timeout: float) -> Any:
        return _FakeResponse()

    monkeypatch.setattr(openmodel.urllib.request, "urlopen", _fake_urlopen)

    rater = openmodel.OpenModelRater(model="local-model", model_version="v1", temperature=0.0)

    with pytest.raises(ModelVersionDriftError, match="v2-snapshot"):
        rater.rate(_record())


# --- providers: batch raters' request builders compose the system prompt -------


def test_anthropic_batch_request_composes_system_prompt() -> None:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    rater = AnthropicBatchRater(model="claude-sonnet-5", model_version="v1", temperature=0.3)

    default_request = rater._request(_record(), "item-0", None)
    custom_request = rater._request(_record(), "item-0", "custom criteria")

    assert default_request["params"]["system"] == compose_system_prompt(None)
    assert custom_request["params"]["system"] == compose_system_prompt("custom criteria")
    assert default_request["params"]["temperature"] == 0.3


def test_anthropic_batch_fetch_raises_on_model_version_drift() -> None:
    from attest.vendors.base import ModelVersionDriftError
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    rater = AnthropicBatchRater(model="claude-sonnet-5", model_version="v1", temperature=0.0)
    block = SimpleNamespace(type="text", text="I")
    message = SimpleNamespace(content=[block], model="v2-snapshot")
    entry = SimpleNamespace(
        custom_id="item-0", result=SimpleNamespace(type="succeeded", message=message)
    )

    class _FakeBatches:
        def results(self, batch_id: str) -> list[Any]:
            return [entry]

    class _FakeMessages:
        batches = _FakeBatches()

    class _FakeClient:
        messages = _FakeMessages()

    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]
    handle = BatchHandle(
        vendor="anthropic",
        model="claude-sonnet-5",
        provider_batch_id="batch-1",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
        ensemble_config_id="cfg",
        id_map={"rec-1": "item-0"},
    )

    with pytest.raises(ModelVersionDriftError, match="v2-snapshot"):
        rater.fetch(handle)


def test_openai_batch_request_line_composes_system_prompt() -> None:
    from attest.vendors.providers.openai import OpenAIBatchRater

    rater = OpenAIBatchRater(model="gpt-5", model_version="v1", temperature=0.3)

    default_line = rater._request_line(_record(), "item-0", None)
    custom_line = rater._request_line(_record(), "item-0", "custom criteria")

    assert default_line["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt(None),
    }
    assert custom_line["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }
    assert default_line["body"]["temperature"] == 0.3


def test_google_batch_request_composes_system_prompt() -> None:
    from attest.vendors.providers.google import GoogleBatchRater

    rater = GoogleBatchRater(model="gemini-1.5-pro", model_version="v1", temperature=0.3)

    default_request = rater._request(_record(), None)
    custom_request = rater._request(_record(), "custom criteria")

    assert default_request["config"]["system_instruction"] == compose_system_prompt(None)
    assert custom_request["config"]["system_instruction"] == compose_system_prompt(
        "custom criteria"
    )
    assert default_request["config"]["temperature"] == 0.3


def test_mistral_batch_request_line_composes_system_prompt() -> None:
    from attest.vendors.providers.mistral import MistralBatchRater

    rater = MistralBatchRater(model="mistral-small-latest", model_version="v1", temperature=0.3)

    default_line = rater._request_line(_record(), "item-0", None)
    custom_line = rater._request_line(_record(), "item-0", "custom criteria")

    assert default_line["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt(None),
    }
    assert custom_line["body"]["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }
    assert default_line["body"]["temperature"] == 0.3
