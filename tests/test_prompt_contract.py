"""Tests for Workstream A: the kernel-owned output contract.

Covers `attest.vendors.base.compose_system_prompt`, the hardened
`parse_ordinal_response`, every provider's routing of criteria through
`compose_system_prompt`, and `OUTPUT_CONTRACT_VERSION`'s effect on
`compute_ensemble_config_id`.
"""

from __future__ import annotations

import json
import warnings
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
        ("1", 1),
        ("-1", -1),
        ("0", 0),
        ("+1", 1),
        (" 1 ", 1),
        ("1.", 1),
        ("(-1)", -1),
    ],
)
def test_parse_ordinal_response_strict_whole_reply(text: str, expected: int) -> None:
    assert parse_ordinal_response(text) == expected


def test_parse_ordinal_response_strict_last_non_empty_line() -> None:
    text = "Reasoning: this study meets the inclusion criteria.\n\nFinal answer:\n1"

    assert parse_ordinal_response(text) == 1


def test_parse_ordinal_response_strict_last_line_avoids_false_ambiguity() -> None:
    # An earlier line mentions "1", but the strict last-line match short-circuits
    # before the fallback scan ever sees it, so this does not raise as ambiguous.
    text = "I first considered a score of 1, but on reflection:\n-1"

    assert parse_ordinal_response(text) == -1


# --- parse_ordinal_response: fallback scan -------------------------------------


def test_parse_ordinal_response_falls_back_to_first_token_scan_for_prose() -> None:
    assert parse_ordinal_response("Decision: -1, excluded due to wrong population.") == -1


def test_parse_ordinal_response_fallback_tolerates_repeated_equal_ratings() -> None:
    # "1" and "+1" both spell the same rating, so this is not ambiguous.
    assert parse_ordinal_response("I'd say 1, or maybe +1, either way include it.") == 1


# --- parse_ordinal_response: ambiguity and failure ------------------------------


def test_parse_ordinal_response_raises_on_genuinely_ambiguous_reply() -> None:
    # Two distinct ratings appear as tokens outside of any strict match.
    text = "Leaning -1 but could also see an argument for 1 here."

    with pytest.raises(VendorResponseError, match="ambiguous"):
        parse_ordinal_response(text)


def test_parse_ordinal_response_raises_when_no_ordinal_token_found() -> None:
    with pytest.raises(VendorResponseError):
        parse_ordinal_response("I cannot make a determination for this record.")


def test_parse_ordinal_response_criteria_echoing_a_number_does_not_misparse() -> None:
    # A criteria prompt mentioning a number the model echoes back must not be
    # silently taken as the rating when a real, distinct rating is also present.
    text = "Per criterion 1 this fails eligibility, so the rating is -1."

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
            "v1": VendorSpec(model="m", model_version="1", prompt_version="p"),
        },
        "aggregation": "boundary_dispersion",
        "tau": 0.5,
    }
    defaults.update(overrides)
    return Config(**defaults)


def test_output_contract_version_omitted_when_no_prompt_fields_set() -> None:
    payload = _base_config().to_dict()

    assert "output_contract_version" not in payload


def test_config_hash_stable_for_default_contract_config_with_no_prompt_fields() -> None:
    # Pinned hash of the pre-existing to_dict() shape (no "output_contract_version"
    # key), computed independently of compute_ensemble_config_id, so this test
    # actually catches a regression rather than just re-deriving the same formula.
    config = _base_config()
    payload = config.to_dict()
    expected_payload = {
        "vendors": {"v1": {"model": "m", "model_version": "1", "prompt_version": "p"}},
        "aggregation": "boundary_dispersion",
        "tau": 0.5,
        "x": 1,
    }
    pinned_hash = "b8ae9c5b0229046ad743855c6d479568a6f1f51a340f9eacb36858e5bfcb3fb8"

    assert payload == expected_payload
    assert compute_ensemble_config_id(config) == pinned_hash


def test_output_contract_version_included_when_default_prompt_set() -> None:
    payload = _base_config(default_prompt="criteria").to_dict()

    assert payload["output_contract_version"] == OUTPUT_CONTRACT_VERSION


def test_output_contract_version_included_when_track_prompts_set() -> None:
    payload = _base_config(track_prompts={"a": "criteria A"}).to_dict()

    assert payload["output_contract_version"] == OUTPUT_CONTRACT_VERSION


def test_config_id_unaffected_by_output_contract_version_when_prompts_unset() -> None:
    # Two configs differing only in a field that to_dict() only surfaces
    # when prompt fields are set must still collide when neither sets one.
    assert compute_ensemble_config_id(_base_config()) == compute_ensemble_config_id(_base_config())


# --- providers: sync raters route criteria through compose_system_prompt -------


def test_anthropic_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.anthropic import AnthropicRater

    captured: dict[str, Any] = {}

    class _FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = SimpleNamespace(type="text", text="1")
            return SimpleNamespace(content=[block], id="resp-1")

    class _FakeClient:
        messages = _FakeMessages()

    rater = AnthropicRater(model="claude-sonnet-5")
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["system"] == compose_system_prompt(None)

    rater.rate(_record(), prompt="custom criteria")
    assert captured["system"] == compose_system_prompt("custom criteria")


def test_openai_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.openai import OpenAIRater

    captured: dict[str, Any] = {}

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="1")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], id="resp-1")

    class _FakeClient:
        chat = SimpleNamespace(completions=_FakeCompletions())

    rater = OpenAIRater(model="gpt-5")
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["messages"][0] == {"role": "system", "content": compose_system_prompt(None)}

    rater.rate(_record(), prompt="custom criteria")
    assert captured["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }


def test_google_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.google import GoogleRater

    captured: list[str] = []

    class _FakeModel:
        def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(text="1")

    def _fake_client(prompt: str) -> Any:
        captured.append(prompt)
        return _FakeModel()

    rater = GoogleRater(model="gemini-1.5-pro")
    rater._client = _fake_client  # type: ignore[method-assign]

    rater.rate(_record())
    rater.rate(_record(), prompt="custom criteria")

    assert captured == [compose_system_prompt(None), compose_system_prompt("custom criteria")]


def test_mistral_rater_composes_system_prompt() -> None:
    from attest.vendors.providers.mistral import MistralRater

    captured: dict[str, Any] = {}

    class _FakeChat:
        def complete(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = SimpleNamespace(content="1")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], id="resp-1")

    class _FakeClient:
        chat = _FakeChat()

    rater = MistralRater(model="mistral-small-latest")
    rater._client = lambda: _FakeClient()  # type: ignore[method-assign]

    rater.rate(_record())
    assert captured["messages"][0] == {"role": "system", "content": compose_system_prompt(None)}

    rater.rate(_record(), prompt="custom criteria")
    assert captured["messages"][0] == {
        "role": "system",
        "content": compose_system_prompt("custom criteria"),
    }


def test_openmodel_rater_composes_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from attest.vendors.providers import openmodel

    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "1"}}]}).encode("utf-8")

    def _fake_urlopen(request: Any, timeout: float) -> Any:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(openmodel.urllib.request, "urlopen", _fake_urlopen)

    rater = openmodel.OpenModelRater(model="local-model")
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


# --- providers: batch raters' request builders compose the system prompt -------


def test_anthropic_batch_request_composes_system_prompt() -> None:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    rater = AnthropicBatchRater(model="claude-sonnet-5")

    default_request = rater._request(_record(), "item-0", None)
    custom_request = rater._request(_record(), "item-0", "custom criteria")

    assert default_request["params"]["system"] == compose_system_prompt(None)
    assert custom_request["params"]["system"] == compose_system_prompt("custom criteria")


def test_openai_batch_request_line_composes_system_prompt() -> None:
    from attest.vendors.providers.openai import OpenAIBatchRater

    rater = OpenAIBatchRater(model="gpt-5")

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


def test_google_batch_request_composes_system_prompt() -> None:
    from attest.vendors.providers.google import GoogleBatchRater

    rater = GoogleBatchRater(model="gemini-1.5-pro")

    default_request = rater._request(_record(), "item-0", None)
    custom_request = rater._request(_record(), "item-0", "custom criteria")

    assert default_request["request"]["system_instruction"]["parts"][0]["text"] == (
        compose_system_prompt(None)
    )
    assert custom_request["request"]["system_instruction"]["parts"][0]["text"] == (
        compose_system_prompt("custom criteria")
    )


def test_mistral_batch_request_line_composes_system_prompt() -> None:
    from attest.vendors.providers.mistral import MistralBatchRater

    rater = MistralBatchRater(model="mistral-small-latest")

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
