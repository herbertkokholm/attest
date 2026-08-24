"""Tests for attest.provenance: config ids, epochs, changelog, and run records."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from attest.provenance.changelog import (
    CHANGE_TYPE_EXPLICIT,
    CHANGE_TYPE_INITIAL,
    CHANGE_TYPE_SENTINEL_DRIFT,
    ChangeLog,
    ChangelogError,
    ConfigChangeEvent,
    diff_config_fields,
)
from attest.provenance.config import Config, VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import maybe_open_epoch, open_epoch
from attest.provenance.runs import RunRecord, start_run


def _config(tau: float = 0.5, temperature: float = 0.0) -> Config:
    return Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-4o",
                model_version="2024-08-06",
                prompt_version="v1",
                temperature=temperature,
            ),
            "anthropic": VendorSpec(
                model="claude-sonnet-5",
                model_version="2026-01",
                prompt_version="v1",
                temperature=temperature,
            ),
        },
        aggregation="majority",
        tau=tau,
    )


@pytest.mark.parametrize("field_name", ["model", "model_version", "prompt_version"])
def test_vendor_spec_rejects_a_todo_placeholder(field_name: str) -> None:
    kwargs = {
        "model": "gpt-4o",
        "model_version": "2024-08-06",
        "prompt_version": "v1",
        "temperature": 0.0,
    }
    kwargs[field_name] = "TODO:pin-openai-current-gen-dated-snapshot"

    with pytest.raises(ValueError, match="unresolved placeholder"):
        VendorSpec(**kwargs)


def test_vendor_spec_accepts_a_value_merely_containing_todo() -> None:
    # Only a literal TODO:-prefix is rejected -- a real value that happens to
    # contain the substring elsewhere must not false-positive.
    VendorSpec(model="not-a-TODO:-value", model_version="1", prompt_version="v1", temperature=0.0)


def test_vendor_spec_to_dict_omits_reasoning_effort_and_send_temperature_at_default() -> None:
    payload = VendorSpec(
        model="gpt-5.6-terra", model_version="v1", prompt_version="p1", temperature=0.0
    ).to_dict()

    assert "reasoning_effort" not in payload
    assert "send_temperature" not in payload


def test_vendor_spec_to_dict_includes_reasoning_effort_when_set() -> None:
    payload = VendorSpec(
        model="gpt-5.6-terra",
        model_version="v1",
        prompt_version="p1",
        temperature=0.0,
        reasoning_effort="none",
    ).to_dict()

    assert payload["reasoning_effort"] == "none"


def test_vendor_spec_to_dict_includes_send_temperature_when_false() -> None:
    payload = VendorSpec(
        model="claude-sonnet-5",
        model_version="v1",
        prompt_version="p1",
        temperature=0.0,
        send_temperature=False,
    ).to_dict()

    assert payload["send_temperature"] is False


def test_reasoning_effort_changes_the_ensemble_config_id() -> None:
    base = _config()
    with_reasoning_effort = Config(
        vendors={
            **base.vendors,
            "openai": VendorSpec(
                model=base.vendors["openai"].model,
                model_version=base.vendors["openai"].model_version,
                prompt_version=base.vendors["openai"].prompt_version,
                temperature=base.vendors["openai"].temperature,
                reasoning_effort="none",
            ),
        },
        aggregation=base.aggregation,
        tau=base.tau,
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(with_reasoning_effort)


def test_send_temperature_changes_the_ensemble_config_id() -> None:
    base = _config()
    with_send_temperature_false = Config(
        vendors={
            **base.vendors,
            "anthropic": VendorSpec(
                model=base.vendors["anthropic"].model,
                model_version=base.vendors["anthropic"].model_version,
                prompt_version=base.vendors["anthropic"].prompt_version,
                temperature=base.vendors["anthropic"].temperature,
                send_temperature=False,
            ),
        },
        aggregation=base.aggregation,
        tau=base.tau,
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(
        with_send_temperature_false
    )


def test_same_descriptor_yields_same_ensemble_config_id() -> None:
    id_a = compute_ensemble_config_id(_config())
    id_b = compute_ensemble_config_id(_config())

    assert id_a == id_b


def test_ensemble_config_id_is_independent_of_dict_construction_order() -> None:
    reordered = Config(
        vendors={
            "anthropic": VendorSpec(
                model="claude-sonnet-5",
                model_version="2026-01",
                prompt_version="v1",
                temperature=0.0,
            ),
            "openai": VendorSpec(
                model="gpt-4o", model_version="2024-08-06", prompt_version="v1", temperature=0.0
            ),
        },
        aggregation="majority",
        tau=0.5,
    )

    assert compute_ensemble_config_id(_config()) == compute_ensemble_config_id(reordered)


def test_config_to_dict_omits_unset_prompt_fields() -> None:
    payload = _config().to_dict()

    assert "default_prompt" not in payload
    assert "track_prompts" not in payload


def test_default_prompt_changes_the_ensemble_config_id() -> None:
    base = _config()
    with_prompt = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        default_prompt="Screen for inclusion in review X.",
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(with_prompt)


def test_track_prompts_change_the_ensemble_config_id_but_not_when_equal() -> None:
    base = _config()
    with_track_prompt = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        track_prompts={"review-a": "Screen for review A."},
    )
    same_again = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        track_prompts={"review-a": "Screen for review A."},
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(with_track_prompt)
    assert compute_ensemble_config_id(with_track_prompt) == compute_ensemble_config_id(same_again)


def test_zero_policy_defaults_to_escalate() -> None:
    assert Config().zero_policy == "escalate"


def test_zero_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unknown zero_policy"):
        Config(zero_policy="exclude")


@pytest.mark.parametrize("bogus", ["exclude", "", "ESCALATE", "auto"])
def test_zero_policy_rejects_every_unrecognized_value(bogus: str) -> None:
    with pytest.raises(ValueError):
        Config(zero_policy=bogus)


def test_zero_policy_omitted_from_to_dict_when_default() -> None:
    payload = _config().to_dict()

    assert "zero_policy" not in payload


def test_zero_policy_included_in_to_dict_when_include() -> None:
    payload = Config(zero_policy="include").to_dict()

    assert payload["zero_policy"] == "include"


def test_confidence_threshold_defaults_to_default_low_threshold() -> None:
    from attest.ensemble.confidence import DEFAULT_LOW_THRESHOLD

    assert Config().confidence_threshold == DEFAULT_LOW_THRESHOLD


@pytest.mark.parametrize("bogus", [-0.01, 1.01])
def test_confidence_threshold_rejects_value_outside_unit_interval(bogus: float) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        Config(confidence_threshold=bogus)


def test_confidence_threshold_never_included_in_to_dict() -> None:
    payload = Config(confidence_threshold=0.9).to_dict()

    assert "confidence_threshold" not in payload


def test_confidence_threshold_does_not_change_the_ensemble_config_id() -> None:
    # Unlike every other Config field tested above (default_prompt,
    # track_prompts, zero_policy), confidence_threshold must NEVER open a
    # new epoch: it changes only how an already-fixed excluded population
    # is stratified for audit, never what a vendor samples or the
    # ensemble's own aggregate decision (see EnsembleConfig.confidence_threshold
    # and docs/logprob_support.md).
    base = _config()
    different_threshold = Config(
        vendors=base.vendors, aggregation=base.aggregation, tau=base.tau, confidence_threshold=0.9
    )

    assert compute_ensemble_config_id(base) == compute_ensemble_config_id(different_threshold)


def test_config_hash_pinned_for_default_config_unaffected_by_zero_policy() -> None:
    # Pinned hash of _config()'s to_dict() shape: a default ("escalate")
    # config with no prompt fields, still omitting zero_policy since it's the
    # default. This pin was retired and recomputed once, deliberately, when
    # output_contract_version became unconditional in to_dict (it used to be
    # omitted here): the kernel-owned output contract is appended to every
    # composed prompt, including this config's no-criteria fallback, so its
    # version must always be config-hash-sensitive. That is a one-time id
    # change for configs supplying no criteria -- their composed prompt has
    # always contained the contract; only the hash was previously blind to
    # it. The zero_policy-omission invariant this test is named for is
    # otherwise unchanged.
    #
    # Retired and recomputed a second time when `VendorSpec.temperature` was
    # added: every vendor now carries a `temperature`, always included in
    # `VendorSpec.to_dict()` (unconditionally, like `model_version` and
    # `prompt_version`), so it is unconditionally hash-sensitive too.
    #
    # Retired and recomputed a third time when `Config.batch_size` was
    # added: it is unconditionally included in `to_dict()`, on par with
    # `vendors`/`aggregation`/`tau`/`x` (see `Config.batch_size`), so even a
    # config built before the field existed -- picking up its `0` default --
    # now hashes differently.
    # Retired and recomputed a fourth time when `Config.batch_size`'s default
    # changed from `0` (a placeholder, always checked against the corpus size
    # rather than applied) to `1` (a real one-record-per-request packing
    # width, applied whether or not a config sets it).
    pinned_hash = "8b47c9b35df524fb1772e65d7549fb0347412594f6805a859c83d8840158a97b"

    assert compute_ensemble_config_id(_config()) == pinned_hash


def test_config_hash_stable_for_default_zero_policy() -> None:
    base = _config()
    explicit_default = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        zero_policy="escalate",
    )

    assert compute_ensemble_config_id(base) == compute_ensemble_config_id(explicit_default)


def test_zero_policy_include_changes_the_ensemble_config_id() -> None:
    base = _config()
    with_include = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        zero_policy="include",
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(with_include)


def test_batch_size_is_included_in_config_to_dict() -> None:
    # Unlike zero_policy/track_prompts/default_prompt, batch_size is never
    # omitted -- it is `b_e` in the manuscript's C_e tuple (Eq. 1), on par
    # with vendors/aggregation/tau/x (see Config.batch_size).
    config = Config(
        vendors=_config().vendors,
        aggregation="majority",
        tau=0.5,
        batch_size=250,
    )

    assert config.to_dict()["batch_size"] == 250


def test_batch_size_change_opens_a_new_epoch() -> None:
    base = _config()
    different_batch_size = Config(
        vendors=base.vendors,
        aggregation=base.aggregation,
        tau=base.tau,
        batch_size=100,
    )

    assert compute_ensemble_config_id(base) != compute_ensemble_config_id(different_batch_size)


def test_batch_size_defaults_to_one() -> None:
    # One record per request, this kernel's original behavior -- not the old
    # placeholder default of 0, which was never actually applied to
    # anything (see Config.batch_size).
    config = Config(vendors=_config().vendors, aggregation="majority", tau=0.5)

    assert config.batch_size == 1


@pytest.mark.parametrize("batch_size", [0, -1, -100])
def test_batch_size_below_one_is_rejected_at_construction(batch_size: int) -> None:
    try:
        Config(vendors=_config().vendors, aggregation="majority", tau=0.5, batch_size=batch_size)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError(f"expected ValueError for batch_size={batch_size}")


def test_batch_output_contract_version_included_only_when_batch_size_above_one() -> None:
    # A batch_size == 1 configuration never composes the multi-record output
    # contract (see attest.vendors.base.BATCH_OUTPUT_CONTRACT), so its
    # version must stay out of to_dict()/ensemble_config_id at batch_size ==
    # 1 -- otherwise every existing batch_size == 1 config's hash would
    # change out from under it the moment this constant existed.
    single = Config(vendors=_config().vendors, aggregation="majority", tau=0.5, batch_size=1)
    batched = Config(vendors=_config().vendors, aggregation="majority", tau=0.5, batch_size=2)

    assert "batch_output_contract_version" not in single.to_dict()
    assert "batch_output_contract_version" in batched.to_dict()
    assert compute_ensemble_config_id(single) != compute_ensemble_config_id(batched)


def test_prompt_for_track_prefers_track_specific_over_default() -> None:
    config = Config(
        default_prompt="generic default",
        track_prompts={"review-a": "review A criteria", "2": "int-keyed track criteria"},
    )

    assert config.prompt_for_track("review-a") == "review A criteria"
    assert config.prompt_for_track(2) == "int-keyed track criteria"
    assert config.prompt_for_track("review-b") == "generic default"
    assert Config().prompt_for_track("review-a") is None


def test_changed_field_yields_different_id_opens_epoch_and_logs_change() -> None:
    base = _config(tau=0.5)
    changed = _config(tau=0.6)

    id_before = compute_ensemble_config_id(base)
    id_after = compute_ensemble_config_id(changed)
    assert id_before != id_after

    opened_at = datetime(2026, 1, 1, tzinfo=UTC)
    epoch_before = open_epoch(base, opened_at=opened_at)
    epoch_after = maybe_open_epoch(epoch_before, changed, opened_at=opened_at)

    assert epoch_after.id != epoch_before.id
    assert epoch_after.ensemble_config_id == id_after

    log = ChangeLog()
    event = log.record(before=id_before, after=id_after, reason="tau raised to 0.6")

    assert log.history_of(id_after) == [event]
    assert log.previous_config_id(id_after) == id_before


def test_temperature_change_yields_different_id_and_opens_epoch() -> None:
    base = _config(temperature=0.0)
    changed = _config(temperature=0.7)

    id_before = compute_ensemble_config_id(base)
    id_after = compute_ensemble_config_id(changed)
    assert id_before != id_after

    opened_at = datetime(2026, 1, 1, tzinfo=UTC)
    epoch_before = open_epoch(base, opened_at=opened_at)
    epoch_after = maybe_open_epoch(epoch_before, changed, opened_at=opened_at)

    assert epoch_after.id != epoch_before.id
    assert epoch_after.ensemble_config_id == id_after


def test_maybe_open_epoch_reuses_current_epoch_when_config_is_unchanged() -> None:
    config = _config()
    epoch = open_epoch(config)

    same = maybe_open_epoch(epoch, config)

    assert same is epoch


def test_run_record_round_trips_to_and_from_dict() -> None:
    run = start_run(
        type="screening",
        track="track-1",
        epoch_id="epoch-1",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    run.counts["screened"] = 42
    run.finish(status="completed", ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC))

    restored = RunRecord.from_dict(run.to_dict())

    assert restored == run


# --- changelog: change_type inference, validation, and field diffing -----------


def test_record_infers_initial_change_type_when_before_is_none() -> None:
    log = ChangeLog()

    event = log.record(before=None, after="cfg-1", reason="first configuration")

    assert event.change_type == CHANGE_TYPE_INITIAL


def test_record_infers_explicit_change_type_when_before_is_given() -> None:
    log = ChangeLog()

    event = log.record(before="cfg-1", after="cfg-2", reason="tau raised")

    assert event.change_type == CHANGE_TYPE_EXPLICIT


def test_record_accepts_explicit_sentinel_drift_change_type_with_equal_ids() -> None:
    log = ChangeLog()

    event = log.record(
        before="cfg-1",
        after="cfg-1",
        reason="sentinel drift: vendor v1",
        change_type=CHANGE_TYPE_SENTINEL_DRIFT,
    )

    assert event.change_type == CHANGE_TYPE_SENTINEL_DRIFT
    assert event.before == event.after


def test_change_event_rejects_unknown_change_type() -> None:
    with pytest.raises(ChangelogError):
        ConfigChangeEvent(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            before="cfg-1",
            after="cfg-2",
            reason="x",
            change_type="not_a_real_type",
        )


def test_change_event_rejects_initial_with_non_none_before() -> None:
    with pytest.raises(ChangelogError):
        ConfigChangeEvent(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            before="cfg-1",
            after="cfg-2",
            reason="x",
            change_type=CHANGE_TYPE_INITIAL,
        )


def test_change_event_rejects_non_initial_with_none_before() -> None:
    with pytest.raises(ChangelogError):
        ConfigChangeEvent(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            before=None,
            after="cfg-2",
            reason="x",
            change_type=CHANGE_TYPE_EXPLICIT,
        )


def test_change_event_round_trips_with_changed_fields_and_approver() -> None:
    event = ConfigChangeEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        before="cfg-1",
        after="cfg-2",
        reason="model version bumped",
        change_type=CHANGE_TYPE_EXPLICIT,
        changed_fields=("vendors.openai.model_version",),
        approver="reviewer-42",
    )

    restored = ConfigChangeEvent.from_dict(event.to_dict())

    assert restored == event


def test_changelog_to_list_and_from_list_round_trip() -> None:
    log = ChangeLog()
    log.record(before=None, after="cfg-1", reason="initial")
    log.record(before="cfg-1", after="cfg-2", reason="tau raised")

    restored = ChangeLog.from_list(log.to_list())

    assert restored == log


def test_diff_config_fields_lists_only_changed_dot_paths() -> None:
    base = _config(tau=0.5)
    changed = _config(tau=0.6)

    diff = diff_config_fields(base, changed)

    assert diff == ["tau"]


def test_diff_config_fields_detects_nested_vendor_field_change() -> None:
    base = _config()
    changed = Config(
        vendors={
            "openai": VendorSpec(
                model="gpt-4o", model_version="2025-01-01", prompt_version="v1", temperature=0.0
            ),
            "anthropic": base.vendors["anthropic"],
        },
        aggregation=base.aggregation,
        tau=base.tau,
    )

    diff = diff_config_fields(base, changed)

    assert diff == ["vendors.openai.model_version"]


def test_diff_config_fields_reports_every_field_when_before_is_none() -> None:
    config = _config()

    diff = diff_config_fields(None, config)

    assert "tau" in diff
    assert "aggregation" in diff
    assert any(key.startswith("vendors.") for key in diff)
