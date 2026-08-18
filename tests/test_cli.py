"""Integration tests for attest.cli: the screen -> audit-draw -> audit-apply -> validate flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attest.cli import _build_parser, main

_GOLD_SET = str(Path(__file__).resolve().parent.parent / "data" / "example_gold_set.json")

# Seed chosen so the two DeterministicRaters, over the four records the
# prefilter keeps, never straddle the exclude/include boundary and never
# exceed tau=1.0 in dispersion, so only the zero_policy rule can escalate
# anything: rec-001 (track 1) excludes by mean-vote sign (-1), rec-002 and
# rec-002-duplicate include by mean-vote sign (+1), and rec-004 (track 2)
# is an exact tie (votes 0, 0) -- escalated under the default
# zero_policy="escalate" rather than silently auto-labeled 0.
_DETERMINISTIC_SEED = 17


def _write_config(path: Path) -> None:
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 1.0,
            }
        )
    )


def test_end_to_end_screen_audit_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert screen_rc == 0
    screen_summary = json.loads(capsys.readouterr().out)
    assert screen_summary["prisma"]["passed"] == 4
    # rec-004's exact tie escalates under the default zero_policy="escalate";
    # it must go to a human, never be silently auto-committed as "uncertain".
    assert screen_summary["escalated"] == 1

    votes_ids = set(json.loads((run_dir / "votes.json").read_text())["votes"])
    raw_responses = json.loads((run_dir / "raw_responses.json").read_text())["raw_responses"]
    assert set(raw_responses) == votes_ids
    for by_vendor in raw_responses.values():
        assert set(by_vendor) == {"v1", "v2"}

    # Resolve the escalated tie before auditing: the screen-excluded
    # population (and hence the audit draw) only ever contains resolved,
    # non-relevant decisions -- see attest.cli._screen_excluded_population.
    adjudicate_rc = main(
        ["adjudicate", "--run-dir", str(run_dir), "--record-id", "rec-004", "--label", "-1"]
    )
    assert adjudicate_rc == 0
    capsys.readouterr()

    draw_rc = main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "2",
            "--seed",
            "1",
        ]
    )
    assert draw_rc == 0
    drawn = json.loads(capsys.readouterr().out)["drawn"]
    drawn_ids = {row["record_id"] for row in drawn}
    assert drawn_ids == {"rec-001", "rec-004"}

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"rec-001": 1, "rec-004": -1}))

    apply_rc = main(["audit-apply", "--run-dir", str(run_dir), "--labels", str(labels_path)])
    assert apply_rc == 0
    labeled = json.loads(capsys.readouterr().out)["labeled"]
    assert set(labeled) == drawn_ids

    validate_rc = main(["validate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert validate_rc == 0
    record = json.loads(capsys.readouterr().out)

    assert record["schema_version"] == "1.2"
    assert record["config"]["zero_policy"] == "escalate"
    assert record["prisma"]["identified"] == 6
    assert record["prisma"]["duplicates_removed"] == 1
    assert record["prisma"]["after_dedup"] == 5
    assert record["prisma"]["prefilter_excluded"] == 1
    assert record["prisma"]["screened"] == 4
    # rec-004's escalation was resolved before this validate ran, so the
    # decisions store no longer shows it as escalating.
    assert record["escalation_rate"] == pytest.approx(0.0)
    assert record["unresolved_escalations"] == 0
    assert record["recall"]["point"] is not None
    assert record["recall"]["floor"] is not None
    assert record["recall"]["floor"] <= record["recall"]["point"]
    assert record["recall"]["audit_n"] == 2
    assert record["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}


def test_validate_fails_closed_on_unresolved_escalation_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert screen_rc == 0
    capsys.readouterr()

    # rec-004 escalates and is never adjudicated here: validate must refuse
    # to silently drop it from the confusion matrix/recall TP.
    rc = main(["validate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert rc == 1
    err = capsys.readouterr().err
    assert "rec-004" in err
    assert "allow_unresolved_escalations" in err or "allow-unresolved-escalations" in err

    rc_allowed = main(
        [
            "validate",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--allow-unresolved-escalations",
        ]
    )
    assert rc_allowed == 0
    record = json.loads(capsys.readouterr().out)
    assert record["unresolved_escalations"] == 1
    # rec-004 has no gold label at all (see data/example_gold_set.json), so
    # its resolution status never affects the confusion matrix -- this is
    # identical to test_end_to_end_screen_audit_validate's fully-resolved
    # result. What differs is PRISMA: rec-004 is dropped from both
    # screen_excluded and included (3 resolved records, not 4) rather than
    # being counted as screen_excluded.
    assert record["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}
    assert record["prisma"]["screen_excluded"] == 1
    assert record["prisma"]["included"] == 2


def test_validate_resolves_escalations_via_human_labels_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    # Resolve rec-004 via an externally-supplied labels file instead of
    # 'attest adjudicate', e.g. an inclusion audit or an external review tool.
    human_labels_path = tmp_path / "human_labels.json"
    human_labels_path.write_text(json.dumps({"rec-004": -1}))

    rc = main(
        [
            "validate",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--human-labels",
            str(human_labels_path),
        ]
    )
    assert rc == 0
    record = json.loads(capsys.readouterr().out)
    assert record["unresolved_escalations"] == 0
    assert record["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}


def _write_confidence_config(path: Path, *, confidence_threshold: float | None = None) -> None:
    # Real vendor names, not "v1"/"v2": attest.ensemble.confidence dispatches
    # logprob extraction by vendor name, and three vendors are required to
    # ever meet MIN_SUPPORTING_VOTES.
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    payload = {
        "vendors": {
            "openai": vendor_spec,
            "mistral": vendor_spec,
            "together": vendor_spec,
        },
        "aggregation": "boundary_dispersion",
        "tau": 1.0,
    }
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold
    path.write_text(json.dumps(payload))


# Seed chosen (by brute-force search over the three-vendor ensemble) so all
# four screened records auto-label without escalating, and three of them
# (all with exactly 3/3 logprob-supporting votes) land in the
# screen-excluded population -- a confidence-stratified draw with no
# adjudication step needed first.
_CONFIDENCE_SEED = 12


def test_screen_with_request_logprobs_persists_logprobs_only_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    _write_confidence_config(config_path)

    without_dir = tmp_path / "run_without"
    without_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(without_dir),
            "--deterministic-seed",
            str(_CONFIDENCE_SEED),
        ]
    )
    assert without_rc == 0
    capsys.readouterr()
    raw_without = json.loads((without_dir / "raw_responses.json").read_text())["raw_responses"]
    assert raw_without
    for by_vendor in raw_without.values():
        for payload in by_vendor.values():
            assert "logprobs" not in payload

    with_dir = tmp_path / "run_with"
    with_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(with_dir),
            "--deterministic-seed",
            str(_CONFIDENCE_SEED),
            "--request-logprobs",
        ]
    )
    assert with_rc == 0
    capsys.readouterr()
    raw_with = json.loads((with_dir / "raw_responses.json").read_text())["raw_responses"]
    assert raw_with
    for by_vendor in raw_with.values():
        for payload in by_vendor.values():
            assert payload["logprobs"]["content"][0]["logprob"] <= 0.0


def test_audit_draw_stratify_by_confidence_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_confidence_config(config_path, confidence_threshold=0.5)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_CONFIDENCE_SEED),
            "--request-logprobs",
        ]
    )
    assert screen_rc == 0
    capsys.readouterr()

    draw_rc = main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "all",
            "--stratify-by-confidence",
        ]
    )
    assert draw_rc == 0
    drawn = json.loads(capsys.readouterr().out)["drawn"]
    assert len(drawn) == 3
    strata = {row["stratum"] for row in drawn}
    # Exactly 3/3 vendors support logprobs here, so every drawn record meets
    # MIN_SUPPORTING_VOTES and none falls back to the "unscored" stratum.
    assert strata <= {"low", "high"}

    policy = json.loads((run_dir / "confidence_policy.json").read_text())
    assert policy == {"low_threshold": 0.5, "min_supporting_votes": 3}

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({row["record_id"]: -1 for row in drawn}))
    apply_rc = main(["audit-apply", "--run-dir", str(run_dir), "--labels", str(labels_path)])
    assert apply_rc == 0
    capsys.readouterr()

    validate_rc = main(["validate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert validate_rc == 0
    record = json.loads(capsys.readouterr().out)
    assert record["confidence_policy"] == {"low_threshold": 0.5, "min_supporting_votes": 3}
    assert record["recall"]["point"] is not None
    assert record["recall"]["audit_n"] == 3


def test_audit_draw_confidence_threshold_comes_from_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_confidence_config(config_path, confidence_threshold=0.9)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_CONFIDENCE_SEED),
            "--request-logprobs",
        ]
    )
    assert screen_rc == 0
    capsys.readouterr()

    # No CLI flag for the threshold exists (mirroring tau, which has none
    # either) -- it must come from config.json's "confidence_threshold": 0.9.
    draw_rc = main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "all",
            "--stratify-by-confidence",
        ]
    )
    assert draw_rc == 0
    capsys.readouterr()

    policy = json.loads((run_dir / "confidence_policy.json").read_text())
    assert policy == {"low_threshold": 0.9, "min_supporting_votes": 3}


def test_audit_draw_rejects_track_and_confidence_stratification_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_confidence_config(config_path)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_CONFIDENCE_SEED),
            "--request-logprobs",
        ]
    )
    assert screen_rc == 0
    capsys.readouterr()

    draw_rc = main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "all",
            "--stratify-by-track",
            "--stratify-by-confidence",
        ]
    )
    assert draw_rc == 1
    err = capsys.readouterr().err
    assert "cannot both be set" in err


def test_audit_draw_size_all_draws_full_population(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    screen_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert screen_rc == 0
    capsys.readouterr()

    adjudicate_rc = main(
        ["adjudicate", "--run-dir", str(run_dir), "--record-id", "rec-004", "--label", "-1"]
    )
    assert adjudicate_rc == 0
    capsys.readouterr()

    # The screen-excluded population here is exactly {rec-001, rec-004} (see
    # test_end_to_end_screen_audit_validate above) -- "all" must draw both,
    # with no --size number needed even though "2" happens to be the answer.
    draw_rc = main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "all",
            "--seed",
            "1",
        ]
    )
    assert draw_rc == 0
    drawn = json.loads(capsys.readouterr().out)["drawn"]
    assert {row["record_id"] for row in drawn} == {"rec-001", "rec-004"}


def test_screen_persists_a_tau_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from attest.ensemble.tau import describe_tau

    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    payload = json.loads((run_dir / "tau_report.json").read_text())
    expected = describe_tau(1.0, 2)
    assert payload["tau"] == expected.tau
    assert payload["x"] == expected.x
    assert payload["dispersion_inert"] is True


def test_screen_warns_for_a_dispersion_inert_tau(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)  # tau=1.0, x=2 -- above the max reachable dispersion.

    with pytest.warns(UserWarning, match="dispersion term"):
        rc = main(
            [
                "screen",
                "--input",
                _GOLD_SET,
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--deterministic-seed",
                str(_DETERMINISTIC_SEED),
            ]
        )
    assert rc == 0
    capsys.readouterr()


def test_screen_rejects_a_nonsensical_tau(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    config_path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 0.0,
            }
        )
    )

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )

    assert rc == 1
    assert "tau must be > 0" in capsys.readouterr().err
    assert not (run_dir / "votes.json").exists()


def test_validate_surfaces_the_tau_report_as_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from attest.ensemble.tau import describe_tau

    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    # rec-004's tie escalates and is deliberately left unadjudicated here --
    # this test is about tau_report surfacing, not escalation handling -- so
    # it must opt in to validating over the unresolved escalation.
    rc = main(
        [
            "validate",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--allow-unresolved-escalations",
        ]
    )
    assert rc == 0
    record = json.loads(capsys.readouterr().out)

    assert record["tau_report"] == describe_tau(1.0, 2).to_dict()
    assert record["unresolved_escalations"] == 1
    # The tau_report addition is CLI-output-only, not a validation-record
    # schema change; schema_version here reflects workstream C's
    # zero_policy field on Config instead.
    assert record["schema_version"] == "1.2"


def test_screen_batch_mode_then_batch_fetch_matches_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    sync_dir = tmp_path / "sync-run"
    sync_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(sync_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert sync_rc == 0
    capsys.readouterr()

    batch_dir = tmp_path / "batch-run"
    submit_rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(batch_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--mode",
            "batch",
        ]
    )
    assert submit_rc == 0
    submit_summary = json.loads(capsys.readouterr().out)
    assert submit_summary["mode"] == "batch"
    assert submit_summary["submitted"] is True
    assert submit_summary["waited"] is False
    # Nothing to fetch yet: votes are only written once batch-fetch completes.
    assert not (batch_dir / "votes.json").exists()

    fetch_rc = main(
        [
            "batch-fetch",
            "--run-dir",
            str(batch_dir),
            "--input",
            _GOLD_SET,
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert fetch_rc == 0
    fetch_summary = json.loads(capsys.readouterr().out)
    assert fetch_summary["prisma"] == submit_summary["prisma"]

    assert (batch_dir / "votes.json").read_text() == (sync_dir / "votes.json").read_text()
    assert (batch_dir / "decisions.json").read_text() == (sync_dir / "decisions.json").read_text()


def test_screen_batch_mode_with_wait_matches_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    sync_dir = tmp_path / "sync-run"
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(sync_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    batch_dir = tmp_path / "batch-run"
    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(batch_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--mode",
            "batch",
            "--wait",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    assert (batch_dir / "votes.json").read_text() == (sync_dir / "votes.json").read_text()
    assert (batch_dir / "decisions.json").read_text() == (sync_dir / "decisions.json").read_text()


def test_adjudicate_lists_and_resolves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    # rec-004's votes are an exact tie (mean 0); zero_policy="escalate" (the
    # default) routes it to adjudication rather than silently auto-labeling
    # the uncertain category, so it is pending here.
    rc = main(["adjudicate", "--run-dir", str(run_dir)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["pending"] == ["rec-004"]

    resolve_rc = main(
        ["adjudicate", "--run-dir", str(run_dir), "--record-id", "rec-004", "--label", "-1"]
    )
    assert resolve_rc == 0
    assert json.loads(capsys.readouterr().out) == {"record_id": "rec-004", "final_label": -1}

    # Resolved: no longer pending.
    rc = main(["adjudicate", "--run-dir", str(run_dir)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["pending"] == []


def test_screen_rejects_an_unrecognized_zero_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    config_path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 1.0,
                "zero_policy": "exclude",
            }
        )
    )

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )

    assert rc == 1
    assert "unknown zero_policy" in capsys.readouterr().err
    assert not (run_dir / "votes.json").exists()


def test_screen_with_zero_policy_include_auto_labels_the_tie(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    config_path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 1.0,
                "zero_policy": "include",
            }
        )
    )

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    # rec-004's tie is folded into +1 rather than escalated.
    assert summary["escalated"] == 0

    decisions = json.loads((run_dir / "decisions.json").read_text())["decisions"]
    assert decisions["rec-004"]["auto_label"] == 1
    assert decisions["rec-004"]["escalate"] is False


def test_ablate_over_gold_labeled_votes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "1",
        "prompt_version": "p",
        "temperature": 0.0,
    }
    config_path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 0.5,
            }
        )
    )

    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    rc = main(["ablate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_vendors"] == ["v1", "v2"]
    assert len(report["results"]) >= 1
    for result in report["results"]:
        assert result["tau_report"]["x"] == result["x"]

    rc_include = main(
        [
            "ablate",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--zero-policy",
            "include",
        ]
    )
    assert rc_include == 0
    json.loads(capsys.readouterr().out)  # --zero-policy include parses and runs cleanly.


@pytest.mark.parametrize(
    "command",
    [
        "screen",
        "batch-fetch",
        "adjudicate",
        "audit-draw",
        "audit-apply",
        "validate",
        "ablate",
        "protocol",
        "manifest",
        "verify",
        "sentinel-init",
        "sentinel-check",
        "active-learning-select",
        "active-learning-review",
    ],
)
def test_help_works_for_every_command(command: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0


def test_top_level_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_build_parser_registers_every_command() -> None:
    parser = _build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers_action.choices) == {
        "screen",
        "batch-fetch",
        "adjudicate",
        "audit-draw",
        "audit-apply",
        "validate",
        "ablate",
        "protocol",
        "manifest",
        "verify",
        "sentinel-init",
        "sentinel-check",
        "active-learning-select",
        "active-learning-review",
    }


# --- changelog wiring: initial_config and explicit_config_change via screen ------


def test_screen_logs_initial_config_changelog_event_on_first_epoch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    events = json.loads((run_dir / "changelog.json").read_text())
    assert len(events) == 1
    assert events[0]["change_type"] == "initial_config"
    assert events[0]["before"] is None
    assert events[0]["approver"] is None

    # Re-running screen over the same, unchanged run directory reuses the
    # existing epoch and must not log a second changelog event.
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()
    events_again = json.loads((run_dir / "changelog.json").read_text())
    assert len(events_again) == 1


def _run_epoch_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Screen a first epoch into its own run directory, returning that run directory.

    Standing in for a runbook's `RUN_DIR=data/run` (epoch 1) so tests of
    `--previous-run-dir` have a real predecessor run directory -- with an
    actually-persisted `config.json` -- to point at, exactly as the CLI
    contract requires (a hand-passed config file path is not accepted).
    """
    run_dir = tmp_path / "run_epoch1"
    config_path = tmp_path / "config_epoch1.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()
    return run_dir


def test_screen_with_previous_run_dir_logs_explicit_change_with_field_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    previous_run_dir = _run_epoch_1(tmp_path, capsys)

    new_run_dir = tmp_path / "run_epoch2"
    new_config_path = tmp_path / "config_epoch2.json"
    vendor_spec = {
        "model": "deterministic-v1",
        "model_version": "2",  # bumped, the deliberate change
        "prompt_version": "p1",
        "temperature": 0.0,
    }
    new_config_path.write_text(
        json.dumps(
            {
                "vendors": {"v1": vendor_spec, "v2": vendor_spec},
                "aggregation": "boundary_dispersion",
                "tau": 1.0,
            }
        )
    )

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(new_config_path),
            "--run-dir",
            str(new_run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--previous-run-dir",
            str(previous_run_dir),
            "--change-reason",
            "bumped model_version to 2",
            "--approver",
            "reviewer-a",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    events = json.loads((new_run_dir / "changelog.json").read_text())
    assert len(events) == 1
    event = events[0]
    assert event["change_type"] == "explicit_config_change"
    assert event["before"] is not None
    assert event["reason"] == "bumped model_version to 2"
    assert event["approver"] == "reviewer-a"
    assert "vendors.v1.model_version" in event["changed_fields"]
    assert "vendors.v2.model_version" in event["changed_fields"]


def test_screen_rejects_previous_run_dir_with_no_stored_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_previous_dir = tmp_path / "not_a_real_run_dir"
    empty_previous_dir.mkdir()
    new_config_path = tmp_path / "config_epoch2.json"
    _write_config(new_config_path)

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(new_config_path),
            "--run-dir",
            str(tmp_path / "run_epoch2"),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--previous-run-dir",
            str(empty_previous_dir),
            "--change-reason",
            "bumped model_version to 2",
        ]
    )
    assert rc == 1


def test_screen_requires_change_reason_with_previous_run_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    previous_run_dir = _run_epoch_1(tmp_path, capsys)
    new_config_path = tmp_path / "config_epoch2.json"
    _write_config(new_config_path)

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(new_config_path),
            "--run-dir",
            str(tmp_path / "run_epoch2"),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--previous-run-dir",
            str(previous_run_dir),
        ]
    )
    assert rc == 1


def test_screen_rejects_approver_without_previous_run_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    rc = main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
            "--approver",
            "reviewer-a",
        ]
    )
    assert rc == 1


# --- adjudicate: reviewer/protocol provenance -------------------------------------


def test_adjudicate_persists_reviewer_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    rc = main(
        [
            "adjudicate",
            "--run-dir",
            str(run_dir),
            "--record-id",
            "rec-004",
            "--label",
            "-1",
            "--reviewer",
            "reviewer-b",
            "--protocol-id",
            "proto-1",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    records = json.loads((run_dir / "adjudication_records.json").read_text())
    assert records["rec-004"]["reviewer"] == "reviewer-b"
    assert records["rec-004"]["protocol_id"] == "proto-1"
    assert records["rec-004"]["human_label"] == -1
    assert records["rec-004"]["selection_reason"] == "tie"


# --- audit-draw / audit-apply: draw and label snapshots ---------------------------


def test_audit_draw_and_apply_persist_snapshots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()
    main(["adjudicate", "--run-dir", str(run_dir), "--record-id", "rec-004", "--label", "-1"])
    capsys.readouterr()

    main(
        [
            "audit-draw",
            "--run-dir",
            str(run_dir),
            "--input",
            _GOLD_SET,
            "--size",
            "2",
            "--seed",
            "1",
        ]
    )
    capsys.readouterr()

    drawn_snapshot = json.loads((run_dir / "audit_draw.json").read_text())["drawn"]
    assert set(drawn_snapshot) == {"rec-001", "rec-004"}

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"rec-001": 1, "rec-004": -1}))
    main(["audit-apply", "--run-dir", str(run_dir), "--labels", str(labels_path)])
    capsys.readouterr()

    labels_snapshot = json.loads((run_dir / "audit_labels.json").read_text())["labels"]
    assert labels_snapshot == {"rec-001": 1, "rec-004": -1}


# --- validate: run-directory-local snapshot ----------------------------------------


def test_validate_persists_a_run_directory_local_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()
    main(["adjudicate", "--run-dir", str(run_dir), "--record-id", "rec-004", "--label", "-1"])
    capsys.readouterr()

    rc = main(["validate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert rc == 0
    stdout_record = json.loads(capsys.readouterr().out)

    snapshot = json.loads((run_dir / "validation_record.json").read_text())
    assert snapshot == stdout_record


# --- protocol / manifest / verify: offline artifact integrity ---------------------


def test_protocol_manifest_verify_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    protocol_rc = main(
        [
            "protocol",
            "--run-dir",
            str(run_dir),
            "--stratify-by",
            "track",
            "--audit-size-policy",
            "n=600",
        ]
    )
    assert protocol_rc == 0
    protocol_payload = json.loads(capsys.readouterr().out)
    assert protocol_payload["audit_design"]["stratify_by"] == "track"

    manifest_rc = main(
        ["manifest", "--run-dir", str(run_dir), "--input", _GOLD_SET, "--seed", "screen=17"]
    )
    assert manifest_rc == 0
    manifest_payload = json.loads(capsys.readouterr().out)
    assert manifest_payload["ensemble_config_id"]
    assert manifest_payload["protocol_id"] == protocol_payload["protocol_id"]
    assert manifest_payload["seeds"] == {"screen": 17}
    assert "config.json" in manifest_payload["artifact_hashes"]
    assert "changelog.json" in manifest_payload["artifact_hashes"]

    verify_rc = main(["verify", "--run-dir", str(run_dir)])
    assert verify_rc == 0
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["ok"] is True

    # Tamper with a hashed artifact directly on disk, bypassing the store.
    (run_dir / "config.json").write_text('{"tampered": true}', encoding="utf-8")

    verify_rc_after_tamper = main(["verify", "--run-dir", str(run_dir)])
    assert verify_rc_after_tamper == 1
    tampered_payload = json.loads(capsys.readouterr().out)
    assert tampered_payload["ok"] is False
    assert any(p["artifact"] == "config.json" for p in tampered_payload["problems"])


def test_verify_without_a_manifest_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    rc = main(["verify", "--run-dir", str(run_dir)])
    assert rc == 1


# --- sentinel-init / sentinel-check: offline drift plumbing -----------------------


def _write_sentinel_input(path: Path, n: int = 6) -> None:
    records = [
        {"id": f"s{i}", "title": f"title {i}", "abstract": f"abstract {i}", "track": "sentinel"}
        for i in range(n)
    ]
    path.write_text(
        json.dumps({"schema_version": "1.0", "project": "sentinel", "records": records})
    )


def test_sentinel_init_then_check_with_identical_seed_shows_no_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    sentinel_input_path = tmp_path / "sentinel.json"
    _write_sentinel_input(sentinel_input_path)

    init_rc = main(
        [
            "sentinel-init",
            "--run-dir",
            str(run_dir),
            "--sentinel-input",
            str(sentinel_input_path),
            "--deterministic-seed",
            "99",
        ]
    )
    assert init_rc == 0
    capsys.readouterr()
    assert (run_dir / "sentinel_baseline.json").exists()

    check_rc = main(
        [
            "sentinel-check",
            "--run-dir",
            str(run_dir),
            "--sentinel-input",
            str(sentinel_input_path),
            "--deterministic-seed",
            "99",
        ]
    )
    assert check_rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["triggered"] is False
    assert "new_epoch" not in result

    evaluations = json.loads((run_dir / "sentinel_evaluations.json").read_text())
    assert len(evaluations) == 1


def test_sentinel_check_without_baseline_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    sentinel_input_path = tmp_path / "sentinel.json"
    _write_sentinel_input(sentinel_input_path)

    rc = main(
        [
            "sentinel-check",
            "--run-dir",
            str(run_dir),
            "--sentinel-input",
            str(sentinel_input_path),
            "--deterministic-seed",
            "99",
        ]
    )
    assert rc == 1


def test_sentinel_check_rejects_a_different_sentinel_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    sentinel_input_path = tmp_path / "sentinel.json"
    _write_sentinel_input(sentinel_input_path, n=6)
    main(
        [
            "sentinel-init",
            "--run-dir",
            str(run_dir),
            "--sentinel-input",
            str(sentinel_input_path),
            "--deterministic-seed",
            "99",
        ]
    )
    capsys.readouterr()

    different_sentinel_path = tmp_path / "sentinel_different.json"
    _write_sentinel_input(different_sentinel_path, n=7)  # different content -> different set id

    rc = main(
        [
            "sentinel-check",
            "--run-dir",
            str(run_dir),
            "--sentinel-input",
            str(different_sentinel_path),
            "--deterministic-seed",
            "99",
        ]
    )
    assert rc == 1


# --- active-learning-select / active-learning-review -------------------------------


def test_active_learning_select_then_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    select_rc = main(
        ["active-learning-select", "--run-dir", str(run_dir), "--dispersion-threshold", "0.0"]
    )
    assert select_rc == 0
    selected = json.loads(capsys.readouterr().out)["selected"]
    assert selected  # rec-004's exact tie has zero dispersion but escalates via zero_policy

    record_id = selected[0]["record_id"]
    review_rc = main(
        [
            "active-learning-review",
            "--run-dir",
            str(run_dir),
            "--record-id",
            record_id,
            "--reviewer",
            "reviewer-c",
            "--notes",
            "worth a prompt tweak",
        ]
    )
    assert review_rc == 0
    capsys.readouterr()

    reviews = json.loads((run_dir / "active_learning_reviews.json").read_text())
    assert len(reviews) == 1
    assert reviews[0]["reviewer"] == "reviewer-c"
    assert reviews[0]["notes"] == "worth a prompt tweak"


def test_active_learning_review_requires_a_prior_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(
        [
            "screen",
            "--input",
            _GOLD_SET,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--deterministic-seed",
            str(_DETERMINISTIC_SEED),
        ]
    )
    capsys.readouterr()

    rc = main(
        [
            "active-learning-review",
            "--run-dir",
            str(run_dir),
            "--record-id",
            "rec-999",
            "--reviewer",
            "reviewer-c",
        ]
    )
    assert rc == 1
