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
    vendor_spec = {"model": "deterministic-v1", "model_version": "1", "prompt_version": "p1"}
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

    assert record["schema_version"] == "1.1"
    assert record["config"]["zero_policy"] == "escalate"
    assert record["prisma"]["identified"] == 6
    assert record["prisma"]["duplicates_removed"] == 1
    assert record["prisma"]["after_dedup"] == 5
    assert record["prisma"]["prefilter_excluded"] == 1
    assert record["prisma"]["screened"] == 4
    # rec-004's escalation was resolved before this validate ran, so the
    # decisions store no longer shows it as escalating.
    assert record["escalation_rate"] == pytest.approx(0.0)
    assert record["recall"]["point"] is not None
    assert record["recall"]["floor"] is not None
    assert record["recall"]["floor"] <= record["recall"]["point"]
    assert record["recall"]["audit_n"] == 2
    assert record["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}


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
    vendor_spec = {"model": "deterministic-v1", "model_version": "1", "prompt_version": "p1"}
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

    rc = main(["validate", "--run-dir", str(run_dir), "--input", _GOLD_SET])
    assert rc == 0
    record = json.loads(capsys.readouterr().out)

    assert record["tau_report"] == describe_tau(1.0, 2).to_dict()
    # The tau_report addition is CLI-output-only, not a validation-record
    # schema change; schema_version here reflects workstream C's
    # zero_policy field on Config instead.
    assert record["schema_version"] == "1.1"


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
    vendor_spec = {"model": "deterministic-v1", "model_version": "1", "prompt_version": "p1"}
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
    vendor_spec = {"model": "deterministic-v1", "model_version": "1", "prompt_version": "p1"}
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
    vendor_spec = {"model": "deterministic-v1", "model_version": "1", "prompt_version": "p"}
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
    ["screen", "batch-fetch", "adjudicate", "audit-draw", "audit-apply", "validate", "ablate"],
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
    }
