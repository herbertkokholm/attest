"""Integration tests for attest.cli: the screen -> audit-draw -> audit-apply -> validate flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attest.cli import _build_parser, main

_GOLD_SET = str(Path(__file__).resolve().parent.parent / "data" / "example_gold_set.json")

# Seed chosen so the two DeterministicRaters, over the four records the
# prefilter keeps, never straddle the exclude/include boundary and never
# exceed tau=1.0 in dispersion -- so nothing escalates -- while still
# excluding rec-001 (track 1) and rec-004 (track 2) by mean-vote sign.
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
    assert screen_summary["escalated"] == 0

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

    assert record["schema_version"] == "1.0"
    assert record["prisma"]["identified"] == 6
    assert record["prisma"]["duplicates_removed"] == 1
    assert record["prisma"]["after_dedup"] == 5
    assert record["prisma"]["prefilter_excluded"] == 1
    assert record["prisma"]["screened"] == 4
    assert record["escalation_rate"] == pytest.approx(0.0)
    assert record["recall"]["point"] is not None
    assert record["recall"]["floor"] is not None
    assert record["recall"]["floor"] <= record["recall"]["point"]
    assert record["recall"]["audit_n"] == 2
    assert record["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}


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

    # A single-vendor ensemble never escalates, so the pending queue is empty.
    rc = main(["adjudicate", "--run-dir", str(run_dir)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["pending"] == []


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


@pytest.mark.parametrize(
    "command",
    ["screen", "adjudicate", "audit-draw", "audit-apply", "validate", "ablate"],
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
        "adjudicate",
        "audit-draw",
        "audit-apply",
        "validate",
        "ablate",
    }
