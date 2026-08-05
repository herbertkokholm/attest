"""Command-line entry point for the attest kernel.

Wires the prefilter, ensemble, adjudication, recall-audit, validation, and
ablation engine modules into file-based subcommands over a local run
directory (`attest.io.store.RunStore`). Every subcommand reads and writes
its state exclusively through `attest.io.store`; `screen` and `batch-fetch`
are the only ones that may reach the network, and only through a `Rater` or
`BatchRater` built by `attest.vendors`. All other subcommands -- `adjudicate`,
`audit-draw`, `audit-apply`, `validate`, `ablate` -- run entirely offline
over files already written to the run directory.

`screen --mode batch` submits one vendor batch per rater and persists the
resulting handles; with `--wait` it then polls each to completion before
writing votes, exactly as `--mode sync` (the default) does synchronously.
Without `--wait`, it exits right after submission, and a later `batch-fetch`
invocation resumes from the persisted handles -- polling, fetching, and
writing the votes -- so a batch job started in one process invocation can be
completed in another.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

from attest.ablation.xsweep import DEFAULT_MAX_SUBSETS_PER_X, AblationReport, sweep
from attest.contracts.input import NormalizedInput
from attest.ensemble.aggregate import (
    AGGREGATION_BOUNDARY_DISPERSION,
    KNOWN_ZERO_POLICIES,
    ZERO_POLICY_ESCALATE,
    Decision,
    g,
)
from attest.ensemble.confidence import (
    DEFAULT_LOW_THRESHOLD,
    MIN_SUPPORTING_VOTES,
    confidence_tier,
    record_confidence,
)
from attest.ensemble.tau import validate_tau
from attest.io.store import RunStore, StoreError, assemble_validation_record, load_input
from attest.planes.adjudication import final_label
from attest.planes.recall_audit import (
    AuditRow,
    ExcludedRecord,
    draw_audit_sample,
    ingest_audit_labels,
)
from attest.prefilter.framework import Prefilter, PrefilterOutcome, require_nonempty
from attest.provenance.config import Config as EnsembleConfig
from attest.provenance.config import VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import Epoch, maybe_open_epoch
from attest.provenance.runs import start_run
from attest.vendors.base import DeterministicRater, EnsembleRun, Rater, run_ensemble
from attest.vendors.batch import (
    BatchRater,
    DeterministicBatchRater,
    poll_and_fetch_batch,
    submit_batch,
)
from attest.vendors.registry import build_batch_raters, build_raters

_RELEVANT_LABEL = 1

# The kernel's default deterministic policy: exclude records with an empty
# abstract, then hand the rest to the ensemble. Applied identically by
# `screen` and `validate` so a validation run's PRISMA counts are
# recomputed exactly, not re-guessed, from the same original input file.
_DEFAULT_PREFILTER = Prefilter(rules=[require_nonempty("abstract")])


class CliError(Exception):
    """Raised for user-facing CLI errors: bad arguments or missing/malformed files."""


def _load_ensemble_config(path: Path) -> EnsembleConfig:
    """Load an ensemble configuration from a JSON file.

    Args:
        path: Path to a JSON file with "vendors" (mapping of vendor name to
            an object with "model", "model_version", "prompt_version",
            "temperature"), "aggregation", "tau", and optionally
            "default_prompt" (string), "track_prompts" (mapping of track to
            prompt text), "zero_policy" (one of "escalate"/"include",
            default "escalate"), and/or "confidence_threshold" (float in
            [0, 1], default `attest.ensemble.confidence.DEFAULT_LOW_THRESHOLD`
            -- the default `--confidence-threshold` a confidence-stratified
            `audit-draw` uses for this review, sourced from this same file
            for the same reason `tau` is, though unlike `tau` it never
            affects `ensemble_config_id`; see `EnsembleConfig.confidence_threshold`)
            fields.

    Returns:
        The parsed `EnsembleConfig`.

    Raises:
        ValueError: If "zero_policy" names an unrecognized policy, or
            "confidence_threshold" is outside `[0, 1]` (both propagated
            from `EnsembleConfig.__post_init__`).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    vendors = {
        name: VendorSpec(
            model=spec["model"],
            model_version=spec["model_version"],
            prompt_version=spec["prompt_version"],
            temperature=float(spec["temperature"]),
        )
        for name, spec in payload["vendors"].items()
    }
    return EnsembleConfig(
        vendors=vendors,
        aggregation=payload["aggregation"],
        tau=float(payload["tau"]),
        default_prompt=payload.get("default_prompt"),
        track_prompts=dict(payload.get("track_prompts", {})),
        zero_policy=payload.get("zero_policy", ZERO_POLICY_ESCALATE),
        confidence_threshold=float(payload.get("confidence_threshold", DEFAULT_LOW_THRESHOLD)),
    )


def _load_labels(path: Path) -> dict[str, int]:
    """Load a JSON object mapping record id to a human ordinal label."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CliError(f"labels file '{path}' must contain a JSON object of record id to label")
    return {str(record_id): int(label) for record_id, label in payload.items()}


def _build_raters(
    config: EnsembleConfig, *, deterministic_seed: int | None, request_logprobs: bool = False
) -> list[Rater]:
    """Build one rater per vendor in `config`, live or network-free.

    Args:
        config: The ensemble configuration naming which vendors participate.
        deterministic_seed: If given, build seeded `DeterministicRater`s
            instead of live vendor adapters, so `screen` never touches the
            network. If None, build live raters via `attest.vendors.registry`.
        request_logprobs: Forwarded to `attest.vendors.registry.build_raters`
            (live raters) or `DeterministicRater.request_logprobs`
            (network-free) -- see `attest.ensemble.confidence` for what
            consumes the resulting `raw_response["logprobs"]`.
    """
    if deterministic_seed is None:
        return build_raters(config, request_logprobs=request_logprobs)
    return [
        DeterministicRater(
            vendor=name,
            model=spec.model,
            seed=deterministic_seed,
            request_logprobs=request_logprobs,
        )
        for name, spec in config.vendors.items()
    ]


def _build_batch_raters(
    config: EnsembleConfig, *, deterministic_seed: int | None, request_logprobs: bool = False
) -> list[BatchRater]:
    """Build one `BatchRater` per vendor in `config`, live or network-free.

    Args:
        config: The ensemble configuration naming which vendors participate.
        deterministic_seed: If given, build seeded `DeterministicBatchRater`s
            instead of live vendor adapters, so batch mode never touches the
            network. If None, build live batch raters via
            `attest.vendors.registry`.
        request_logprobs: Forwarded to
            `attest.vendors.registry.build_batch_raters` (live raters) or
            `DeterministicBatchRater.request_logprobs` (network-free). Must
            be passed identically to both the `screen --mode batch` call
            that submits and the later `batch-fetch` call that fetches --
            like `deterministic_seed`, it is not itself persisted in
            `BatchHandle`, so the caller is responsible for consistency
            across the two invocations.
    """
    if deterministic_seed is None:
        return build_batch_raters(config, request_logprobs=request_logprobs)
    return [
        DeterministicBatchRater(
            vendor=name,
            model=spec.model,
            seed=deterministic_seed,
            request_logprobs=request_logprobs,
        )
        for name, spec in config.vendors.items()
    ]


def _persist_ensemble_run(
    store: RunStore,
    config: EnsembleConfig,
    ensemble_config_id: str,
    epoch: Epoch,
    ensemble_run: EnsembleRun,
    outcome: PrefilterOutcome,
    track: str,
) -> dict[str, Any]:
    """Write an ensemble run's votes, raw responses, decisions, and provenance
    record, and summarize them.

    Shared by `screen` (both sync and waited batch mode) and `batch-fetch`,
    so every path that produces an `EnsembleRun` persists it identically.
    """
    store.write_votes(ensemble_run.votes)
    store.write_raw_responses(ensemble_config_id, ensemble_run.raw_responses)

    decisions = {
        vv.record_id: g(
            vv, aggregation=config.aggregation, tau=config.tau, zero_policy=config.zero_policy
        )
        for vv in ensemble_run.votes
    }
    store.write_decisions(ensemble_config_id, decisions)

    run = start_run(type="screening", track=track, epoch_id=epoch.id)
    run.counts.update(
        {
            "identified": outcome.prisma.identified,
            "screened": outcome.prisma.passed,
            "escalated": sum(1 for d in decisions.values() if d.escalate),
        }
    )
    run.finish()
    store.write_run_record(run)

    return {
        "ensemble_config_id": ensemble_config_id,
        "epoch": epoch.id,
        "prisma": outcome.prisma.to_dict(),
        "escalated": run.counts["escalated"],
    }


def _screen_excluded_population(
    store: RunStore, normalized: NormalizedInput
) -> list[ExcludedRecord]:
    """Recompute the screen-excluded population from stored decisions.

    A record is screen-excluded if its stored decision is resolved (not
    still escalating) and its final label is not the relevant label.
    Escalated decisions that have not yet been adjudicated (see the
    `adjudicate` command) are left out, since their final label is not yet
    known.

    Args:
        store: A `RunStore` holding written decisions for this epoch.
        normalized: The original input, used to look up each record's track.
    """
    track_by_id = {record.id: record.track for record in normalized.records}
    population: list[ExcludedRecord] = []
    for record_id, decision in store.read_decisions().items():
        if decision.escalate or decision.auto_label == _RELEVANT_LABEL:
            continue
        population.append(
            ExcludedRecord(record_id=record_id, track=track_by_id.get(record_id, "unknown"))
        )
    return population


def _attach_confidence_tiers(
    population: Sequence[ExcludedRecord], store: RunStore, *, low_threshold: float
) -> list[ExcludedRecord]:
    """Compute and attach each record's confidence tier from already-stored votes/raw responses.

    Only called for `--stratify-by-confidence`: reads `votes.json`/
    `raw_responses.json` (already-persisted ensemble output; no new vendor
    calls) and scores each record via
    `attest.ensemble.confidence.record_confidence`/`confidence_tier`, the
    same figure `attest.planes.active_learning.select_for_review` reuses --
    a single, shared computation, not something recomputed differently per
    call site.

    Args:
        population: Screen-excluded records to score (track already set).
        store: The run's `RunStore`, read from only.
        low_threshold: The confidence-tier policy in force for this call
            (see `attest.ensemble.confidence.confidence_tier`).

    Returns:
        `population`, with each record's `confidence_tier` set.

    Raises:
        CliError: If a record in `population` has no stored vote vector --
            confidence cannot be scored without the votes it was derived
            from.
    """
    raw_responses = store.read_raw_responses()
    votes_by_id = {vote_vector.record_id: vote_vector for vote_vector in store.read_votes()}
    tiered: list[ExcludedRecord] = []
    for record in population:
        vote_vector = votes_by_id.get(record.record_id)
        if vote_vector is None:
            raise CliError(f"record '{record.record_id}': no stored votes to score confidence from")
        confidence = record_confidence(vote_vector, raw_responses.get(record.record_id, {}))
        tier = confidence_tier(confidence, low_threshold=low_threshold)
        tiered.append(replace(record, confidence_tier=tier))
    return tiered


def _audit_draw_size(value: str) -> int | None:
    """Parse `--size`: an int, or the literal "all" for the full population.

    Returns None for "all", resolved to the actual population size once it
    is known in `_cmd_audit_draw` (population size isn't known at argparse
    time).
    """
    if value == "all":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--size must be an integer or 'all', got {value!r}"
        ) from exc


def _population_sizes(
    population: Sequence[ExcludedRecord], *, include_confidence: bool = False
) -> dict[str, int]:
    """Build stratum population sizes covering both stratified and unstratified audit draws.

    Returns a mapping with an "all" entry (the full excluded population
    size, for unstratified draws) alongside one entry per distinct track
    (for `stratify_by_track=True` draws) and, when `include_confidence` is
    set and `population`'s records already carry a `confidence_tier` (see
    `_attach_confidence_tiers`), one entry per distinct tier (for
    `stratify_by_confidence=True` draws) -- so `validate` can look up
    whichever stratum names the stored audit rows actually used. Track and
    confidence-tier names share this same flat namespace, exactly as
    `attest.planes.recall_audit.AuditRow.stratum`/
    `attest.stats.recall.Stratum.name` already do; a track literally named
    `"low"` would collide with the confidence tier of the same name, but
    the two stratification modes are mutually exclusive per draw (see
    `attest.planes.recall_audit.draw_audit_sample`), so in practice only
    one of the two ever populates a given run's audit rows.
    """
    sizes: dict[str, int] = defaultdict(int)
    sizes["all"] = len(population)
    for record in population:
        sizes[str(record.track)] += 1
        if include_confidence and record.confidence_tier is not None:
            sizes[record.confidence_tier] += 1
    return dict(sizes)


def _ablation_report_to_dict(report: AblationReport) -> dict[str, Any]:
    """Serialize an `AblationReport` to a plain, JSON-serializable dict."""
    return {
        "candidate_vendors": list(report.candidate_vendors),
        "results": [
            {
                "x": result.x,
                "vendors": list(result.vendors),
                "alpha": result.alpha,
                "raw_agreement": result.raw_agreement,
                "recall": result.recall,
                "precision": result.precision,
                "escalation_rate": result.escalation_rate,
                "n_records": result.n_records,
                "leave_one_out": {
                    vendor: {
                        "delta_alpha": contribution.delta_alpha,
                        "delta_raw_agreement": contribution.delta_raw_agreement,
                        "delta_recall": contribution.delta_recall,
                        "delta_precision": contribution.delta_precision,
                        "delta_escalation_rate": contribution.delta_escalation_rate,
                    }
                    for vendor, contribution in result.leave_one_out.items()
                },
                "tau_report": result.tau_report.to_dict(),
            }
            for result in report.results.values()
        ],
    }


def _write_or_print(payload: str, out: str | None) -> None:
    if out is not None:
        Path(out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _cmd_screen(args: argparse.Namespace) -> int:
    """Run the deterministic prefilter, then the ensemble, over an input file.

    In `--mode sync` (the default), the ensemble runs synchronously and the
    votes, decisions, and run record are persisted before this returns. In
    `--mode batch`, one vendor batch per rater is submitted and its handle
    persisted; with `--wait`, this then polls every batch to completion and
    persists the run exactly as sync mode would, otherwise it returns right
    after submission and a later `batch-fetch` invocation finishes the run.
    """
    normalized = load_input(args.input)
    config = _load_ensemble_config(Path(args.config))
    store = RunStore(Path(args.run_dir))

    ensemble_config_id = store.write_config(config)

    try:
        current_epoch = store.read_epoch()
    except StoreError:
        current_epoch = None
    epoch = maybe_open_epoch(current_epoch, config)
    store.write_epoch(epoch)

    tau_report = validate_tau(config.tau, config.x)
    for message in tau_report.warnings:
        warnings.warn(message, stacklevel=2)
    store.write_tau_report(tau_report)

    outcome = _DEFAULT_PREFILTER.run(normalized.records)

    if args.mode == "sync":
        raters = _build_raters(
            config,
            deterministic_seed=args.deterministic_seed,
            request_logprobs=args.request_logprobs,
        )
        ensemble_run = run_ensemble(outcome.kept, raters, config)
    elif args.mode == "batch":
        batch_raters = _build_batch_raters(
            config,
            deterministic_seed=args.deterministic_seed,
            request_logprobs=args.request_logprobs,
        )
        submit_batch(outcome.kept, batch_raters, config, store)
        if not args.wait:
            summary = {
                "ensemble_config_id": ensemble_config_id,
                "epoch": epoch.id,
                "prisma": outcome.prisma.to_dict(),
                "mode": "batch",
                "submitted": True,
                "waited": False,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        ensemble_run = poll_and_fetch_batch(batch_raters, store)

    summary = _persist_ensemble_run(
        store, config, ensemble_config_id, epoch, ensemble_run, outcome, args.track
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_batch_fetch(args: argparse.Namespace) -> int:
    """Resume a `screen --mode batch` run: poll persisted handles, fetch, and persist votes."""
    normalized = load_input(args.input)
    store = RunStore(Path(args.run_dir))
    config = store.read_config()
    ensemble_config_id = compute_ensemble_config_id(config)
    epoch = store.read_epoch()

    outcome = _DEFAULT_PREFILTER.run(normalized.records)

    batch_raters = _build_batch_raters(
        config, deterministic_seed=args.deterministic_seed, request_logprobs=args.request_logprobs
    )
    ensemble_run = poll_and_fetch_batch(
        batch_raters,
        store,
        poll_interval=args.poll_interval,
        max_poll_interval=args.max_poll_interval,
    )

    summary = _persist_ensemble_run(
        store, config, ensemble_config_id, epoch, ensemble_run, outcome, args.track
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_adjudicate(args: argparse.Namespace) -> int:
    """List pending escalated decisions, or resolve one with an authoritative human label."""
    store = RunStore(Path(args.run_dir))
    ensemble_config_id = compute_ensemble_config_id(store.read_config())
    decisions = store.read_decisions()

    if args.record_id is None:
        pending = sorted(record_id for record_id, d in decisions.items() if d.escalate)
        print(json.dumps({"pending": pending}, indent=2, sort_keys=True))
        return 0

    if args.label is None:
        raise CliError("--label is required when --record-id is given")
    if args.record_id not in decisions:
        raise CliError(f"record '{args.record_id}' has no stored decision")

    decision = decisions[args.record_id]
    resolved_label = final_label(args.record_id, decision, args.label)
    resolved_decision = Decision(
        auto_label=resolved_label,
        escalate=False,
        dispersion=decision.dispersion,
        boundary=decision.boundary,
    )
    store.write_decisions(ensemble_config_id, {args.record_id: resolved_decision})

    print(json.dumps({"record_id": args.record_id, "final_label": resolved_label}, indent=2))
    return 0


def _cmd_audit_draw(args: argparse.Namespace) -> int:
    """Draw a random recall-audit sample from the current screen-excluded population."""
    store = RunStore(Path(args.run_dir))
    normalized = load_input(args.input)
    config = store.read_config()
    ensemble_config_id = compute_ensemble_config_id(config)

    population = _screen_excluded_population(store, normalized)
    if not population:
        raise CliError("no screen-excluded records are available to audit")

    if args.stratify_by_confidence:
        # low_threshold comes from config.confidence_threshold only -- the
        # same config.json this run's tau/vendors came from, and (like tau)
        # with no CLI override: one source of truth, no drift between what
        # config.json declares and what a draw actually used.
        low_threshold = config.confidence_threshold
        population = _attach_confidence_tiers(population, store, low_threshold=low_threshold)
        store.write_confidence_policy(
            low_threshold=low_threshold, min_supporting_votes=MIN_SUPPORTING_VOTES
        )

    size = args.size if args.size is not None else len(population)
    rng = Random(args.seed) if args.seed is not None else None
    rows = draw_audit_sample(
        population,
        size,
        stratify_by_track=args.stratify_by_track,
        stratify_by_confidence=args.stratify_by_confidence,
        rng=rng,
    )
    store.write_audit_rows(ensemble_config_id, rows)

    drawn = [{"record_id": row.record_id, "stratum": row.stratum} for row in rows]
    print(json.dumps({"drawn": drawn}, indent=2))
    return 0


def _cmd_audit_apply(args: argparse.Namespace) -> int:
    """Apply human gold-check labels to previously drawn, unlabeled audit rows."""
    store = RunStore(Path(args.run_dir))
    ensemble_config_id = compute_ensemble_config_id(store.read_config())
    labels = _load_labels(Path(args.labels))

    rows: list[AuditRow] = store.read_audit_rows()
    unlabeled = [row for row in rows if row.human_label is None]
    if not unlabeled:
        raise CliError("no unlabeled audit rows are pending")

    updated = ingest_audit_labels(unlabeled, labels)
    store.write_audit_rows(ensemble_config_id, updated)

    print(json.dumps({"labeled": [row.record_id for row in updated]}, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Assemble a full validation record for the run directory's current epoch."""
    store = RunStore(Path(args.run_dir))
    normalized = load_input(args.input)
    outcome = _DEFAULT_PREFILTER.run(normalized.records)

    truths = {
        record.id: record.gold_label
        for record in normalized.records
        if record.has_gold and record.gold_label is not None
    }
    population = _screen_excluded_population(store, normalized)
    # A stored confidence_policy.json is this run's evidence that its most
    # recent audit draw was confidence-stratified (see
    # RunStore.write_confidence_policy) -- reused here, not re-asked for on
    # the command line, so population sizes are rebuilt with the exact
    # low_threshold that actually produced the stored audit rows' strata.
    confidence_policy = store.read_confidence_policy()
    if confidence_policy is not None:
        population = _attach_confidence_tiers(
            population, store, low_threshold=confidence_policy["low_threshold"]
        )
    population_sizes = _population_sizes(
        population, include_confidence=confidence_policy is not None
    )

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes=population_sizes,
        confidence=args.confidence,
    )

    payload = record.to_dict()
    tau_report = store.read_tau_report()
    if tau_report is not None:
        # Provenance, not part of the frozen validation-record schema: the
        # tau proof (see attest.ensemble.tau) for the config this epoch ran
        # under, so every validation record self-documents why its tau
        # behaves the way it does.
        payload["tau_report"] = tau_report.to_dict()
    if confidence_policy is not None:
        # Same provenance treatment as tau_report: the confidence-tier
        # policy (attest.ensemble.confidence) this epoch's audit
        # stratification actually used.
        payload["confidence_policy"] = confidence_policy

    _write_or_print(json.dumps(payload, indent=2), args.out)
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    """Run a controlled ablation sweep over stored votes restricted to gold-labeled records."""
    store = RunStore(Path(args.run_dir))
    normalized = load_input(args.input)
    truths = {
        record.id: record.gold_label
        for record in normalized.records
        if record.has_gold and record.gold_label is not None
    }

    votes = [vv for vv in store.read_votes() if vv.record_id in truths]
    if not votes:
        raise CliError("no stored votes have a matching gold label to ablate over")

    rng = Random(args.seed) if args.seed is not None else None
    report = sweep(
        votes,
        truths,
        aggregation=args.aggregation,
        tau=args.tau,
        zero_policy=args.zero_policy,
        max_subsets_per_x=args.max_subsets_per_x,
        rng=rng,
    )

    payload = json.dumps(_ablation_report_to_dict(report), indent=2, sort_keys=True)
    _write_or_print(payload, args.out)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the `attest` argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="attest", description="Screening-and-self-validation kernel for LLM-ensemble review."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser(
        "screen", help="Run the prefilter and ensemble over an input file."
    )
    screen.add_argument("--input", required=True, help="Path to an input-contract JSON file.")
    screen.add_argument(
        "--config", required=True, help="Path to an ensemble configuration JSON file."
    )
    screen.add_argument("--run-dir", required=True, help="Run directory to persist results into.")
    screen.add_argument(
        "--track", default="default", help="Track label recorded on this run's provenance record."
    )
    screen.add_argument(
        "--deterministic-seed",
        type=int,
        default=None,
        help="Use network-free DeterministicRaters seeded with this value instead of live vendors.",
    )
    screen.add_argument(
        "--mode",
        choices=("sync", "batch"),
        default="sync",
        help="Run the ensemble synchronously (default) or submit vendor batch jobs instead.",
    )
    screen.add_argument(
        "--wait",
        action="store_true",
        help="In --mode batch, poll every submitted batch to completion before returning "
        "instead of exiting right after submission.",
    )
    screen.add_argument(
        "--request-logprobs",
        action="store_true",
        help="Ask every vendor that supports it (see docs/logprob_support.md) for per-vote "
        "logprobs, retained in raw_responses.json. Required for --stratify-by-confidence "
        "on a later audit-draw, or for confidence-driven active-learning selection.",
    )
    screen.set_defaults(handler=_cmd_screen)

    batch_fetch = subparsers.add_parser(
        "batch-fetch",
        help="Resume a 'screen --mode batch' run: poll handles, fetch, and persist votes.",
    )
    batch_fetch.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    batch_fetch.add_argument(
        "--input", required=True, help="Original input-contract JSON file, for PRISMA counts."
    )
    batch_fetch.add_argument(
        "--track", default="default", help="Track label recorded on this run's provenance record."
    )
    batch_fetch.add_argument(
        "--deterministic-seed",
        type=int,
        default=None,
        help="Use network-free DeterministicBatchRaters seeded with this value, not live vendors.",
    )
    batch_fetch.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds to wait before the first re-poll of a still-pending batch.",
    )
    batch_fetch.add_argument(
        "--max-poll-interval",
        type=float,
        default=30.0,
        help="Upper bound on the backed-off poll interval.",
    )
    batch_fetch.add_argument(
        "--request-logprobs",
        action="store_true",
        help="Must match the --request-logprobs value passed to the 'screen --mode batch' "
        "call that submitted this batch -- not itself persisted in the batch handle.",
    )
    batch_fetch.set_defaults(handler=_cmd_batch_fetch)

    adjudicate = subparsers.add_parser(
        "adjudicate", help="List pending escalated decisions, or resolve one."
    )
    adjudicate.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    adjudicate.add_argument(
        "--record-id", default=None, help="Record to resolve; omit to list pending records."
    )
    adjudicate.add_argument(
        "--label",
        type=int,
        default=None,
        choices=(-1, 0, 1),
        help="Authoritative human label for --record-id.",
    )
    adjudicate.set_defaults(handler=_cmd_adjudicate)

    audit_draw = subparsers.add_parser(
        "audit-draw", help="Draw a random recall-audit sample from the screen-excluded population."
    )
    audit_draw.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    audit_draw.add_argument(
        "--input", required=True, help="Original input-contract JSON file, for track lookup."
    )
    audit_draw.add_argument(
        "--size",
        type=_audit_draw_size,
        required=True,
        help="Number of records to draw, or 'all' to draw the entire "
        "screen-excluded population (for exact, not just floored, recall).",
    )
    audit_draw.add_argument(
        "--stratify-by-track", action="store_true", help="Stratify the draw by record track."
    )
    audit_draw.add_argument(
        "--stratify-by-confidence",
        action="store_true",
        help="Stratify the draw by ensemble confidence tier (low/high/unscored) instead of "
        "track. Mutually exclusive with --stratify-by-track. Requires 'screen "
        "--request-logprobs' to have been used for this run's votes. The 'low' cutoff comes "
        "from this run's config.json 'confidence_threshold' (see "
        "attest.ensemble.confidence.confidence_tier) -- the same runbook setting tau comes "
        "from, with no separate CLI override -- and is recorded to confidence_policy.json as "
        "run evidence, reused by 'validate' to reconstruct matching population sizes.",
    )
    audit_draw.add_argument(
        "--seed", type=int, default=None, help="Seed for a reproducible random draw."
    )
    audit_draw.set_defaults(handler=_cmd_audit_draw)

    audit_apply = subparsers.add_parser(
        "audit-apply", help="Apply human gold-check labels to drawn audit rows."
    )
    audit_apply.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    audit_apply.add_argument(
        "--labels", required=True, help="JSON file mapping record id to human ordinal label."
    )
    audit_apply.set_defaults(handler=_cmd_audit_apply)

    validate = subparsers.add_parser(
        "validate", help="Assemble a validation record from a run directory."
    )
    validate.add_argument("--run-dir", required=True, help="Run directory to read from.")
    validate.add_argument(
        "--input",
        required=True,
        help="Original input-contract JSON file, for gold labels and PRISMA counts.",
    )
    validate.add_argument(
        "--confidence", type=float, default=0.95, help="Confidence level for the recall floor."
    )
    validate.add_argument(
        "--out", default=None, help="Path to write the validation record; defaults to stdout."
    )
    validate.set_defaults(handler=_cmd_validate)

    ablate = subparsers.add_parser(
        "ablate", help="Run a controlled ablation sweep over stored votes on a gold set."
    )
    ablate.add_argument("--run-dir", required=True, help="Run directory holding stored votes.")
    ablate.add_argument("--input", required=True, help="Frozen gold-set input-contract JSON file.")
    ablate.add_argument("--aggregation", default=AGGREGATION_BOUNDARY_DISPERSION)
    ablate.add_argument("--tau", type=float, default=0.0)
    ablate.add_argument(
        "--zero-policy",
        choices=KNOWN_ZERO_POLICIES,
        default=ZERO_POLICY_ESCALATE,
        help="Disposition of a would-be auto_label==0 decision, held fixed across the sweep.",
    )
    ablate.add_argument("--max-subsets-per-x", type=int, default=DEFAULT_MAX_SUBSETS_PER_X)
    ablate.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling.")
    ablate.add_argument(
        "--out", default=None, help="Path to write the ablation report; defaults to stdout."
    )
    ablate.set_defaults(handler=_cmd_ablate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested attest subcommand.

    Args:
        argv: Command-line arguments, excluding the program name; defaults
            to `sys.argv[1:]` when None.

    Returns:
        Process exit code: 0 on success, 1 if the subcommand raised a
        recognized, user-facing error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (CliError, ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
