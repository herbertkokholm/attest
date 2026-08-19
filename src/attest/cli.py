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
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any

from attest.ablation.xsweep import DEFAULT_MAX_SUBSETS_PER_X, AblationReport, sweep
from attest.contracts.input import NormalizedInput
from attest.contracts.validation_record import SCHEMA_VERSION as VALIDATION_RECORD_SCHEMA_VERSION
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
from attest.io.store import (
    RunStore,
    StoreError,
    assemble_validation_record,
    hash_input_file,
    load_input,
)
from attest.planes.active_learning import ActiveLearningReview, select_for_review
from attest.planes.adjudication import AdjudicationItem, escalation_reason, final_label
from attest.planes.recall_audit import (
    AuditRow,
    ExcludedRecord,
    draw_audit_sample,
    ingest_audit_labels,
    population_frame_hash,
)
from attest.prefilter.framework import Prefilter, PrefilterOutcome, require_nonempty
from attest.provenance.changelog import (
    CHANGE_TYPE_EXPLICIT,
    CHANGE_TYPE_INITIAL,
    ConfigChangeEvent,
    diff_config_fields,
)
from attest.provenance.config import Config as EnsembleConfig
from attest.provenance.config import VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import Epoch, maybe_open_epoch
from attest.provenance.protocol import (
    STRATIFY_CONFIDENCE,
    STRATIFY_NONE,
    STRATIFY_TRACK,
    AdjudicationProtocol,
    AuditDesign,
    ReportingSpec,
    SentinelPolicy,
    ValidationProtocol,
)
from attest.provenance.runs import start_run
from attest.provenance.sentinel import (
    DEFAULT_ADVISORY_ALPHA_THRESHOLD,
    DEFAULT_HARD_TRIGGER_CROSSINGS,
    capture_baseline,
    check_sentinel_staleness,
    collect_sentinel_votes,
    compute_sentinel_set_id,
    evaluate_sentinel,
    open_epoch_for_hard_trigger,
)
from attest.vendors.base import DeterministicRater, EnsembleRun, Rater, run_ensemble
from attest.vendors.batch import (
    BatchRater,
    DeterministicBatchRater,
    poll_and_fetch_batch,
    submit_batch,
)
from attest.vendors.registry import build_batch_raters, build_raters
from attest.vendors.sdk_versions import sdk_versions

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
            "temperature"), "aggregation", "tau", and optionally "batch_size"
            (int -- the maximum number of records `attest screen` packs into
            one vendor request, see `EnsembleConfig.batch_size`; default 1,
            one record per request), "default_prompt" (string),
            "track_prompts" (mapping of track to prompt text), "zero_policy"
            (one of "escalate"/"include", default "escalate"), and/or
            "confidence_threshold" (float in [0, 1], default
            `attest.ensemble.confidence.DEFAULT_LOW_THRESHOLD` -- the default
            `--confidence-threshold` a confidence-stratified `audit-draw`
            uses for this review, sourced from this same file for the same
            reason `tau` is, though unlike `tau` it never affects
            `ensemble_config_id`; see `EnsembleConfig.confidence_threshold`)
            fields.

    Returns:
        The parsed `EnsembleConfig`.

    Raises:
        ValueError: If "batch_size" is less than 1, "zero_policy" names an
            unrecognized policy, or "confidence_threshold" is outside
            `[0, 1]` (all propagated from `EnsembleConfig.__post_init__`);
            or if any vendor's "model"/"model_version"/"prompt_version" is
            still an unresolved `TODO:`-prefixed placeholder (propagated
            from `VendorSpec.__post_init__`).
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
        batch_size=int(payload.get("batch_size", 1)),
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


def _record_screen_config_change(
    store: RunStore,
    args: argparse.Namespace,
    config: EnsembleConfig,
    ensemble_config_id: str,
    epoch: Epoch,
) -> None:
    """Log this run directory's first epoch to its persisted changelog.

    Called exactly once per run directory, when `screen` opens its epoch for
    the first time (a run directory is locked to one config/epoch, so every
    later `screen` invocation over the same directory reuses that epoch and
    logs nothing further). Without `--previous-run-dir`, this is an
    `initial_config` event (`before=None`); with it, this run directory is
    understood to be the *new* run directory of a deliberate, explicit
    configuration change (see `attest.io.store.RunStore` and the runbook's
    own "fresh RUN_DIR per epoch" convention), and this logs an
    `explicit_config_change` event carrying the machine-readable field diff.

    `--previous-run-dir` points at the *predecessor run directory*, not a
    hand-passed config file: the predecessor's own persisted `config.json`
    (`RunStore.read_config`) is what actually produced that epoch's votes,
    so reading it back is the only way to guarantee the logged `before` id
    and field diff describe what really ran, rather than trusting a sidecar
    file the caller might have edited or gone stale since then.
    """
    if args.previous_run_dir is None:
        if args.approver is not None:
            raise CliError("--approver requires --previous-run-dir")
        store.append_change_event(
            ConfigChangeEvent(
                timestamp=epoch.opened_at,
                before=None,
                after=ensemble_config_id,
                reason=args.change_reason
                or "initial ensemble configuration for this run directory",
                change_type=CHANGE_TYPE_INITIAL,
            )
        )
        return

    if not args.change_reason:
        raise CliError("--change-reason is required when --previous-run-dir is given")
    previous_store = RunStore(Path(args.previous_run_dir))
    try:
        previous_config = previous_store.read_config()
    except StoreError as exc:
        raise CliError(
            f"--previous-run-dir '{args.previous_run_dir}' has no stored config; it must be "
            "a run directory a prior 'screen' invocation already wrote to"
        ) from exc
    before_id = compute_ensemble_config_id(previous_config)
    store.append_change_event(
        ConfigChangeEvent(
            timestamp=epoch.opened_at,
            before=before_id,
            after=ensemble_config_id,
            reason=args.change_reason,
            change_type=CHANGE_TYPE_EXPLICIT,
            changed_fields=tuple(diff_config_fields(previous_config, config)),
            approver=args.approver,
        )
    )


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

    if current_epoch is None:
        _record_screen_config_change(store, args, config, ensemble_config_id, epoch)

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
    store.write_adjudication_record(
        AdjudicationItem(
            record_id=args.record_id,
            ensemble_config_id=ensemble_config_id,
            dispersion=decision.dispersion,
            boundary=decision.boundary,
            selection_reason=escalation_reason(decision),
            human_label=resolved_label,
            reviewer=args.reviewer,
            resolved_at=datetime.now(UTC),
            protocol_id=args.protocol_id,
        )
    )

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
    frame_hash = population_frame_hash(population)
    store.write_audit_rows(ensemble_config_id, rows)
    store.write_audit_draw(ensemble_config_id, rows, sampling_frame_hash=frame_hash)

    drawn = [{"record_id": row.record_id, "stratum": row.stratum} for row in rows]
    print(json.dumps({"drawn": drawn, "sampling_frame_hash": frame_hash}, indent=2))
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

    updated = ingest_audit_labels(unlabeled, labels, reviewer=args.reviewer, blinded=args.blinded)
    store.write_audit_rows(ensemble_config_id, updated)
    applied = {row.record_id: row.human_label for row in updated if row.human_label is not None}
    store.write_audit_labels(ensemble_config_id, applied)

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

    # Auto-resolve escalations from this run's own 'attest adjudicate' provenance
    # first, since that's the authoritative in-kernel resolution path; an
    # optional --human-labels file can supply (and override) resolutions made
    # outside 'attest adjudicate'. Restricted to record ids still marked
    # escalate=True in decisions.json: 'attest adjudicate' already rewrites a
    # resolved record's stored Decision (auto_label set, escalate cleared),
    # and final_label() rejects a human_label supplied for a decision that
    # isn't escalated -- so re-forwarding an already-applied resolution here
    # would turn an already-resolved record into a falsely "unresolved" one.
    still_escalated = {
        record_id for record_id, decision in store.read_decisions().items() if decision.escalate
    }
    human_labels: dict[str, int] = {
        record_id: item["human_label"]
        for record_id, item in store.read_adjudication_records().items()
        if item.get("human_label") is not None and record_id in still_escalated
    }
    if args.human_labels:
        human_labels.update(
            {k: v for k, v in _load_labels(Path(args.human_labels)).items() if k in still_escalated}
        )

    record = assemble_validation_record(
        store,
        prefilter_prisma=outcome.prisma,
        truths=truths,
        population_sizes=population_sizes,
        human_labels=human_labels,
        confidence=args.confidence,
        allow_unresolved_escalations=args.allow_unresolved_escalations,
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

    if args.max_staleness_days is not None:
        staleness = check_sentinel_staleness(
            store.read_sentinel_evaluations(), max_staleness_days=args.max_staleness_days
        )
        payload["sentinel_staleness"] = {
            "last_checked_at": (
                staleness.last_checked_at.isoformat()
                if staleness.last_checked_at is not None
                else None
            ),
            "staleness_days": staleness.staleness_days,
            "max_staleness_days": staleness.max_staleness_days,
            "stale": staleness.stale,
        }
        if staleness.stale:
            detail = (
                "no sentinel evaluation has ever been recorded for this run"
                if staleness.last_checked_at is None
                else f"the sentinel was last checked {staleness.staleness_days:.1f} day(s) ago"
            )
            message = (
                f"sentinel staleness: {detail}, exceeding the configured max of "
                f"{args.max_staleness_days} day(s) -- see 'attest sentinel-check'"
            )
            if args.fail_on_stale_sentinel:
                raise CliError(message)
            print(f"warning: {message}", file=sys.stderr)

    # Persist a run-directory-local copy so `attest manifest`/`verify` can
    # hash "the validation record" as an artifact regardless of wherever
    # --out (if any) pointed.
    store.write_validation_record_snapshot(payload)

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


def _cmd_protocol(args: argparse.Namespace) -> int:
    """Build and persist this run directory's validation-protocol descriptor.

    A validation protocol is analysis-plan provenance, not screening
    behavior: it may be written or revised at any point in a run
    directory's lifecycle, unlike `config.json`/`epoch.json` (see
    `attest.provenance.protocol` and `RunStore.write_protocol`).
    """
    store = RunStore(Path(args.run_dir))
    protocol = ValidationProtocol(
        audit_design=AuditDesign(
            stratify_by=args.stratify_by,
            audit_size_policy=args.audit_size_policy,
            confidence_level=args.confidence_level,
        ),
        adjudication_protocol=AdjudicationProtocol(
            protocol_version=args.adjudication_protocol_version,
            description=args.adjudication_description,
        ),
        sentinel_policy=SentinelPolicy(
            hard_trigger_crossings=args.hard_trigger_crossings,
            advisory_alpha_threshold=args.advisory_alpha_threshold,
            cadence_note=args.sentinel_cadence_note,
        ),
        reporting_spec=ReportingSpec(
            validation_record_schema_version=VALIDATION_RECORD_SCHEMA_VERSION,
            notes=args.reporting_notes,
        ),
    )
    protocol_id = store.write_protocol(protocol)
    print(json.dumps({"protocol_id": protocol_id, **protocol.to_dict()}, indent=2, sort_keys=True))
    return 0


def _parse_seeds(values: Sequence[str] | None) -> dict[str, int]:
    """Parse repeated `--seed NAME=VALUE` CLI arguments into a name-to-seed mapping."""
    seeds: dict[str, int] = {}
    for entry in values or ():
        if "=" not in entry:
            raise CliError(f"--seed must be NAME=VALUE, got {entry!r}")
        name, _, value = entry.partition("=")
        try:
            seeds[name] = int(value)
        except ValueError as exc:
            raise CliError(f"--seed value for '{name}' must be an integer, got {value!r}") from exc
    return seeds


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Build and persist a run manifest hashing this run directory's current artifacts."""
    store = RunStore(Path(args.run_dir))
    config = store.read_config()
    ensemble_config_id = compute_ensemble_config_id(config)
    protocol_id = store.read_protocol_id()
    input_hash = hash_input_file(args.input) if args.input else None
    manifest = store.write_manifest(
        ensemble_config_id=ensemble_config_id,
        protocol_id=protocol_id,
        input_hash=input_hash,
        input_source=args.input,
        seeds=_parse_seeds(args.seed),
        sdk_versions=sdk_versions(config.vendors),
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Offline-verify a run directory's artifacts against its stored manifest.

    Exits 1 (without raising) when integrity problems are found, so a CI
    step can gate on this command's exit code without parsing its output.
    """
    store = RunStore(Path(args.run_dir))
    report = store.verify()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _cmd_sentinel_init(args: argparse.Namespace) -> int:
    """Capture this ensemble configuration's baseline ratings on a frozen sentinel set."""
    store = RunStore(Path(args.run_dir))
    config = store.read_config()
    ensemble_config_id = compute_ensemble_config_id(config)
    epoch = store.read_epoch()

    sentinel_input = load_input(args.sentinel_input)
    sentinel_set_id = compute_sentinel_set_id(sentinel_input.records)

    raters = _build_raters(config, deterministic_seed=args.deterministic_seed)
    votes = collect_sentinel_votes(sentinel_input.records, raters)
    baseline = capture_baseline(sentinel_set_id, ensemble_config_id, epoch.id, votes)
    store.write_sentinel_baseline(baseline)

    print(
        json.dumps(
            {"sentinel_set_id": sentinel_set_id, "vendors": sorted(votes), "epoch": epoch.id},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_sentinel_check(args: argparse.Namespace) -> int:
    """Re-evaluate every vendor against its stored sentinel baseline.

    On a hard trigger, opens a new epoch and appends a `sentinel_drift`
    changelog event to this run directory (see
    `attest.provenance.sentinel.open_epoch_for_hard_trigger`) -- but, like
    an explicit config change, does not itself start writing further
    screening artifacts under the new epoch: this run directory stays
    locked to its original epoch, and continued screening under the new
    epoch belongs in a fresh run directory (printed in the result).
    """
    store = RunStore(Path(args.run_dir))
    config = store.read_config()
    baseline = store.read_sentinel_baseline()
    if baseline is None:
        raise CliError(
            "no sentinel baseline stored for this run directory; run sentinel-init first"
        )

    sentinel_input = load_input(args.sentinel_input)
    sentinel_set_id = compute_sentinel_set_id(sentinel_input.records)
    if sentinel_set_id != baseline.sentinel_set_id:
        raise CliError(
            f"--sentinel-input hashes to sentinel set '{sentinel_set_id}', which does not "
            f"match the stored baseline's sentinel set '{baseline.sentinel_set_id}'"
        )

    raters = _build_raters(config, deterministic_seed=args.deterministic_seed)
    current_votes = collect_sentinel_votes(sentinel_input.records, raters)

    protocol = store.read_protocol()
    sentinel_policy = protocol.sentinel_policy if protocol is not None else None
    hard_trigger_crossings = (
        args.hard_trigger_crossings
        if args.hard_trigger_crossings is not None
        else (
            sentinel_policy.hard_trigger_crossings
            if sentinel_policy is not None
            else DEFAULT_HARD_TRIGGER_CROSSINGS
        )
    )
    advisory_alpha_threshold = (
        args.advisory_alpha_threshold
        if args.advisory_alpha_threshold is not None
        else (
            sentinel_policy.advisory_alpha_threshold
            if sentinel_policy is not None
            else DEFAULT_ADVISORY_ALPHA_THRESHOLD
        )
    )

    evaluation = evaluate_sentinel(
        baseline,
        current_votes,
        hard_trigger_crossings=hard_trigger_crossings,
        advisory_alpha_threshold=advisory_alpha_threshold,
    )
    store.write_sentinel_evaluation(evaluation)

    result: dict[str, Any] = evaluation.to_dict()
    if evaluation.triggered:
        new_epoch, event = open_epoch_for_hard_trigger(config, evaluation)
        store.append_change_event(event)
        result["new_epoch"] = {
            "id": new_epoch.id,
            "ensemble_config_id": new_epoch.ensemble_config_id,
            "opened_at": new_epoch.opened_at.isoformat(),
        }
        result["action_required"] = (
            "hard trigger fired: this run directory's changelog now records the sentinel-drift "
            "transition; continue screening this configuration under a new run directory "
            "using the returned epoch, the same convention an explicit config change follows"
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_active_learning_select(args: argparse.Namespace) -> int:
    """Select stored votes for active-learning review and persist their provenance."""
    store = RunStore(Path(args.run_dir))
    config = store.read_config()
    ensemble_config_id = compute_ensemble_config_id(config)
    votes = store.read_votes()
    raw_responses = store.read_raw_responses()
    confidence_threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else config.confidence_threshold
    )

    selections = select_for_review(
        votes,
        dispersion_threshold=args.dispersion_threshold,
        include_boundary=not args.no_boundary,
        confidence_threshold=confidence_threshold,
        raw_responses=raw_responses,
        limit=args.limit,
    )
    store.write_active_learning_selections(ensemble_config_id, selections)

    payload = [
        {
            "record_id": s.record_id,
            "dispersion": s.dispersion,
            "boundary": s.boundary,
            "selection_reason": list(s.selection_reason),
        }
        for s in selections
    ]
    print(json.dumps({"selected": payload}, indent=2))
    return 0


def _cmd_active_learning_review(args: argparse.Namespace) -> int:
    """Record a human review of a previously selected active-learning record.

    Provenance only: recording a review never itself tunes a prompt,
    retrains anything, or writes a decision. A reviewer's `--notes` may
    record that this review motivates a later explicit config change; that
    change is then a separate, human-initiated `attest screen
    --previous-run-dir ... --approver ...` invocation, not something this
    command triggers.
    """
    store = RunStore(Path(args.run_dir))
    ensemble_config_id = compute_ensemble_config_id(store.read_config())
    stored = store.read_active_learning_selections()
    if args.record_id not in stored:
        raise CliError(
            f"record '{args.record_id}' has no stored active-learning selection; "
            "run active-learning-select first"
        )
    selection_reason = tuple(stored[args.record_id].get("selection_reason", ()))

    review = ActiveLearningReview(
        record_id=args.record_id,
        ensemble_config_id=ensemble_config_id,
        selection_reason=selection_reason,
        reviewer=args.reviewer,
        reviewed_at=datetime.now(UTC),
        protocol_id=args.protocol_id,
        notes=args.notes or "",
    )
    store.write_active_learning_review(review)

    print(json.dumps({"record_id": args.record_id, "reviewer": args.reviewer}, indent=2))
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
    screen.add_argument(
        "--previous-run-dir",
        default=None,
        help="Path to the predecessor run directory this run directory's configuration "
        "deliberately replaces -- its stored config.json (not a hand-passed file) is read "
        "back as the 'before' configuration, so the logged diff always describes what that "
        "epoch actually ran. When given, this run directory's first-epoch changelog event is "
        "logged as an explicit_config_change (with a machine-readable field diff) instead of "
        "the default initial_config -- the same 'fresh RUN_DIR per config change' convention "
        "the runbook already follows (e.g. RUN_DIR=data/run_epoch2).",
    )
    screen.add_argument(
        "--change-reason",
        default=None,
        help="Free-text reason for this run directory's first-epoch changelog event. Required "
        "when --previous-run-dir is given; optional (with a generic default) otherwise.",
    )
    screen.add_argument(
        "--approver",
        default=None,
        help="Id or pseudonym of the reviewer/approver who authorized this config change. "
        "Requires --previous-run-dir.",
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
    adjudicate.add_argument(
        "--reviewer",
        default=None,
        help="Id or pseudonym of the resolving reviewer, recorded to adjudication_records.json.",
    )
    adjudicate.add_argument(
        "--protocol-id",
        default=None,
        help="Id of the adjudication protocol in force (see 'attest protocol'), recorded to "
        "adjudication_records.json.",
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
    audit_apply.add_argument(
        "--reviewer",
        default=None,
        help="Id or pseudonym of the human who produced every label in --labels, recorded on "
        "each updated audit row.",
    )
    audit_apply.add_argument(
        "--blinded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether --reviewer gold-checked these rows without seeing the ensemble's "
        "screen-excluded decision, recorded on each updated audit row.",
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
        "--human-labels",
        default=None,
        help="JSON file mapping record id to human ordinal label, merged on top of this run's "
        "stored 'attest adjudicate' resolutions (read automatically) to resolve escalated "
        "decisions. Only needed for escalations resolved outside 'attest adjudicate'.",
    )
    validate.add_argument(
        "--allow-unresolved-escalations",
        action="store_true",
        help="Proceed even if some escalated decisions have no resolved human label, omitting "
        "them from the confusion matrix, PRISMA counts, and recall's TP (reported in the "
        "output's 'unresolved_escalations' field). Default: refuse to validate until every "
        "escalation is resolved.",
    )
    validate.add_argument(
        "--max-staleness-days",
        type=float,
        default=None,
        help="If given, check this run's most recent sentinel evaluation (see "
        "'attest sentinel-check') against this cadence in days, reporting the result in the "
        "output's 'sentinel_staleness' field. A stale or never-checked sentinel prints a "
        "warning by default; pass --fail-on-stale-sentinel to make it a hard failure instead. "
        "Omit for no check at all (default).",
    )
    validate.add_argument(
        "--fail-on-stale-sentinel",
        action="store_true",
        help="With --max-staleness-days, refuse to validate (instead of warning) when the "
        "sentinel is stale or was never checked.",
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

    protocol = subparsers.add_parser(
        "protocol", help="Build and persist this run directory's validation-protocol descriptor."
    )
    protocol.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    protocol.add_argument(
        "--stratify-by",
        choices=(STRATIFY_NONE, STRATIFY_TRACK, STRATIFY_CONFIDENCE),
        default=STRATIFY_NONE,
        help="What the random recall audit stratifies by, for the protocol record.",
    )
    protocol.add_argument("--audit-size-policy", default="", help="Free-text audit-budget policy.")
    protocol.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Two-sided confidence level for the recall floor/interval.",
    )
    protocol.add_argument("--adjudication-protocol-version", default="1")
    protocol.add_argument("--adjudication-description", default="")
    protocol.add_argument(
        "--hard-trigger-crossings",
        type=int,
        default=DEFAULT_HARD_TRIGGER_CROSSINGS,
        help="Sentinel hard-trigger threshold recorded into the protocol (see 'sentinel-check').",
    )
    protocol.add_argument(
        "--advisory-alpha-threshold", type=float, default=DEFAULT_ADVISORY_ALPHA_THRESHOLD
    )
    protocol.add_argument("--sentinel-cadence-note", default="")
    protocol.add_argument("--reporting-notes", default="")
    protocol.set_defaults(handler=_cmd_protocol)

    manifest = subparsers.add_parser(
        "manifest",
        help="Build and persist a run manifest hashing this run directory's current artifacts.",
    )
    manifest.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    manifest.add_argument(
        "--input",
        default=None,
        help="Path to the input-contract file this run screened, hashed (not copied) into the "
        "manifest as provenance of what was screened.",
    )
    manifest.add_argument(
        "--seed",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="A named random seed used by this run, e.g. --seed screen=42. Repeatable.",
    )
    manifest.set_defaults(handler=_cmd_manifest)

    verify = subparsers.add_parser(
        "verify", help="Offline-verify a run directory's artifacts against its stored manifest."
    )
    verify.add_argument("--run-dir", required=True, help="Run directory to verify.")
    verify.set_defaults(handler=_cmd_verify)

    sentinel_init = subparsers.add_parser(
        "sentinel-init",
        help="Capture this ensemble configuration's baseline ratings on a frozen sentinel set.",
    )
    sentinel_init.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    sentinel_init.add_argument(
        "--sentinel-input",
        required=True,
        help="Input-contract JSON file holding the frozen sentinel set's records.",
    )
    sentinel_init.add_argument(
        "--deterministic-seed",
        type=int,
        default=None,
        help="Use network-free DeterministicRaters seeded with this value instead of live vendors.",
    )
    sentinel_init.set_defaults(handler=_cmd_sentinel_init)

    sentinel_check = subparsers.add_parser(
        "sentinel-check",
        help="Re-evaluate every vendor against its stored sentinel baseline.",
    )
    sentinel_check.add_argument("--run-dir", required=True, help="Run directory to read/write.")
    sentinel_check.add_argument(
        "--sentinel-input",
        required=True,
        help="Input-contract JSON file holding the same frozen sentinel set used by sentinel-init.",
    )
    sentinel_check.add_argument(
        "--deterministic-seed",
        type=int,
        default=None,
        help="Use network-free DeterministicRaters seeded with this value instead of live vendors.",
    )
    sentinel_check.add_argument(
        "--hard-trigger-crossings",
        type=int,
        default=None,
        help="Override the hard-trigger crossing threshold; defaults to the stored protocol's "
        "sentinel_policy, or the kernel default if no protocol is stored.",
    )
    sentinel_check.add_argument(
        "--advisory-alpha-threshold",
        type=float,
        default=None,
        help="Override the advisory alpha threshold; defaults as --hard-trigger-crossings does.",
    )
    sentinel_check.set_defaults(handler=_cmd_sentinel_check)

    active_learning_select = subparsers.add_parser(
        "active-learning-select",
        help="Select stored votes for active-learning review and persist their provenance.",
    )
    active_learning_select.add_argument(
        "--run-dir", required=True, help="Run directory to read/write."
    )
    active_learning_select.add_argument("--dispersion-threshold", type=float, default=0.0)
    active_learning_select.add_argument(
        "--no-boundary", action="store_true", help="Disable the boundary-split selection signal."
    )
    active_learning_select.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override config.json's confidence_threshold for this selection.",
    )
    active_learning_select.add_argument("--limit", type=int, default=None)
    active_learning_select.set_defaults(handler=_cmd_active_learning_select)

    active_learning_review = subparsers.add_parser(
        "active-learning-review",
        help="Record a human review of a previously selected active-learning record.",
    )
    active_learning_review.add_argument(
        "--run-dir", required=True, help="Run directory to read/write."
    )
    active_learning_review.add_argument("--record-id", required=True)
    active_learning_review.add_argument("--reviewer", required=True)
    active_learning_review.add_argument("--protocol-id", default=None)
    active_learning_review.add_argument("--notes", default=None)
    active_learning_review.set_defaults(handler=_cmd_active_learning_review)

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
