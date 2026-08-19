"""Local, storage-agnostic persistence: a JSON run directory, no database or object storage.

This module is the only place in the kernel that touches a filesystem. It
reads an input-contract payload into a `NormalizedInput`, persists one
epoch's raw votes, decisions, config, audit sample, batch handles, and run
records as plain JSON files under a run directory, and assembles a full
`attest.contracts.validation_record.ValidationRecord` from that stored
state. Writes are idempotent: writing the same content twice yields the same
file, and re-running a write for the same record id upserts it rather than
duplicating it. Every epoch-scoped artifact (votes, decisions, audit rows) is
stamped with the `ensemble_config_id` it was produced under, and a run
directory refuses to mix data stamped with two different ids. Batch handles
(`attest.vendors.batch.BatchHandle`) are persisted as opaque JSON, keyed by
vendor name, so `attest.vendors.batch` can round-trip them without this
module importing it back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from attest.contracts.input import NormalizedInput, validate_and_normalize
from attest.contracts.validation_record import (
    Agreement,
    ErrorCorrelation,
    PairwiseFnCorrelation,
    Recall,
    ValidationRecord,
)
from attest.contracts.validation_record import Config as RecordConfig
from attest.contracts.validation_record import build as build_validation_record
from attest.ensemble.aggregate import ZERO_POLICY_ESCALATE, Decision
from attest.ensemble.confidence import DEFAULT_LOW_THRESHOLD
from attest.ensemble.tau import TauReport
from attest.ensemble.votes import Vote, VoteVector
from attest.planes.active_learning import ActiveLearningReview, ActiveLearningSelection
from attest.planes.adjudication import AdjudicationError, AdjudicationItem, final_label
from attest.planes.recall_audit import AuditRow, build_strata
from attest.prefilter.framework import Prisma as PrefilterPrisma
from attest.provenance.changelog import ChangeLog, ConfigChangeEvent
from attest.provenance.config import Config as EnsembleConfig
from attest.provenance.config import VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import Epoch
from attest.provenance.manifest import IntegrityReport as ManifestIntegrityReport
from attest.provenance.manifest import RunManifest, build_manifest, hash_file, verify_artifacts
from attest.provenance.protocol import ValidationProtocol, compute_protocol_id
from attest.provenance.runs import RunRecord
from attest.provenance.sentinel import SentinelBaseline, SentinelEvaluation
from attest.stats.agreement import agreement_report, pairwise_alpha
from attest.stats.confusion import RELEVANT_LABEL
from attest.stats.confusion import confusion_matrix as _confusion_matrix
from attest.stats.correlation import build_predictions_by_vendor, pairwise_fn_correlation
from attest.stats.recall import stratified_recall

CONFIG_FILENAME = "config.json"
EPOCH_FILENAME = "epoch.json"
VOTES_FILENAME = "votes.json"
DECISIONS_FILENAME = "decisions.json"
AUDIT_FILENAME = "audit.json"
RUNS_FILENAME = "runs.json"
BATCH_HANDLES_FILENAME = "batch_handles.json"
RAW_RESPONSES_FILENAME = "raw_responses.json"
TAU_REPORT_FILENAME = "tau_report.json"
CONFIDENCE_POLICY_FILENAME = "confidence_policy.json"
CHANGELOG_FILENAME = "changelog.json"
PROTOCOL_FILENAME = "protocol.json"
MANIFEST_FILENAME = "manifest.json"
AUDIT_DRAW_FILENAME = "audit_draw.json"
AUDIT_LABELS_FILENAME = "audit_labels.json"
VALIDATION_RECORD_FILENAME = "validation_record.json"
SENTINEL_BASELINE_FILENAME = "sentinel_baseline.json"
SENTINEL_EVALUATIONS_FILENAME = "sentinel_evaluations.json"
ADJUDICATION_RECORDS_FILENAME = "adjudication_records.json"
ACTIVE_LEARNING_SELECTIONS_FILENAME = "active_learning_selections.json"
ACTIVE_LEARNING_REVIEWS_FILENAME = "active_learning_reviews.json"

# The artifact set `attest manifest`/`attest verify` hash for offline
# integrity verification (see attest.provenance.manifest). Deliberately not
# every file `RunStore` can produce -- batch_handles.json and
# confidence_policy.json are execution-strategy/policy plumbing, not one of
# the manuscript's named artifact categories -- but every category task 4
# names: config, protocol descriptor, input (hashed separately, see
# RunManifest.input_hash), votes, raw responses, decisions, epoch,
# changelog, audit draw, audit labels, and the validation record.
MANIFEST_ARTIFACTS = (
    CONFIG_FILENAME,
    PROTOCOL_FILENAME,
    VOTES_FILENAME,
    RAW_RESPONSES_FILENAME,
    DECISIONS_FILENAME,
    EPOCH_FILENAME,
    CHANGELOG_FILENAME,
    AUDIT_DRAW_FILENAME,
    AUDIT_LABELS_FILENAME,
    VALIDATION_RECORD_FILENAME,
)


class StoreError(ValueError):
    """Raised when run-directory contents are missing, malformed, or inconsistent."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_input(path: str | Path) -> NormalizedInput:
    """Read an input-contract JSON file and validate/normalize it.

    Args:
        path: Path to a JSON file matching the input contract wire shape
            (`attest.contracts.input`).

    Returns:
        The validated, normalized `NormalizedInput`.

    Raises:
        StoreError: If `path` does not exist.
        ContractError: If the file's content violates the input contract.
    """
    payload = _read_json(Path(path))
    if payload is None:
        raise StoreError(f"input file '{path}' does not exist")
    return validate_and_normalize(payload)


def _vote_to_dict(vote: Vote) -> dict[str, Any]:
    return {"vendor": vote.vendor, "rating": vote.rating}


def _vote_from_dict(payload: Mapping[str, Any]) -> Vote:
    return Vote(vendor=payload["vendor"], rating=payload["rating"])


def _decision_to_dict(decision: Decision) -> dict[str, Any]:
    return {
        "auto_label": decision.auto_label,
        "escalate": decision.escalate,
        "dispersion": decision.dispersion,
        "boundary": decision.boundary,
    }


def _decision_from_dict(payload: Mapping[str, Any]) -> Decision:
    return Decision(
        auto_label=payload["auto_label"],
        escalate=payload["escalate"],
        dispersion=payload["dispersion"],
        boundary=payload["boundary"],
    )


def _config_to_dict(config: EnsembleConfig, ensemble_config_id: str) -> dict[str, Any]:
    payload = config.to_dict()
    payload["ensemble_config_id"] = ensemble_config_id
    # confidence_threshold is deliberately excluded from Config.to_dict()
    # itself (see that method's docstring) since it must never affect
    # ensemble_config_id -- added here instead, at the one place a
    # persisted config.json and a live Config object actually meet, so it
    # still round-trips through RunStore like every other field a caller
    # set on Config.
    payload["confidence_threshold"] = config.confidence_threshold
    return payload


def _config_from_dict(payload: Mapping[str, Any]) -> EnsembleConfig:
    raw_vendors: dict[str, Any] = payload["vendors"]
    vendors = {name: VendorSpec(**spec) for name, spec in raw_vendors.items()}
    return EnsembleConfig(
        vendors=vendors,
        aggregation=payload["aggregation"],
        tau=payload["tau"],
        batch_size=payload.get("batch_size", 0),
        default_prompt=payload.get("default_prompt"),
        track_prompts=dict(payload.get("track_prompts", {})),
        zero_policy=payload.get("zero_policy", ZERO_POLICY_ESCALATE),
        confidence_threshold=payload.get("confidence_threshold", DEFAULT_LOW_THRESHOLD),
    )


def _epoch_to_dict(epoch: Epoch) -> dict[str, Any]:
    return {
        "id": epoch.id,
        "ensemble_config_id": epoch.ensemble_config_id,
        "opened_at": epoch.opened_at.isoformat(),
    }


def _epoch_from_dict(payload: Mapping[str, Any]) -> Epoch:
    return Epoch(
        id=payload["id"],
        ensemble_config_id=payload["ensemble_config_id"],
        opened_at=datetime.fromisoformat(payload["opened_at"]),
    )


def _to_record_config(config: EnsembleConfig) -> RecordConfig:
    return RecordConfig(
        vendors=sorted(config.vendors),
        models={name: spec.model for name, spec in config.vendors.items()},
        prompts={name: spec.prompt_version for name, spec in config.vendors.items()},
        aggregation=config.aggregation,
        tau=config.tau,
        batch_size=config.batch_size,
        x=config.x,
        zero_policy=config.zero_policy,
    )


@dataclass
class RunStore:
    """A local run directory: idempotent JSON persistence for one epoch's artifacts.

    Every file in the directory is plain JSON; there is no database or
    object storage involved. `votes.json`, `decisions.json`, and
    `audit.json` are each stamped with the `ensemble_config_id` they were
    written under, and a write that would mix in data for a different id is
    refused rather than silently overwriting the stamp.

    Attributes:
        root: Directory the run's JSON files live under. Created on
            construction if it does not already exist.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_stamped(self, filename: str, key: str) -> tuple[str | None, dict[str, Any]]:
        payload = _read_json(self.root / filename)
        if payload is None:
            return None, {}
        ensemble_config_id: str | None = payload.get("ensemble_config_id")
        items: dict[str, Any] = payload.get(key, {})
        return ensemble_config_id, dict(items)

    def _write_stamped(
        self,
        filename: str,
        key: str,
        ensemble_config_id: str,
        items: Mapping[str, Any],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        existing_id, existing_items = self._read_stamped(filename, key)
        if existing_id is not None and existing_id != ensemble_config_id:
            raise StoreError(
                f"'{filename}' in run directory '{self.root}' already holds data for "
                f"ensemble_config_id '{existing_id}'; refusing to mix in data for "
                f"'{ensemble_config_id}'"
            )
        existing_items.update(items)
        payload: dict[str, Any] = {"ensemble_config_id": ensemble_config_id, key: existing_items}
        if extra:
            payload.update(extra)
        _write_json(self.root / filename, payload)

    def write_config(self, config: EnsembleConfig) -> str:
        """Persist the ensemble configuration in force for this run directory.

        Args:
            config: The ensemble configuration to persist.

        Returns:
            The content-derived `ensemble_config_id` of `config`.

        Raises:
            StoreError: If this run directory already holds a different
                ensemble configuration.
        """
        ensemble_config_id = compute_ensemble_config_id(config)
        path = self.root / CONFIG_FILENAME
        existing = _read_json(path)
        if existing is not None and existing.get("ensemble_config_id") != ensemble_config_id:
            raise StoreError(
                f"run directory '{self.root}' already holds a different ensemble "
                f"configuration (existing id '{existing.get('ensemble_config_id')}', "
                f"new id '{ensemble_config_id}')"
            )
        _write_json(path, _config_to_dict(config, ensemble_config_id))
        return ensemble_config_id

    def read_config(self) -> EnsembleConfig:
        """Read the ensemble configuration stored for this run directory.

        Raises:
            StoreError: If no configuration has been written yet.
        """
        payload = _read_json(self.root / CONFIG_FILENAME)
        if payload is None:
            raise StoreError(f"run directory '{self.root}' has no stored config")
        return _config_from_dict(payload)

    def write_epoch(self, epoch: Epoch) -> None:
        """Persist the stable epoch this run directory covers.

        Raises:
            StoreError: If this run directory already holds a different epoch.
        """
        path = self.root / EPOCH_FILENAME
        existing = _read_json(path)
        if existing is not None and existing.get("id") != epoch.id:
            raise StoreError(
                f"run directory '{self.root}' already holds a different epoch "
                f"(existing id '{existing.get('id')}', new id '{epoch.id}')"
            )
        _write_json(path, _epoch_to_dict(epoch))

    def read_epoch(self) -> Epoch:
        """Read the epoch stored for this run directory.

        Raises:
            StoreError: If no epoch has been written yet.
        """
        payload = _read_json(self.root / EPOCH_FILENAME)
        if payload is None:
            raise StoreError(f"run directory '{self.root}' has no stored epoch")
        return _epoch_from_dict(payload)

    def write_votes(self, votes: Sequence[VoteVector]) -> None:
        """Upsert raw vote vectors into `votes.json`, keyed by record id.

        Args:
            votes: Vote vectors to persist. All must share the same
                `ensemble_config_id`, which becomes (or must match) this
                file's stamp.

        Raises:
            StoreError: If `votes` mixes more than one `ensemble_config_id`,
                or contains one that conflicts with an existing stamp.
        """
        if not votes:
            return
        ids = {vv.ensemble_config_id for vv in votes}
        if len(ids) > 1:
            raise StoreError(f"votes batch stamps more than one ensemble_config_id: {sorted(ids)}")
        ensemble_config_id = next(iter(ids))
        items = {vv.record_id: [_vote_to_dict(v) for v in vv.votes] for vv in votes}
        self._write_stamped(VOTES_FILENAME, "votes", ensemble_config_id, items)

    def read_votes(self) -> list[VoteVector]:
        """Read all stored vote vectors, ordered by record id."""
        ensemble_config_id, items = self._read_stamped(VOTES_FILENAME, "votes")
        if ensemble_config_id is None:
            return []
        return [
            VoteVector(
                record_id=record_id,
                ensemble_config_id=ensemble_config_id,
                votes=tuple(_vote_from_dict(v) for v in raw_votes),
            )
            for record_id, raw_votes in sorted(items.items())
        ]

    def write_raw_responses(
        self, ensemble_config_id: str, raw_responses: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Upsert raw per-vendor rater responses into `raw_responses.json`, keyed by record id.

        Retained for audit and debugging, exactly as
        `attest.vendors.base.EnsembleRun.raw_responses` documents them --
        never part of the versioned vote contract in `votes.json`, and
        opaque to this store: each vendor's payload shape is whatever that
        vendor's `Rater.rate` returned.

        Args:
            ensemble_config_id: Configuration id these responses were
                produced under; must match this file's existing stamp, if any.
            raw_responses: Mapping of record id to a mapping of vendor name
                to that vendor's raw, implementation-specific response payload.
        """
        items = {record_id: dict(by_vendor) for record_id, by_vendor in raw_responses.items()}
        self._write_stamped(RAW_RESPONSES_FILENAME, "raw_responses", ensemble_config_id, items)

    def read_raw_responses(self) -> dict[str, dict[str, Any]]:
        """Read all stored raw per-vendor responses, keyed by record id."""
        _ensemble_config_id, items = self._read_stamped(RAW_RESPONSES_FILENAME, "raw_responses")
        return {record_id: dict(by_vendor) for record_id, by_vendor in items.items()}

    def write_decisions(self, ensemble_config_id: str, decisions: Mapping[str, Decision]) -> None:
        """Upsert aggregation decisions into `decisions.json`, keyed by record id.

        Args:
            ensemble_config_id: Configuration id these decisions were
                computed under; must match this file's existing stamp, if any.
            decisions: Mapping of record id to its `Decision`.
        """
        items = {record_id: _decision_to_dict(d) for record_id, d in decisions.items()}
        self._write_stamped(DECISIONS_FILENAME, "decisions", ensemble_config_id, items)

    def read_decisions(self) -> dict[str, Decision]:
        """Read all stored decisions, keyed by record id."""
        _ensemble_config_id, items = self._read_stamped(DECISIONS_FILENAME, "decisions")
        return {record_id: _decision_from_dict(d) for record_id, d in items.items()}

    def write_audit_rows(self, ensemble_config_id: str, rows: Sequence[AuditRow]) -> None:
        """Upsert random recall-audit rows into `audit.json`, keyed by record id.

        Args:
            ensemble_config_id: Configuration id these rows were drawn
                under; must match this file's existing stamp, if any.
            rows: The audit rows to persist, labeled or not.
        """
        items = {
            row.record_id: {
                "stratum": row.stratum,
                "human_label": row.human_label,
                "reviewer": row.reviewer,
                "blinded": row.blinded,
            }
            for row in rows
        }
        self._write_stamped(AUDIT_FILENAME, "audit_rows", ensemble_config_id, items)

    def read_audit_rows(self) -> list[AuditRow]:
        """Read all stored recall-audit rows, ordered by record id."""
        _ensemble_config_id, items = self._read_stamped(AUDIT_FILENAME, "audit_rows")
        return [
            AuditRow(
                record_id=record_id,
                stratum=row["stratum"],
                human_label=row["human_label"],
                reviewer=row.get("reviewer"),
                blinded=row.get("blinded"),
            )
            for record_id, row in sorted(items.items())
        ]

    def write_run_record(self, run: RunRecord) -> None:
        """Upsert a run's provenance record into `runs.json`, keyed by run id."""
        raw: list[dict[str, Any]] = _read_json(self.root / RUNS_FILENAME) or []
        by_id = {r["id"]: r for r in raw}
        by_id[run.id] = run.to_dict()
        _write_json(self.root / RUNS_FILENAME, [by_id[key] for key in sorted(by_id)])

    def read_run_records(self) -> list[RunRecord]:
        """Read all stored run records, ordered by run id."""
        raw: list[dict[str, Any]] = _read_json(self.root / RUNS_FILENAME) or []
        return [RunRecord.from_dict(r) for r in raw]

    def write_batch_handle(self, vendor: str, handle: Mapping[str, Any]) -> None:
        """Upsert one vendor's batch handle into `batch_handles.json`, keyed by vendor name.

        `handle` is persisted as an opaque, plain JSON-serializable dict --
        this store is agnostic to `attest.vendors.batch.BatchHandle`'s shape,
        so `attest.vendors.batch` (which depends on this module for
        persistence) never has to be imported back here.

        Args:
            vendor: Name of the vendor this handle belongs to.
            handle: The handle's content, e.g. `BatchHandle.to_dict()`.
        """
        existing: dict[str, Any] = _read_json(self.root / BATCH_HANDLES_FILENAME) or {}
        existing[vendor] = dict(handle)
        _write_json(self.root / BATCH_HANDLES_FILENAME, existing)

    def read_batch_handles(self) -> dict[str, dict[str, Any]]:
        """Read all persisted batch handles as plain dicts, keyed by vendor name."""
        payload: dict[str, Any] = _read_json(self.root / BATCH_HANDLES_FILENAME) or {}
        return {vendor: dict(handle) for vendor, handle in payload.items()}

    def write_tau_report(self, report: TauReport) -> None:
        """Persist the tau proof (see `attest.ensemble.tau`) for this run's current config.

        Overwrites any previously stored report -- there is exactly one
        current tau proof per run directory, not one per epoch, since it
        describes the currently loaded config's `tau`/`x`, not any
        particular batch of votes.
        """
        _write_json(self.root / TAU_REPORT_FILENAME, report.to_dict())

    def read_tau_report(self) -> TauReport | None:
        """Read the persisted tau proof, or None if `write_tau_report` was never called."""
        payload = _read_json(self.root / TAU_REPORT_FILENAME)
        if payload is None:
            return None
        return TauReport.from_dict(payload)

    def write_confidence_policy(self, *, low_threshold: float, min_supporting_votes: int) -> None:
        """Persist the confidence-tier policy used for a confidence-stratified audit draw.

        `low_threshold` (see `attest.ensemble.confidence.confidence_tier`) is
        a deliberate policy choice, not hash-versioned into
        `attest.provenance.config.Config`/`ensemble_config_id` -- without
        recording it here, a stored `stratum` of `"low"`/`"high"` on an
        audit row would carry no trace of which threshold produced it, and
        a later `validate` call would have no way to reconstruct matching
        per-tier population sizes. `min_supporting_votes` (see
        `attest.ensemble.confidence.MIN_SUPPORTING_VOTES`) is recorded
        alongside it for the same reason: it is currently a fixed constant,
        but a stored policy should self-document the coverage rule that was
        actually in force when it was written, not assume today's constant
        forever.

        Overwrites any previously stored policy -- like `write_tau_report`,
        there is exactly one current policy per run directory, describing
        the threshold last actually used, not a history of every value ever
        tried.
        """
        _write_json(
            self.root / CONFIDENCE_POLICY_FILENAME,
            {"low_threshold": low_threshold, "min_supporting_votes": min_supporting_votes},
        )

    def read_confidence_policy(self) -> dict[str, Any] | None:
        """Read the persisted confidence policy, or None if never written.

        A non-None result also signals that this run's most recent audit
        draw was confidence-stratified, which `attest.cli._cmd_validate`
        uses to decide whether to recompute confidence tiers when
        rebuilding population sizes for `attest.stats.recall`.
        """
        payload: dict[str, Any] | None = _read_json(self.root / CONFIDENCE_POLICY_FILENAME)
        return payload

    # --- changelog: append-only run artifact ------------------------------

    def write_changelog(self, changelog: ChangeLog) -> None:
        """Persist an append-only changelog, refusing to rewrite already-stored history.

        A stored changelog may only be *extended*: every previously written
        event must appear, unchanged, as a prefix of `changelog.events`.
        This is the mechanical enforcement of "append-only" -- once an event
        is on disk, it is permanent provenance, not an editable log line.

        Args:
            changelog: The changelog to persist; must extend (or equal) what
                is already stored.

        Raises:
            StoreError: If `changelog` would shrink or rewrite previously
                stored history.
        """
        existing_events = self.read_changelog().events
        new_events = list(changelog.events)
        if (
            len(new_events) < len(existing_events)
            or new_events[: len(existing_events)] != existing_events
        ):
            raise StoreError(
                f"changelog in run directory '{self.root}' is append-only; the given "
                "changelog does not extend the previously stored history"
            )
        _write_json(self.root / CHANGELOG_FILENAME, [e.to_dict() for e in new_events])

    def read_changelog(self) -> ChangeLog:
        """Read the changelog stored for this run directory, empty if never written."""
        raw: list[dict[str, Any]] = _read_json(self.root / CHANGELOG_FILENAME) or []
        return ChangeLog.from_list(raw)

    def append_change_event(self, event: ConfigChangeEvent) -> ChangeLog:
        """Append one change event to this run directory's persisted changelog.

        Args:
            event: The change event to append.

        Returns:
            The changelog after appending, as persisted.
        """
        changelog = self.read_changelog()
        changelog.events.append(event)
        self.write_changelog(changelog)
        return changelog

    # --- validation protocol -----------------------------------------------

    def write_protocol(self, protocol: ValidationProtocol) -> str:
        """Persist the validation protocol in force for this run directory.

        Unlike `write_config`, a run directory is not locked to one protocol
        forever: the audit design, adjudication protocol, sentinel
        thresholds, and reporting spec are pure analysis-plan choices (see
        `attest.provenance.protocol`'s module docstring) that may legitimately
        be revised without opening a new epoch. Each call simply overwrites
        the stored protocol.

        Args:
            protocol: The validation protocol to persist.

        Returns:
            The content-derived `protocol_id` of `protocol`.
        """
        protocol_id = compute_protocol_id(protocol)
        payload = protocol.to_dict()
        payload["protocol_id"] = protocol_id
        _write_json(self.root / PROTOCOL_FILENAME, payload)
        return protocol_id

    def read_protocol(self) -> ValidationProtocol | None:
        """Read the validation protocol stored for this run directory, or None if never written."""
        payload = _read_json(self.root / PROTOCOL_FILENAME)
        if payload is None:
            return None
        return ValidationProtocol.from_dict(payload)

    def read_protocol_id(self) -> str | None:
        """Read the stored protocol's content-derived id, or None if never written."""
        payload = _read_json(self.root / PROTOCOL_FILENAME)
        if payload is None:
            return None
        protocol_id: str | None = payload.get("protocol_id")
        return protocol_id

    # --- audit draw/labels: immutable snapshots for manifest integrity -----

    def write_audit_draw(
        self,
        ensemble_config_id: str,
        rows: Sequence[AuditRow],
        *,
        sampling_frame_hash: str | None = None,
    ) -> None:
        """Upsert the drawn (pre-label) recall-audit rows into `audit_draw.json`.

        A snapshot distinct from `audit.json` (which `assemble_validation_record`
        reads and which `audit-apply` later upserts labels into): this file
        exists so the manifest can hash "what was drawn" and "what was
        labeled" (`write_audit_labels`) as two separately verifiable
        artifacts, matching the manuscript's own artifact list, without
        changing `audit.json`'s existing combined read path.

        Args:
            ensemble_config_id: Configuration id these rows were drawn under.
            rows: The freshly drawn (unlabeled) rows.
            sampling_frame_hash: This draw's
                `attest.planes.recall_audit.population_frame_hash`, if
                available -- the eligible population's content hash,
                recorded so the sampling frame this draw ran against is
                independently verifiable later. Once written for a given
                `ensemble_config_id`, later draws under the same id must
                supply the identical hash (the frame is fixed per epoch);
                a mismatch raises `StoreError`.

        Raises:
            StoreError: If `sampling_frame_hash` conflicts with a hash
                already stored for this run directory.
        """
        items = {row.record_id: row.stratum for row in rows}
        existing_hash = self.read_audit_sampling_frame_hash()
        if (
            sampling_frame_hash is not None
            and existing_hash is not None
            and sampling_frame_hash != existing_hash
        ):
            raise StoreError(
                f"run directory '{self.root}' already recorded sampling_frame_hash "
                f"'{existing_hash}' for a prior draw; refusing to record a different hash "
                f"'{sampling_frame_hash}' for the same ensemble_config_id -- the screen-excluded "
                "population must not change within an epoch's audit draws"
            )
        extra = {"sampling_frame_hash": sampling_frame_hash or existing_hash}
        self._write_stamped(AUDIT_DRAW_FILENAME, "drawn", ensemble_config_id, items, extra=extra)

    def read_audit_draw(self) -> dict[str, str]:
        """Read the stored audit-draw snapshot, mapping record id to stratum."""
        _ensemble_config_id, items = self._read_stamped(AUDIT_DRAW_FILENAME, "drawn")
        return {record_id: stratum for record_id, stratum in items.items()}

    def read_audit_sampling_frame_hash(self) -> str | None:
        """Read the stored draw's sampling-frame hash, or None if never recorded."""
        payload = _read_json(self.root / AUDIT_DRAW_FILENAME)
        if payload is None:
            return None
        frame_hash: str | None = payload.get("sampling_frame_hash")
        return frame_hash

    def write_audit_labels(self, ensemble_config_id: str, labels: Mapping[str, int]) -> None:
        """Upsert applied human gold-check labels into `audit_labels.json`.

        Args:
            ensemble_config_id: Configuration id in force when these labels
                were applied.
            labels: Mapping of record id to the human ordinal audit label
                actually applied by `audit-apply`.
        """
        self._write_stamped(AUDIT_LABELS_FILENAME, "labels", ensemble_config_id, dict(labels))

    def read_audit_labels(self) -> dict[str, int]:
        """Read the stored audit-labels snapshot, mapping record id to human label."""
        _ensemble_config_id, items = self._read_stamped(AUDIT_LABELS_FILENAME, "labels")
        return {record_id: int(label) for record_id, label in items.items()}

    # --- validation record snapshot -----------------------------------------

    def write_validation_record_snapshot(self, payload: Mapping[str, Any]) -> None:
        """Persist a copy of the assembled validation record inside the run directory.

        `attest validate --out PATH` may write its human-facing copy
        anywhere (e.g. a paper repo's `results/`); this snapshot is the
        run-directory-local copy the manifest hashes as the "validation
        record" artifact, so integrity verification does not depend on
        wherever `--out` happened to point.
        """
        _write_json(self.root / VALIDATION_RECORD_FILENAME, dict(payload))

    def read_validation_record_snapshot(self) -> dict[str, Any] | None:
        """Read the stored validation-record snapshot, or None if never written."""
        payload: dict[str, Any] | None = _read_json(self.root / VALIDATION_RECORD_FILENAME)
        return payload

    # --- sentinel drift ------------------------------------------------------

    def write_sentinel_baseline(self, baseline: SentinelBaseline) -> None:
        """Persist the frozen sentinel set's baseline ratings for this run directory.

        Raises:
            StoreError: If a baseline for a *different* sentinel set is
                already stored -- a run directory's sentinel baseline is
                pinned to one frozen set, like `write_config` pins one
                ensemble configuration.
        """
        existing = _read_json(self.root / SENTINEL_BASELINE_FILENAME)
        if existing is not None and existing.get("sentinel_set_id") != baseline.sentinel_set_id:
            raise StoreError(
                f"run directory '{self.root}' already holds a sentinel baseline for a "
                f"different sentinel set (existing '{existing.get('sentinel_set_id')}', new "
                f"'{baseline.sentinel_set_id}')"
            )
        _write_json(self.root / SENTINEL_BASELINE_FILENAME, baseline.to_dict())

    def read_sentinel_baseline(self) -> SentinelBaseline | None:
        """Read the stored sentinel baseline, or None if never written."""
        payload = _read_json(self.root / SENTINEL_BASELINE_FILENAME)
        if payload is None:
            return None
        return SentinelBaseline.from_dict(payload)

    def write_sentinel_evaluation(self, evaluation: SentinelEvaluation) -> None:
        """Append one sentinel evaluation to this run directory's evaluation history.

        Every check is appended, not upserted -- the sentinel is run
        periodically against the same baseline, and the full history of
        readings (not just the latest) is itself provenance.
        """
        raw: list[dict[str, Any]] = _read_json(self.root / SENTINEL_EVALUATIONS_FILENAME) or []
        raw.append(evaluation.to_dict())
        _write_json(self.root / SENTINEL_EVALUATIONS_FILENAME, raw)

    def read_sentinel_evaluations(self) -> list[SentinelEvaluation]:
        """Read every stored sentinel evaluation, oldest first."""
        raw: list[dict[str, Any]] = _read_json(self.root / SENTINEL_EVALUATIONS_FILENAME) or []
        return [SentinelEvaluation.from_dict(e) for e in raw]

    # --- adjudication and active-learning provenance --------------------------

    def write_adjudication_record(self, item: AdjudicationItem) -> None:
        """Upsert one adjudication item's reviewer/selection provenance, keyed by record id.

        Distinct from `write_decisions` (the authoritative final label used
        downstream by `assemble_validation_record`): this file exists purely
        for the auditable reviewer trail -- who resolved this escalation,
        when, under which protocol, and why it escalated in the first place.
        """
        raw: dict[str, Any] = _read_json(self.root / ADJUDICATION_RECORDS_FILENAME) or {}
        raw[item.record_id] = {
            "record_id": item.record_id,
            "ensemble_config_id": item.ensemble_config_id,
            "dispersion": item.dispersion,
            "boundary": item.boundary,
            "selection_reason": item.selection_reason,
            "human_label": item.human_label,
            "reviewer": item.reviewer,
            "resolved_at": item.resolved_at.isoformat() if item.resolved_at is not None else None,
            "protocol_id": item.protocol_id,
        }
        _write_json(self.root / ADJUDICATION_RECORDS_FILENAME, raw)

    def read_adjudication_records(self) -> dict[str, dict[str, Any]]:
        """Read every stored adjudication provenance record, keyed by record id."""
        return _read_json(self.root / ADJUDICATION_RECORDS_FILENAME) or {}

    def write_active_learning_selections(
        self, ensemble_config_id: str, selections: Sequence[ActiveLearningSelection]
    ) -> None:
        """Upsert active-learning selections' provenance, keyed by record id."""
        items = {
            s.record_id: {
                "dispersion": s.dispersion,
                "boundary": s.boundary,
                "selection_reason": list(s.selection_reason),
                "confidence_scored": s.confidence.scored,
                "confidence_median_probability": s.confidence.median_probability,
            }
            for s in selections
        }
        self._write_stamped(
            ACTIVE_LEARNING_SELECTIONS_FILENAME, "selections", ensemble_config_id, items
        )

    def read_active_learning_selections(self) -> dict[str, dict[str, Any]]:
        """Read stored active-learning selection provenance, keyed by record id."""
        _ensemble_config_id, items = self._read_stamped(
            ACTIVE_LEARNING_SELECTIONS_FILENAME, "selections"
        )
        return items

    def write_active_learning_review(self, review: ActiveLearningReview) -> None:
        """Append one human review of an active-learning selection, keyed by record id.

        Appended, not upserted: the same selection may legitimately be
        reviewed more than once over a project's lifetime (e.g. a second
        pass after a prompt change), and each review is its own provenance
        record.
        """
        raw: list[dict[str, Any]] = _read_json(self.root / ACTIVE_LEARNING_REVIEWS_FILENAME) or []
        raw.append(
            {
                "record_id": review.record_id,
                "ensemble_config_id": review.ensemble_config_id,
                "selection_reason": list(review.selection_reason),
                "reviewer": review.reviewer,
                "reviewed_at": review.reviewed_at.isoformat(),
                "protocol_id": review.protocol_id,
                "notes": review.notes,
            }
        )
        _write_json(self.root / ACTIVE_LEARNING_REVIEWS_FILENAME, raw)

    def read_active_learning_reviews(self) -> list[dict[str, Any]]:
        """Read every stored active-learning review, oldest first."""
        return _read_json(self.root / ACTIVE_LEARNING_REVIEWS_FILENAME) or []

    # --- run manifest and offline integrity verification -----------------------

    def write_manifest(
        self,
        *,
        ensemble_config_id: str,
        protocol_id: str | None = None,
        input_hash: str | None = None,
        input_source: str | None = None,
        seeds: Mapping[str, int] | None = None,
        sdk_versions: Mapping[str, str] | None = None,
    ) -> RunManifest:
        """Build and persist a `RunManifest` hashing this run directory's current artifacts.

        Hashes exactly the artifact set named in `MANIFEST_ARTIFACTS` (see
        that constant's docstring for why), skipping any not yet produced.
        Overwrites any previously stored manifest -- like `tau_report.json`,
        there is exactly one current manifest per run directory, describing
        the artifacts as they stand right now.

        Args:
            ensemble_config_id: The screening configuration this run executed under.
            protocol_id: The validation protocol this run executed under, if any.
            input_hash: SHA-256 hex digest of the input file screened.
            input_source: Free-text/path identifying the input file.
            seeds: Named random seeds used by this run.
            sdk_versions: Mapping of vendor name to its provider SDK's
                installed version (see
                `attest.vendors.sdk_versions.sdk_versions`).

        Returns:
            The freshly built and persisted `RunManifest`.
        """
        manifest = build_manifest(
            root=self.root,
            artifact_filenames=MANIFEST_ARTIFACTS,
            ensemble_config_id=ensemble_config_id,
            protocol_id=protocol_id,
            input_hash=input_hash,
            input_source=input_source,
            seeds=seeds,
            sdk_versions=sdk_versions,
        )
        _write_json(self.root / MANIFEST_FILENAME, manifest.to_dict())
        return manifest

    def read_manifest(self) -> RunManifest | None:
        """Read the stored manifest, or None if `write_manifest` was never called."""
        payload = _read_json(self.root / MANIFEST_FILENAME)
        if payload is None:
            return None
        return RunManifest.from_dict(payload)

    def verify(self) -> ManifestIntegrityReport:
        """Offline-verify this run directory's artifacts against its stored manifest.

        Returns:
            An `attest.provenance.manifest.IntegrityReport`: `ok=True` iff
            every artifact the manifest recorded a hash for is present and
            byte-identical to when the manifest was built.

        Raises:
            StoreError: If no manifest has been written for this run directory.
        """
        manifest = self.read_manifest()
        if manifest is None:
            raise StoreError(
                f"run directory '{self.root}' has no stored manifest to verify against"
            )
        return verify_artifacts(self.root, manifest)


def hash_input_file(path: str | Path) -> str | None:
    """Return the SHA-256 hex digest of an input-contract file's raw bytes, for manifest provenance.

    Hashes the file the caller loaded (e.g. via `load_input`) without
    duplicating its (potentially large) content into the run directory --
    only the hash is recorded (see `attest.provenance.manifest.RunManifest.input_hash`).

    Returns:
        The digest, or None if `path` does not exist.
    """
    return hash_file(Path(path))


def _final_labels(
    decisions: Mapping[str, Decision], human_labels: Mapping[str, int]
) -> tuple[dict[str, int], list[str]]:
    """Resolve every decision to its final label, tracking which escalations could not be resolved.

    Returns:
        A `(resolved, unresolved_ids)` pair: `resolved` maps record id to
        final label for every decision that could be resolved; `unresolved_ids`
        lists, in decision order, the ids of escalated decisions with no
        entry in `human_labels` -- callers must not treat `resolved` as the
        complete decision set without accounting for these.
    """
    resolved: dict[str, int] = {}
    unresolved: list[str] = []
    for record_id, decision in decisions.items():
        try:
            resolved[record_id] = final_label(record_id, decision, human_labels.get(record_id))
        except AdjudicationError:
            unresolved.append(record_id)
    return resolved, unresolved


def assemble_validation_record(
    store: RunStore,
    *,
    prefilter_prisma: PrefilterPrisma,
    truths: Mapping[str, int],
    population_sizes: Mapping[str, int],
    human_labels: Mapping[str, int] | None = None,
    confidence: float = 0.95,
    allow_unresolved_escalations: bool = False,
) -> ValidationRecord:
    """Assemble a full `ValidationRecord` for an epoch from a completed run directory.

    Seeds the PRISMA counts from `prefilter_prisma`, then fills in
    everything the stats modules can compute from the run's stored votes,
    decisions, and audit rows: inter-rater agreement (overall and pairwise),
    conditional false-negative correlation, escalation rate, the confusion
    matrix against gold labels, and -- when the run's audit rows include at
    least one gold-checked record -- stratified recall with its
    rule-of-three worst-case floor.

    Args:
        store: A `RunStore` that already holds a written config, epoch, and
            votes for the epoch being reported on.
        prefilter_prisma: The completed prefilter run's PRISMA counts,
            seeding `identified`, `duplicates_removed`, `after_dedup`,
            `prefilter_excluded`, and `screened`.
        truths: Mapping of record id to gold ordinal label, used for error
            correlation, the confusion matrix, and recall's true-positive count.
        population_sizes: Mapping of audit stratum name to the total
            screen-excluded population size for that stratum, passed
            through to `attest.planes.recall_audit.build_strata`.
        human_labels: Authoritative human labels for escalated decisions,
            keyed by record id. A stored decision that escalated and has no
            entry here is unresolved: by default this raises `StoreError`,
            since the manuscript's true-positive definition (methods §2.9)
            requires every include-and-escalate record to be either
            adjudicated or covered by a separate inclusion audit -- a
            validation record built over silently-dropped escalations does
            not support that definition.
        confidence: Two-sided confidence level for the recall floor and
            interval.
        allow_unresolved_escalations: If True, proceed even when some
            escalated decisions have no entry in `human_labels`, omitting
            them from the confusion matrix and PRISMA screen_excluded/included
            counts as before. The omitted count is still reported in the
            returned record's `unresolved_escalations` field so this choice
            is never silent in the output. Default False: fail closed.

    Returns:
        A `ValidationRecord` with every field the stored run supports filled in.

    Raises:
        StoreError: If the run directory has no stored votes, its stored
            epoch was opened for a different ensemble configuration than
            the one currently stored, or (unless `allow_unresolved_escalations`
            is set) one or more escalated decisions have no resolved
            human label.
    """
    config = store.read_config()
    epoch = store.read_epoch()
    votes = store.read_votes()
    decisions = store.read_decisions()
    audit_rows = store.read_audit_rows()

    if not votes:
        raise StoreError(f"run directory '{store.root}' has no stored votes to assemble from")

    ensemble_config_id = compute_ensemble_config_id(config)
    if epoch.ensemble_config_id != ensemble_config_id:
        raise StoreError(
            f"stored epoch '{epoch.id}' was opened for ensemble_config_id "
            f"'{epoch.ensemble_config_id}', which does not match the stored config's "
            f"current id '{ensemble_config_id}'"
        )

    record = build_validation_record(
        ensemble_config_id=ensemble_config_id,
        epoch=epoch.id,
        config=_to_record_config(config),
        prefilter_prisma=prefilter_prisma,
    )

    overall = agreement_report(votes)
    record.agreement = Agreement(krippendorff_alpha=overall.alpha, pairwise=pairwise_alpha(votes))

    predictions_by_vendor, ordered_truths = build_predictions_by_vendor(votes, truths)
    correlations = pairwise_fn_correlation(predictions_by_vendor, ordered_truths)
    record.error_correlation = ErrorCorrelation(
        pairwise_fn_on_relevant={
            key: PairwiseFnCorrelation(
                correlation=result.correlation,
                n=result.n,
                both=result.both,
                only_a=result.only_a,
                only_b=result.only_b,
                neither=result.neither,
            )
            for key, result in correlations.items()
        }
    )

    if decisions:
        record.escalation_rate = sum(1 for d in decisions.values() if d.escalate) / len(decisions)

    resolved_labels, unresolved_ids = _final_labels(decisions, human_labels or {})
    record.unresolved_escalations = len(unresolved_ids)
    if unresolved_ids and not allow_unresolved_escalations:
        shown = ", ".join(sorted(unresolved_ids)[:10])
        more = f" (+{len(unresolved_ids) - 10} more)" if len(unresolved_ids) > 10 else ""
        raise StoreError(
            f"{len(unresolved_ids)} escalated decision(s) have no resolved human label: "
            f"{shown}{more} -- resolve them (e.g. via 'attest adjudicate' and "
            "human_labels/read_adjudication_records) before validating, or pass "
            "allow_unresolved_escalations=True to explicitly accept that they will be "
            "excluded from the confusion matrix, PRISMA counts, and recall's TP"
        )
    record.prisma.screen_excluded = sum(
        1 for label in resolved_labels.values() if label != RELEVANT_LABEL
    )
    record.prisma.included = sum(1 for label in resolved_labels.values() if label == RELEVANT_LABEL)
    matrix = _confusion_matrix(resolved_labels, truths)
    record.confusion = {"tp": matrix.tp, "fp": matrix.fp, "fn": matrix.fn, "tn": matrix.tn}

    labeled_audit_rows = [row for row in audit_rows if row.human_label is not None]
    if labeled_audit_rows:
        strata = build_strata(labeled_audit_rows, population_sizes)
        true_positives = record.confusion["tp"]
        estimate = stratified_recall(strata, true_positives, confidence=confidence)
        audit_n = sum(s.n for s in strata)
        audited_population = sum(population_sizes.get(s.name, 0) for s in strata)
        record.recall = Recall(
            point=estimate.point,
            floor=estimate.floor,
            exact_floor=estimate.exact_floor,
            ci=estimate.ci,
            audit_n=audit_n,
            audit_budget_note=f"{audit_n} of {audited_population} screen-excluded records audited",
        )

    return record


__all__ = [
    "ACTIVE_LEARNING_REVIEWS_FILENAME",
    "ACTIVE_LEARNING_SELECTIONS_FILENAME",
    "ADJUDICATION_RECORDS_FILENAME",
    "AUDIT_DRAW_FILENAME",
    "AUDIT_FILENAME",
    "AUDIT_LABELS_FILENAME",
    "BATCH_HANDLES_FILENAME",
    "CHANGELOG_FILENAME",
    "CONFIDENCE_POLICY_FILENAME",
    "CONFIG_FILENAME",
    "DECISIONS_FILENAME",
    "EPOCH_FILENAME",
    "MANIFEST_ARTIFACTS",
    "MANIFEST_FILENAME",
    "PROTOCOL_FILENAME",
    "RAW_RESPONSES_FILENAME",
    "RUNS_FILENAME",
    "SENTINEL_BASELINE_FILENAME",
    "SENTINEL_EVALUATIONS_FILENAME",
    "TAU_REPORT_FILENAME",
    "VALIDATION_RECORD_FILENAME",
    "VOTES_FILENAME",
    "RunStore",
    "StoreError",
    "assemble_validation_record",
    "hash_input_file",
    "load_input",
]
