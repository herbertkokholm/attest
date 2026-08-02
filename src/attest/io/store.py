"""Local, storage-agnostic persistence: a JSON run directory, no database or object storage.

This module is the only place in the kernel that touches a filesystem. It
reads an input-contract payload into a `NormalizedInput`, persists one
epoch's raw votes, decisions, config, audit sample, and run records as plain
JSON files under a run directory, and assembles a full
`attest.contracts.validation_record.ValidationRecord` from that stored
state. Writes are idempotent: writing the same content twice yields the same
file, and re-running a write for the same record id upserts it rather than
duplicating it. Every epoch-scoped artifact (votes, decisions, audit rows) is
stamped with the `ensemble_config_id` it was produced under, and a run
directory refuses to mix data stamped with two different ids.
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
    Recall,
    ValidationRecord,
)
from attest.contracts.validation_record import Config as RecordConfig
from attest.contracts.validation_record import build as build_validation_record
from attest.ensemble.aggregate import Decision
from attest.ensemble.votes import Vote, VoteVector
from attest.planes.adjudication import AdjudicationError, final_label
from attest.planes.recall_audit import AuditRow, build_strata
from attest.prefilter.framework import Prisma as PrefilterPrisma
from attest.provenance.config import Config as EnsembleConfig
from attest.provenance.config import VendorSpec, compute_ensemble_config_id
from attest.provenance.epochs import Epoch
from attest.provenance.runs import RunRecord
from attest.stats.agreement import agreement_report, pairwise_alpha
from attest.stats.correlation import pairwise_fn_correlation
from attest.stats.recall import stratified_recall

_RELEVANT_LABEL = 1

CONFIG_FILENAME = "config.json"
EPOCH_FILENAME = "epoch.json"
VOTES_FILENAME = "votes.json"
DECISIONS_FILENAME = "decisions.json"
AUDIT_FILENAME = "audit.json"
RUNS_FILENAME = "runs.json"


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
    return payload


def _config_from_dict(payload: Mapping[str, Any]) -> EnsembleConfig:
    raw_vendors: dict[str, Any] = payload["vendors"]
    vendors = {name: VendorSpec(**spec) for name, spec in raw_vendors.items()}
    return EnsembleConfig(vendors=vendors, aggregation=payload["aggregation"], tau=payload["tau"])


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
        x=config.x,
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
        self, filename: str, key: str, ensemble_config_id: str, items: Mapping[str, Any]
    ) -> None:
        existing_id, existing_items = self._read_stamped(filename, key)
        if existing_id is not None and existing_id != ensemble_config_id:
            raise StoreError(
                f"'{filename}' in run directory '{self.root}' already holds data for "
                f"ensemble_config_id '{existing_id}'; refusing to mix in data for "
                f"'{ensemble_config_id}'"
            )
        existing_items.update(items)
        _write_json(
            self.root / filename, {"ensemble_config_id": ensemble_config_id, key: existing_items}
        )

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
            row.record_id: {"stratum": row.stratum, "human_label": row.human_label} for row in rows
        }
        self._write_stamped(AUDIT_FILENAME, "audit_rows", ensemble_config_id, items)

    def read_audit_rows(self) -> list[AuditRow]:
        """Read all stored recall-audit rows, ordered by record id."""
        _ensemble_config_id, items = self._read_stamped(AUDIT_FILENAME, "audit_rows")
        return [
            AuditRow(record_id=record_id, stratum=row["stratum"], human_label=row["human_label"])
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


def _predictions_by_vendor(
    votes: Sequence[VoteVector], truths: Mapping[str, int]
) -> tuple[dict[str, list[int]], list[int]]:
    """Build per-vendor prediction lists and the matching truths list.

    Restricted to gold-labeled records that every vendor present in `votes`
    cast a vote on, since `pairwise_fn_correlation` requires equal-length,
    aligned sequences per vendor.
    """
    by_record: dict[str, dict[str, int]] = {
        vv.record_id: {vote.vendor: vote.rating for vote in vv.votes} for vv in votes
    }
    vendors = sorted({vendor for ratings in by_record.values() for vendor in ratings})
    gold_ids = sorted(
        record_id
        for record_id in truths
        if record_id in by_record and all(vendor in by_record[record_id] for vendor in vendors)
    )
    predictions = {
        vendor: [by_record[record_id][vendor] for record_id in gold_ids] for vendor in vendors
    }
    ordered_truths = [truths[record_id] for record_id in gold_ids]
    return predictions, ordered_truths


def _final_labels(
    decisions: Mapping[str, Decision], human_labels: Mapping[str, int]
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for record_id, decision in decisions.items():
        try:
            resolved[record_id] = final_label(record_id, decision, human_labels.get(record_id))
        except AdjudicationError:
            continue
    return resolved


def _confusion_matrix(final_labels: Mapping[str, int], truths: Mapping[str, int]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for record_id, label in final_labels.items():
        if record_id not in truths:
            continue
        predicted_positive = label == _RELEVANT_LABEL
        actual_positive = truths[record_id] == _RELEVANT_LABEL
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def assemble_validation_record(
    store: RunStore,
    *,
    prefilter_prisma: PrefilterPrisma,
    truths: Mapping[str, int],
    population_sizes: Mapping[str, int],
    human_labels: Mapping[str, int] | None = None,
    confidence: float = 0.95,
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
            entry here is left out of the confusion matrix and PRISMA
            screen_excluded/included counts, since it is not yet resolved.
        confidence: Two-sided confidence level for the recall floor and
            interval.

    Returns:
        A `ValidationRecord` with every field the stored run supports filled in.

    Raises:
        StoreError: If the run directory has no stored votes, or its stored
            epoch was opened for a different ensemble configuration than
            the one currently stored.
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

    predictions_by_vendor, ordered_truths = _predictions_by_vendor(votes, truths)
    correlations = pairwise_fn_correlation(predictions_by_vendor, ordered_truths)
    record.error_correlation = ErrorCorrelation(
        pairwise_fn_on_relevant={
            key: result.correlation
            for key, result in correlations.items()
            if result.correlation is not None
        }
    )

    if decisions:
        record.escalation_rate = sum(1 for d in decisions.values() if d.escalate) / len(decisions)

    resolved_labels = _final_labels(decisions, human_labels or {})
    record.prisma.screen_excluded = sum(
        1 for label in resolved_labels.values() if label != _RELEVANT_LABEL
    )
    record.prisma.included = sum(
        1 for label in resolved_labels.values() if label == _RELEVANT_LABEL
    )
    record.confusion = _confusion_matrix(resolved_labels, truths)

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
            ci=estimate.ci,
            audit_n=audit_n,
            audit_budget_note=f"{audit_n} of {audited_population} screen-excluded records audited",
        )

    return record


__all__ = [
    "CONFIG_FILENAME",
    "AUDIT_FILENAME",
    "DECISIONS_FILENAME",
    "EPOCH_FILENAME",
    "RUNS_FILENAME",
    "VOTES_FILENAME",
    "RunStore",
    "StoreError",
    "assemble_validation_record",
    "load_input",
]
