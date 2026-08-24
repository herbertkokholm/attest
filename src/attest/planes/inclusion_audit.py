"""Random inclusion audit plane: the only plane from which TP may be statistically estimated.

Mirrors `attest.planes.recall_audit` for the other side of the manuscript's
recall estimand (Eq. 2/3, Section 2.9): where the recall audit draws a
probability sample of the screen-excluded population to estimate FN, this
plane draws a probability sample of the include-and-escalate population --
"records the ensemble did not exclude" -- to estimate TP, for the case
where that set is too large to review in full. `build_inclusion_strata`
enforces the same firewall as `recall_audit.build_strata`: any row not
tagged `PLANE_INCLUSION_AUDIT` is refused outright, not silently ignored.

The include-and-escalate population is a screening-time property (whether
the ensemble's aggregation rule excluded a record or not), not an
adjudication-time one, so this plane's draw does not need to wait for
escalations to be resolved -- it can run as soon as `attest screen` has
written decisions, in parallel with `attest adjudicate`.

When the include-and-escalate set is fully reviewed instead (TP known
exactly, with no sampling error), this plane is not used at all --
`attest.io.store.assemble_validation_record` falls back to treating
`confusion["tp"]` as exact, exactly as it always has.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from attest.ensemble.votes import VALID_RATINGS
from attest.planes import PLANE_INCLUSION_AUDIT
from attest.planes._apportionment import allocate_proportionally
from attest.planes.recall_audit import AuditError, AuditPlaneRow
from attest.stats.recall import Stratum

_UNSTRATIFIED_NAME = "all"


STATUS_INCLUDED = "included"
STATUS_ESCALATED = "escalated"


@dataclass(frozen=True)
class IncludedRecord:
    """One include-or-escalate record eligible for the random inclusion audit.

    Attributes:
        record_id: Id of the record.
        track: The screening track this record belongs to, used as the
            stratification key when `stratify_by_track` is set.
        status: Whether screening auto-labeled this record relevant
            (`"included"`) or escalated it (`"escalated"`), used as the
            stratification key when `stratify_by_status` is set. These two
            subpopulations plausibly have different true-relevance rates by
            construction -- `"included"` passed the ensemble's confident
            aggregation rule, `"escalated"` is exactly the disagreement/
            low-confidence case that rule could not resolve -- so pooling
            them into one SRS draw is valid (unbiased regardless of
            subgroup heterogeneity) but not efficient; stratifying by
            `status` lets a draw allocate proportionally across the two
            instead of by chance.
    """

    record_id: str
    track: int | str
    status: str = STATUS_INCLUDED

    def __post_init__(self) -> None:
        if self.status not in (STATUS_INCLUDED, STATUS_ESCALATED):
            raise AuditError(
                f"record '{self.record_id}': status must be '{STATUS_INCLUDED}' or "
                f"'{STATUS_ESCALATED}', got {self.status!r}"
            )


@dataclass(frozen=True)
class InclusionAuditRow:
    """One record drawn into the random inclusion audit.

    Structurally identical to `attest.planes.recall_audit.AuditRow` (both
    satisfy `AuditPlaneRow`), but stamped `PLANE_INCLUSION_AUDIT` instead --
    a separate type rather than a `plane` constructor argument on `AuditRow`,
    so that a row's plane is fixed by which dataclass it is, not by a value
    a caller could pass incorrectly.

    Attributes:
        record_id: Id of the drawn record.
        stratum: Stratum name this row was drawn from.
        human_label: Ordinal audit label from gold-checking, or None before
            the row has been audited.
        reviewer: Id or pseudonym of the human who gold-checked this row,
            or None while pending or when not supplied.
        blinded: Whether the reviewer gold-checked this row without seeing
            the ensemble's include-or-escalate decision for it, or None
            when not recorded. Provenance only, as in `AuditRow`.
        plane: Fixed to `PLANE_INCLUSION_AUDIT`; the tag
            `build_inclusion_strata` checks to refuse rows that did not
            come from this plane.
    """

    record_id: str
    stratum: str
    human_label: int | None = None
    reviewer: str | None = None
    blinded: bool | None = None
    plane: str = field(default=PLANE_INCLUSION_AUDIT, init=False)


def _track_key(record: IncludedRecord) -> str:
    return str(record.track)


def _status_key(record: IncludedRecord) -> str:
    return record.status


def draw_inclusion_audit_sample(
    population: Sequence[IncludedRecord],
    n: int,
    *,
    stratify_by_track: bool = False,
    stratify_by_status: bool = False,
    rng: random.Random | None = None,
) -> list[InclusionAuditRow]:
    """Draw a random sample of n include-or-escalate records for the inclusion audit.

    Without stratification, `n` records are drawn uniformly at random from
    the whole population, all tagged into a single pooled stratum. With
    `stratify_by_track=True` (or `stratify_by_status=True`), `n` is
    allocated across per-track (or per-status) strata in proportion to each
    stratum's population size (the same largest-remainder apportionment
    `recall_audit.draw_audit_sample` uses, via the shared
    `attest.planes._apportionment.allocate_proportionally`), then that many
    records are drawn uniformly at random within each stratum.

    `stratify_by_status` groups by `IncludedRecord.status`
    (`"included"`/`"escalated"`) rather than `track` -- see `status`'s
    docstring for why these two subpopulations are worth stratifying by:
    pooling them (the default) remains statistically valid, just not as
    efficient as it could be. Mutually exclusive with `stratify_by_track`,
    matching `recall_audit.draw_audit_sample`'s existing
    `stratify_by_track`/`stratify_by_confidence` exclusivity -- combined
    stratification across both keys is the same open design question noted
    there, not yet supported here either.

    Unlike `recall_audit.draw_audit_sample`, there is no
    `stratify_by_confidence` option here: ensemble confidence tiers were
    defined around the exclusion decision's escalation boundary and have no
    established analog on the inclusion side yet -- left open for a future
    extension, not implemented here.

    Args:
        population: The include-or-escalate records eligible for audit.
        n: Total number of records to draw.
        stratify_by_track: Whether to stratify the draw by `track`.
        stratify_by_status: Whether to stratify the draw by `status`
            instead. Mutually exclusive with `stratify_by_track`.
        rng: Source of randomness; defaults to a fresh `random.Random()`.
            Pass a seeded `random.Random` for a reproducible draw.

    Returns:
        The drawn rows, unlabeled (`human_label` is None), each tagged
        `PLANE_INCLUSION_AUDIT`.

    Raises:
        AuditError: If `n` is not positive or exceeds the population size,
            or if both `stratify_by_track` and `stratify_by_status` are set.
    """
    if n <= 0:
        raise AuditError(f"audit sample size n must be positive, got {n}")
    if n > len(population):
        raise AuditError(f"audit sample size n ({n}) exceeds population size ({len(population)})")
    if stratify_by_track and stratify_by_status:
        raise AuditError(
            "stratify_by_track and stratify_by_status cannot both be set: combined "
            "stratification across both keys is not yet supported -- choose one"
        )

    active_rng = rng if rng is not None else random.Random()

    if not stratify_by_track and not stratify_by_status:
        sampled = active_rng.sample(list(population), n)
        return [
            InclusionAuditRow(record_id=r.record_id, stratum=_UNSTRATIFIED_NAME) for r in sampled
        ]

    key_fn = _status_key if stratify_by_status else _track_key
    by_stratum: dict[str, list[IncludedRecord]] = defaultdict(list)
    for record in population:
        by_stratum[key_fn(record)].append(record)

    allocation = allocate_proportionally(n, {name: len(recs) for name, recs in by_stratum.items()})

    rows: list[InclusionAuditRow] = []
    for stratum_name, count in allocation.items():
        sampled = active_rng.sample(by_stratum[stratum_name], count)
        rows.extend(InclusionAuditRow(record_id=r.record_id, stratum=stratum_name) for r in sampled)
    return rows


def ingest_inclusion_audit_labels(
    rows: Sequence[InclusionAuditRow],
    labels: Mapping[str, int],
    *,
    reviewer: str | None = None,
    blinded: bool | None = None,
) -> list[InclusionAuditRow]:
    """Attach human gold-check labels to drawn, unlabeled inclusion-audit rows.

    Mirrors `attest.planes.recall_audit.ingest_audit_labels` exactly, for
    `InclusionAuditRow` instead of `AuditRow`.

    Args:
        rows: Rows previously drawn by `draw_inclusion_audit_sample`.
        labels: Mapping of record id to the human ordinal audit label.
        reviewer: Id or pseudonym of the human who produced every label in
            `labels`, recorded on each updated row.
        blinded: Whether `reviewer` gold-checked these rows without seeing
            the ensemble's include-or-escalate decision, recorded on every
            updated row.

    Returns:
        New `InclusionAuditRow`s with `human_label` (and `reviewer`/
        `blinded`, if given) set; still tagged `PLANE_INCLUSION_AUDIT`.

    Raises:
        AuditError: If a row's record id has no entry in `labels`, or its
            label is not a valid ordinal rating.
    """
    updated: list[InclusionAuditRow] = []
    for row in rows:
        if row.record_id not in labels:
            raise AuditError(f"record '{row.record_id}': no human audit label supplied")
        label = labels[row.record_id]
        if label not in VALID_RATINGS:
            raise AuditError(
                f"record '{row.record_id}': human_label must be one of {VALID_RATINGS}, got {label}"
            )
        updated.append(replace(row, human_label=label, reviewer=reviewer, blinded=blinded))
    return updated


def build_inclusion_strata(
    rows: Sequence[AuditPlaneRow],
    population_sizes: Mapping[str, int],
    *,
    relevant_labels: Sequence[int] = (1,),
) -> list[Stratum]:
    """Convert labeled inclusion-audit rows into `attest.stats.recall.Stratum`s.

    This is the inclusion-side firewall's enforcement point, mirroring
    `attest.planes.recall_audit.build_strata`: every row is checked against
    `PLANE_INCLUSION_AUDIT` before it can contribute to a TP estimate. Here
    `stratum.m` counts records found truly relevant among the audited
    include-or-escalate sample -- TP, not FN as in the exclusion-side
    strata `build_strata` produces -- so a `Stratum` from this function
    should be passed to `attest.stats.recall.inclusion_count_lower_bound` /
    `estimated_true_positives` (or, combined with exclusion-side strata, to
    `stratified_recall_with_audited_tp`), never to `exclusion_count_upper_bound`.

    Args:
        rows: Labeled rows to summarize into strata, grouped by `stratum`.
        population_sizes: Mapping of stratum name to the total
            include-and-escalate population size for that stratum.
        relevant_labels: Human ordinal labels counted as a true positive
            when found among the includes. Defaults to include-only (`1`).

    Returns:
        One `Stratum` per distinct `stratum` value present in `rows`.

    Raises:
        AuditError: If any row is not tagged `PLANE_INCLUSION_AUDIT`, any
            row is not yet labeled (`human_label is None`), or a row's
            stratum has no entry in `population_sizes`.
    """
    by_stratum: dict[str, list[AuditPlaneRow]] = defaultdict(list)
    for row in rows:
        if row.plane != PLANE_INCLUSION_AUDIT:
            raise AuditError(
                f"record '{row.record_id}': row is tagged plane '{row.plane}', not "
                f"'{PLANE_INCLUSION_AUDIT}' -- TP may only be estimated from the "
                "random inclusion audit plane"
            )
        if row.human_label is None:
            raise AuditError(f"record '{row.record_id}': row has no human audit label yet")
        by_stratum[row.stratum].append(row)

    strata: list[Stratum] = []
    for name, stratum_rows in by_stratum.items():
        if name not in population_sizes:
            raise AuditError(f"stratum '{name}': no population size given")
        m = sum(1 for row in stratum_rows if row.human_label in relevant_labels)
        strata.append(
            Stratum(
                name=name,
                n=len(stratum_rows),
                m=m,
                population=population_sizes[name],
                role="inclusion",
            )
        )
    return strata
