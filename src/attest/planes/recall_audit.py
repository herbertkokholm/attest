"""Random recall audit plane: the only plane from which recall may be statistically estimated.

A probability sample of the screen-excluded population is drawn, gold-checked
by a human auditor, and the resulting counts are handed to
`attest.stats.recall`. This is the only plane recall may be estimated from:
adjudication and active-learning selections are not probability samples of
the excluded population (adjudication is triggered by disagreement,
active-learning by uncertainty), so using them here would bias the estimate
in an unknown direction. `build_strata` enforces this at the boundary: any
row not tagged `PLANE_RECALL_AUDIT` is refused outright, not silently ignored.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from attest.ensemble.confidence import UNSCORED_TIER
from attest.ensemble.votes import VALID_RATINGS
from attest.planes import PLANE_RECALL_AUDIT
from attest.planes._apportionment import allocate_proportionally
from attest.stats.recall import Stratum

_UNSTRATIFIED_NAME = "all"


class AuditError(ValueError):
    """Raised when recall-audit sampling, ingestion, or plane enforcement is violated."""


@dataclass(frozen=True)
class ExcludedRecord:
    """One screen-excluded record eligible for the random recall audit.

    Attributes:
        record_id: Id of the screen-excluded record.
        track: The screening track this record belongs to, used as the
            stratification key when `stratify_by_track` is set.
        confidence_tier: This record's ensemble-confidence tier (`"low"`,
            `"high"`, or `attest.ensemble.confidence.UNSCORED_TIER` for
            coverage-gated records), computed by
            `attest.ensemble.confidence.confidence_tier` and used as the
            stratification key when `stratify_by_confidence` is set.
            `None` if confidence stratification is not in use for this run.
    """

    record_id: str
    track: int | str
    confidence_tier: str | None = None


@dataclass(frozen=True)
class AuditRow:
    """One record drawn into the random recall audit.

    Attributes:
        record_id: Id of the drawn record.
        stratum: Stratum name this row was drawn from.
        human_label: Ordinal audit label from gold-checking, or None before
            the row has been audited.
        reviewer: Id or pseudonym of the human who gold-checked this row,
            or None while pending or when not supplied. Optional, like
            `attest.planes.adjudication.AdjudicationItem.reviewer` --
            attest never requires reviewer identity to function, only
            records it when supplied.
        blinded: Whether the reviewer gold-checked this row without seeing
            the ensemble's screen-excluded decision for it, or None when
            not recorded. Provenance only, like `reviewer`: this plane's
            statistical validity comes from `population` being a
            probability sample (see `draw_audit_sample`'s
            `sampling_frame_hash`), not from blinding, but an unblinded
            audit is worth being able to see in the record.
        plane: Fixed to `PLANE_RECALL_AUDIT`; the tag `build_strata` checks
            to refuse rows that did not come from this plane.
    """

    record_id: str
    stratum: str
    human_label: int | None = None
    reviewer: str | None = None
    blinded: bool | None = None
    plane: str = field(default=PLANE_RECALL_AUDIT, init=False)


class AuditPlaneRow(Protocol):
    """Structural shape `build_strata` requires: enough to check the plane tag and count.

    Declared as read-only properties, not plain attributes, so that frozen
    dataclasses (e.g. `AuditRow` itself) satisfy this protocol: mypy treats a
    frozen dataclass's fields as read-only, which is incompatible with a
    Protocol's implicitly settable plain attributes.
    """

    @property
    def record_id(self) -> str: ...
    @property
    def stratum(self) -> str: ...
    @property
    def human_label(self) -> int | None: ...
    @property
    def plane(self) -> str: ...


def _track_key(record: ExcludedRecord) -> str:
    return str(record.track)


def _confidence_key(record: ExcludedRecord) -> str:
    if record.confidence_tier is None:
        raise AuditError(
            f"record '{record.record_id}': stratify_by_confidence=True requires every "
            "record to carry a confidence_tier (see attest.ensemble.confidence."
            f"confidence_tier, which returns {UNSCORED_TIER!r} for coverage-gated "
            "records rather than None) -- got None"
        )
    return record.confidence_tier


class _HasRecordId(Protocol):
    """Structural shape `population_frame_hash` requires: just a record id.

    Both `ExcludedRecord` and `attest.planes.inclusion_audit.IncludedRecord`
    satisfy this, so both planes hash their sampling frame with the exact
    same function.
    """

    @property
    def record_id(self) -> str: ...


def population_frame_hash(population: Sequence[_HasRecordId]) -> str:
    """Return a content hash of the eligible population a draw was sampled from.

    Persisted alongside a draw (see `attest.io.store.RunStore.write_audit_draw`)
    so the sampling frame is provably fixed before any row is labeled: a
    reviewer can recompute this hash from an independently reconstructed
    screen-excluded population and confirm it matches what the draw actually
    ran against, rather than trusting an unverifiable claim that the frame
    wasn't narrowed or widened after the fact. Order-independent: two
    populations containing the same records in a different order hash the
    same, since population order has no statistical meaning here.

    Args:
        population: The screen-excluded records eligible for the draw.

    Returns:
        A SHA-256 hex digest of the sorted, deduplicated record ids.
    """
    ids = sorted({record.record_id for record in population})
    digest = hashlib.sha256("\n".join(ids).encode("utf-8"))
    return digest.hexdigest()


def draw_audit_sample(
    population: Sequence[ExcludedRecord],
    n: int,
    *,
    stratify_by_track: bool = False,
    stratify_by_confidence: bool = False,
    rng: random.Random | None = None,
) -> list[AuditRow]:
    """Draw a random sample of n screen-excluded records for the recall audit.

    Without stratification, `n` records are drawn uniformly at random from
    the whole population, all tagged into a single pooled stratum. With
    `stratify_by_track=True` (or `stratify_by_confidence=True`), `n` is
    allocated across per-track (or per-confidence-tier) strata in
    proportion to each stratum's population size (largest-remainder /
    Hamilton apportionment), then that many records are drawn uniformly at
    random within each stratum -- so no stratum's draw ever exceeds its
    population.

    `stratify_by_confidence` groups by `ExcludedRecord.confidence_tier`
    (`"low"`/`"high"`/`attest.ensemble.confidence.UNSCORED_TIER`, computed
    per-record by `attest.ensemble.confidence.confidence_tier`) rather than
    `track`. This is a proportional draw, not a prioritized one: it does
    not yet oversample the low-confidence tier the way "stratify toward
    low-confidence records" implies -- see the "left open" note in
    `docs/logprob_support.md` for that further, deliberately deferred,
    Neyman-allocation refinement.

    Args:
        population: The screen-excluded records eligible for audit.
        n: Total number of records to draw.
        stratify_by_track: Whether to stratify the draw by `track`.
        stratify_by_confidence: Whether to stratify the draw by
            `confidence_tier` instead. Mutually exclusive with
            `stratify_by_track` -- combined stratification across both
            keys is an open design question, not yet supported.
        rng: Source of randomness; defaults to a fresh `random.Random()`.
            Pass a seeded `random.Random` for a reproducible draw.

    Returns:
        The drawn rows, unlabeled (`human_label` is None), each tagged
        `PLANE_RECALL_AUDIT`.

    Raises:
        AuditError: If `n` is not positive or exceeds the population size;
            if both `stratify_by_track` and `stratify_by_confidence` are
            set; or if `stratify_by_confidence` is set and some record has
            no `confidence_tier`.
    """
    if n <= 0:
        raise AuditError(f"audit sample size n must be positive, got {n}")
    if n > len(population):
        raise AuditError(f"audit sample size n ({n}) exceeds population size ({len(population)})")
    if stratify_by_track and stratify_by_confidence:
        raise AuditError(
            "stratify_by_track and stratify_by_confidence cannot both be set: combined "
            "stratification across both keys is an open design question (see "
            "docs/logprob_support.md's 'left open' note), not yet supported -- choose one"
        )

    active_rng = rng if rng is not None else random.Random()

    if not stratify_by_track and not stratify_by_confidence:
        sampled = active_rng.sample(list(population), n)
        return [AuditRow(record_id=r.record_id, stratum=_UNSTRATIFIED_NAME) for r in sampled]

    key_fn = _confidence_key if stratify_by_confidence else _track_key
    by_stratum: dict[str, list[ExcludedRecord]] = defaultdict(list)
    for record in population:
        by_stratum[key_fn(record)].append(record)

    allocation = allocate_proportionally(n, {name: len(recs) for name, recs in by_stratum.items()})

    rows: list[AuditRow] = []
    for stratum_name, count in allocation.items():
        sampled = active_rng.sample(by_stratum[stratum_name], count)
        rows.extend(AuditRow(record_id=r.record_id, stratum=stratum_name) for r in sampled)
    return rows


def ingest_audit_labels(
    rows: Sequence[AuditRow],
    labels: Mapping[str, int],
    *,
    reviewer: str | None = None,
    blinded: bool | None = None,
) -> list[AuditRow]:
    """Attach human gold-check labels to drawn, unlabeled audit rows.

    Args:
        rows: Rows previously drawn by `draw_audit_sample`.
        labels: Mapping of record id to the human ordinal audit label.
        reviewer: Id or pseudonym of the human who produced every label in
            `labels`, recorded on each updated row. Applies uniformly to
            this whole call -- a labeling pass split across reviewers is
            two separate `ingest_audit_labels` calls, one per reviewer.
        blinded: Whether `reviewer` gold-checked these rows without seeing
            the ensemble's screen-excluded decision, recorded on every
            updated row.

    Returns:
        New `AuditRow`s with `human_label` (and `reviewer`/`blinded`, if
        given) set; still tagged `PLANE_RECALL_AUDIT`.

    Raises:
        AuditError: If a row's record id has no entry in `labels`, or its
            label is not a valid ordinal rating.
    """
    updated: list[AuditRow] = []
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


def build_strata(
    rows: Sequence[AuditPlaneRow],
    population_sizes: Mapping[str, int],
    *,
    relevant_labels: Sequence[int] = (1,),
) -> list[Stratum]:
    """Convert labeled audit rows into `attest.stats.recall.Stratum`s.

    This is the firewall's enforcement point: every row is checked against
    `PLANE_RECALL_AUDIT` before it can contribute to a recall estimate. A
    row from any other plane -- most importantly an active-learning
    selection or adjudication item, neither of which is a probability
    sample -- is refused outright.

    Args:
        rows: Labeled rows to summarize into strata, grouped by `stratum`.
        population_sizes: Mapping of stratum name to |U_h|, the total
            screen-excluded population size for that stratum.
        relevant_labels: Human ordinal labels counted as a false negative
            (a missed truly-relevant record) when found among the excludes.
            Defaults to include-only (`1`).

    Returns:
        One `Stratum` per distinct `stratum` value present in `rows`.

    Raises:
        AuditError: If any row is not tagged `PLANE_RECALL_AUDIT`, any row
            is not yet labeled (`human_label is None`), or a row's stratum
            has no entry in `population_sizes`.
    """
    by_stratum: dict[str, list[AuditPlaneRow]] = defaultdict(list)
    for row in rows:
        if row.plane != PLANE_RECALL_AUDIT:
            raise AuditError(
                f"record '{row.record_id}': row is tagged plane '{row.plane}', not "
                f"'{PLANE_RECALL_AUDIT}' -- recall may only be estimated from the "
                "random recall audit plane"
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
                role="exclusion",
            )
        )
    return strata
