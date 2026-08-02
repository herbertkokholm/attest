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

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from attest.ensemble.votes import VALID_RATINGS
from attest.planes import PLANE_RECALL_AUDIT
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
    """

    record_id: str
    track: int | str


@dataclass(frozen=True)
class AuditRow:
    """One record drawn into the random recall audit.

    Attributes:
        record_id: Id of the drawn record.
        stratum: Stratum name this row was drawn from.
        human_label: Ordinal audit label from gold-checking, or None before
            the row has been audited.
        plane: Fixed to `PLANE_RECALL_AUDIT`; the tag `build_strata` checks
            to refuse rows that did not come from this plane.
    """

    record_id: str
    stratum: str
    human_label: int | None = None
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


def draw_audit_sample(
    population: Sequence[ExcludedRecord],
    n: int,
    *,
    stratify_by_track: bool = False,
    rng: random.Random | None = None,
) -> list[AuditRow]:
    """Draw a random sample of n screen-excluded records for the recall audit.

    Without stratification, `n` records are drawn uniformly at random from
    the whole population, all tagged into a single pooled stratum. With
    `stratify_by_track=True`, `n` is allocated across per-track strata in
    proportion to each track's population size (largest-remainder /
    Hamilton apportionment), then that many records are drawn uniformly at
    random within each stratum -- so no stratum's draw ever exceeds its
    population.

    Args:
        population: The screen-excluded records eligible for audit.
        n: Total number of records to draw.
        stratify_by_track: Whether to stratify the draw by `track`.
        rng: Source of randomness; defaults to a fresh `random.Random()`.
            Pass a seeded `random.Random` for a reproducible draw.

    Returns:
        The drawn rows, unlabeled (`human_label` is None), each tagged
        `PLANE_RECALL_AUDIT`.

    Raises:
        AuditError: If `n` is not positive or exceeds the population size.
    """
    if n <= 0:
        raise AuditError(f"audit sample size n must be positive, got {n}")
    if n > len(population):
        raise AuditError(f"audit sample size n ({n}) exceeds population size ({len(population)})")

    active_rng = rng if rng is not None else random.Random()

    if not stratify_by_track:
        sampled = active_rng.sample(list(population), n)
        return [AuditRow(record_id=r.record_id, stratum=_UNSTRATIFIED_NAME) for r in sampled]

    by_track: dict[str, list[ExcludedRecord]] = defaultdict(list)
    for record in population:
        by_track[str(record.track)].append(record)

    allocation = _allocate_proportionally(n, {name: len(recs) for name, recs in by_track.items()})

    rows: list[AuditRow] = []
    for track_name, count in allocation.items():
        sampled = active_rng.sample(by_track[track_name], count)
        rows.extend(AuditRow(record_id=r.record_id, stratum=track_name) for r in sampled)
    return rows


def _allocate_proportionally(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Allocate n draws across strata proportional to size, via largest-remainder apportionment.

    Guarantees each stratum's allocation never exceeds its size, and the
    allocations sum exactly to n.
    """
    total = sum(sizes.values())
    raw = {name: n * size / total for name, size in sizes.items()}
    floors = {name: math.floor(value) for name, value in raw.items()}
    remainder = n - sum(floors.values())
    by_largest_fraction = sorted(sizes, key=lambda name: raw[name] - floors[name], reverse=True)
    for name in by_largest_fraction[:remainder]:
        floors[name] += 1
    return floors


def ingest_audit_labels(rows: Sequence[AuditRow], labels: Mapping[str, int]) -> list[AuditRow]:
    """Attach human gold-check labels to drawn, unlabeled audit rows.

    Args:
        rows: Rows previously drawn by `draw_audit_sample`.
        labels: Mapping of record id to the human ordinal audit label.

    Returns:
        New `AuditRow`s with `human_label` set from `labels`; still tagged
        `PLANE_RECALL_AUDIT`.

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
        updated.append(replace(row, human_label=label))
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
            Stratum(name=name, n=len(stratum_rows), m=m, population=population_sizes[name])
        )
    return strata
