"""Adjudication plane: disagreement review, statistically separate from the audit plane.

Escalated vote vectors (see `attest.ensemble.aggregate.g`) are routed here for
human review. A supplied human ordinal label is authoritative for an
escalated record -- it replaces the absent ensemble auto-label outright,
never blends with it. A record that did not escalate is never routed here:
it keeps the ensemble's auto-label as-is, since resolving it would be
adjudicating a disagreement that never occurred.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from attest.ensemble.aggregate import Decision
from attest.ensemble.votes import VALID_RATINGS
from attest.planes import PLANE_ADJUDICATION


class AdjudicationError(ValueError):
    """Raised when adjudication routing or resolution violates the plane's invariants."""


@dataclass(frozen=True)
class AdjudicationItem:
    """One escalated record awaiting or resolved by human adjudication.

    Attributes:
        record_id: Id of the escalated record.
        ensemble_config_id: Ensemble configuration in force when it escalated.
        dispersion: The vote vector's dispersion at the time of escalation.
        boundary: Whether the vote vector straddled the exclude/include boundary.
        human_label: The authoritative human ordinal label, or None while pending.
        plane: Fixed to `PLANE_ADJUDICATION`.
    """

    record_id: str
    ensemble_config_id: str
    dispersion: float
    boundary: bool
    human_label: int | None = None
    plane: str = field(default=PLANE_ADJUDICATION, init=False)

    @property
    def resolved(self) -> bool:
        """Whether this item carries an authoritative human label."""
        return self.human_label is not None


@dataclass
class AdjudicationQueue:
    """A human-review queue of escalated records, keyed by record id."""

    items: dict[str, AdjudicationItem] = field(default_factory=dict)

    def enqueue(
        self, record_id: str, ensemble_config_id: str, decision: Decision
    ) -> AdjudicationItem:
        """Route an escalated decision to the adjudication queue.

        Args:
            record_id: Id of the record the decision was computed for.
            ensemble_config_id: Ensemble configuration in force for `decision`.
            decision: The escalated `Decision` from `attest.ensemble.aggregate.g`.

        Returns:
            The newly queued, unresolved `AdjudicationItem`.

        Raises:
            AdjudicationError: If `decision` did not escalate -- agreed
                records keep the ensemble label and are never routed here.
        """
        if not decision.escalate:
            raise AdjudicationError(
                f"record '{record_id}': decision did not escalate; nothing to adjudicate "
                "-- use the ensemble auto-label directly"
            )
        item = AdjudicationItem(
            record_id=record_id,
            ensemble_config_id=ensemble_config_id,
            dispersion=decision.dispersion,
            boundary=decision.boundary,
        )
        self.items[record_id] = item
        return item

    def resolve(self, record_id: str, human_label: int) -> AdjudicationItem:
        """Resolve a queued item with the authoritative human ordinal label.

        Args:
            record_id: Id of a record previously enqueued.
            human_label: The human adjudicator's ordinal label: -1, 0, or +1.

        Returns:
            The resolved `AdjudicationItem`, with `human_label` now authoritative.

        Raises:
            AdjudicationError: If `record_id` was never enqueued, or
                `human_label` is not a valid ordinal rating.
        """
        if record_id not in self.items:
            raise AdjudicationError(f"record '{record_id}' is not in the adjudication queue")
        if human_label not in VALID_RATINGS:
            raise AdjudicationError(
                f"record '{record_id}': human_label must be one of {VALID_RATINGS}, "
                f"got {human_label}"
            )
        resolved = replace(self.items[record_id], human_label=human_label)
        self.items[record_id] = resolved
        return resolved

    def pending(self) -> list[AdjudicationItem]:
        """Return queued items that do not yet carry an authoritative human label."""
        return [item for item in self.items.values() if not item.resolved]


def final_label(record_id: str, decision: Decision, human_label: int | None = None) -> int:
    """Return the authoritative final ordinal label for a record's decision.

    An escalated decision requires `human_label`: the human label is
    authoritative and is returned outright. A non-escalated (agreed)
    decision keeps its ensemble `auto_label`; `human_label` must not be
    supplied for it, since there is no disagreement to adjudicate.

    Args:
        record_id: Id of the record, used only for error messages.
        decision: The `Decision` computed by `attest.ensemble.aggregate.g`.
        human_label: The human adjudicator's label, required iff `decision.escalate`.

    Returns:
        The final ordinal label: -1, 0, or +1.

    Raises:
        AdjudicationError: If an escalated decision is missing `human_label`
            (or it is invalid), or a non-escalated decision is given one.
    """
    if decision.escalate:
        if human_label is None:
            raise AdjudicationError(
                f"record '{record_id}': decision escalated and requires a human_label to resolve"
            )
        if human_label not in VALID_RATINGS:
            raise AdjudicationError(
                f"record '{record_id}': human_label must be one of {VALID_RATINGS}, "
                f"got {human_label}"
            )
        return human_label

    if human_label is not None:
        raise AdjudicationError(
            f"record '{record_id}': decision did not escalate; human_label must not be supplied"
        )
    if decision.auto_label is None:
        raise AdjudicationError(
            f"record '{record_id}': non-escalated decision has no auto_label (invariant violation)"
        )
    return decision.auto_label
