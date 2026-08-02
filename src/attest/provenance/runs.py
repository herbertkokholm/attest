"""Run tracking: the per-run provenance record linking a run to its epoch.

A `RunRecord` is the offline-usable provenance object for one execution of
the kernel -- a screening pass, a validation pass, an ablation sweep, or an
audit draw. It records what kind of run it was, which track and epoch it
ran under, its timing, and outcome counts, independent of any storage or
scheduling layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


@dataclass
class RunRecord:
    """Per-run provenance: what ran, on what track, under which epoch, and how it went.

    Attributes:
        id: Unique identifier for this run.
        type: Kind of run (e.g. "screening", "validation", "ablation", "audit").
        track: The screening track this run covers.
        epoch_id: Identifier of the stable epoch this run executed under.
        started_at: When the run started.
        ended_at: When the run ended, or None while still in progress.
        counts: Free-form named counts produced by the run (e.g. "screened").
        status: Current lifecycle status of the run.
    """

    id: str
    type: str
    track: int | str
    epoch_id: str
    started_at: datetime
    ended_at: datetime | None = None
    counts: dict[str, int] = field(default_factory=dict)
    status: str = PENDING

    def finish(self, *, status: str = COMPLETED, ended_at: datetime | None = None) -> None:
        """Mark this run finished, recording its end time and final status.

        Args:
            status: The final lifecycle status to record.
            ended_at: Timestamp to record as the end time; defaults to now (UTC).
        """
        self.ended_at = ended_at or datetime.now(UTC)
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        """Return this run record as a plain, JSON-serializable dict."""
        return {
            "id": self.id,
            "type": self.type,
            "track": self.track,
            "epoch_id": self.epoch_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "counts": dict(self.counts),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        """Reconstruct a run record from a plain dict produced by `to_dict`.

        Args:
            payload: A dict matching the shape returned by `to_dict`.

        Returns:
            The equivalent `RunRecord`.
        """
        ended_at = payload.get("ended_at")
        return cls(
            id=payload["id"],
            type=payload["type"],
            track=payload["track"],
            epoch_id=payload["epoch_id"],
            started_at=datetime.fromisoformat(payload["started_at"]),
            ended_at=datetime.fromisoformat(ended_at) if ended_at is not None else None,
            counts=dict(payload.get("counts", {})),
            status=payload["status"],
        )


def start_run(
    *,
    type: str,
    track: int | str,
    epoch_id: str,
    started_at: datetime | None = None,
) -> RunRecord:
    """Start a new run record in "running" status.

    Args:
        type: Kind of run (e.g. "screening", "validation", "ablation", "audit").
        track: The screening track this run covers.
        epoch_id: Identifier of the stable epoch this run executes under.
        started_at: Timestamp to record as the start time; defaults to now (UTC).

    Returns:
        A new `RunRecord` with a freshly generated id and status "running".
    """
    return RunRecord(
        id=str(uuid.uuid4()),
        type=type,
        track=track,
        epoch_id=epoch_id,
        started_at=started_at or datetime.now(UTC),
        status=RUNNING,
    )
