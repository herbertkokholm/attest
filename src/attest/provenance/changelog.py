"""Changelog of ensemble configuration transitions across epochs.

The changelog is the append-only record of *why* the ensemble configuration
changed from one epoch to the next. Given a config id, it can be walked
backward through recorded events to recover the lineage of configurations
that led to it. Persisted as an append-only run artifact by
`attest.io.store.RunStore.write_changelog`/`append_change_event`: a stored
changelog can only be extended, never rewritten, so once an event is
persisted it is part of the permanent provenance record.

Three change types are recognized (`KNOWN_CHANGE_TYPES`):

- `CHANGE_TYPE_INITIAL` -- the first configuration ever adopted for a run
  directory (`before` is always `None`).
- `CHANGE_TYPE_EXPLICIT` -- a deliberate change to the hashed screening
  configuration (a new model/prompt/threshold), which always changes
  `ensemble_config_id`.
- `CHANGE_TYPE_SENTINEL_DRIFT` -- a latent-vendor-drift hard trigger (see
  `attest.provenance.sentinel` and `docs/sentinel_drift_rule.md`). This is
  the one case where `before == after`: drift is a vendor *behavior* change
  invisible to `Config`'s content hash, so it still forces a new epoch
  without changing `ensemble_config_id`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from attest.provenance.config import Config

CHANGE_TYPE_INITIAL = "initial_config"
CHANGE_TYPE_EXPLICIT = "explicit_config_change"
CHANGE_TYPE_SENTINEL_DRIFT = "sentinel_drift"

KNOWN_CHANGE_TYPES = (CHANGE_TYPE_INITIAL, CHANGE_TYPE_EXPLICIT, CHANGE_TYPE_SENTINEL_DRIFT)


class ChangelogError(ValueError):
    """Raised when a change event or changelog violates its invariants."""


@dataclass(frozen=True)
class ConfigChangeEvent:
    """A single, immutable transition between two ensemble configurations.

    Attributes:
        timestamp: When the transition was recorded (UTC).
        before: The `ensemble_config_id` in force before the change, or None
            iff this event's `change_type` is `CHANGE_TYPE_INITIAL`.
        after: The `ensemble_config_id` in force after the change. Equal to
            `before` for a `CHANGE_TYPE_SENTINEL_DRIFT` event, since drift is
            invisible to the content hash.
        reason: Free-text explanation for why the configuration changed.
        change_type: One of `KNOWN_CHANGE_TYPES`.
        changed_fields: Machine-readable, dot-flattened list of the
            `Config.to_dict()` field paths that differ between `before` and
            `after` (see `diff_config_fields`). Empty for an initial or
            sentinel-drift event, since neither is a `Config` field diff.
        approver: Optional reviewer/approver id or pseudonym who authorized
            this change, e.g. for an explicit config change decided after
            adjudication review (see `attest.planes.adjudication`).
    """

    timestamp: datetime
    before: str | None
    after: str
    reason: str
    change_type: str = CHANGE_TYPE_EXPLICIT
    changed_fields: tuple[str, ...] = ()
    approver: str | None = None

    def __post_init__(self) -> None:
        if self.change_type not in KNOWN_CHANGE_TYPES:
            raise ChangelogError(
                f"unknown change_type '{self.change_type}': expected one of {KNOWN_CHANGE_TYPES}"
            )
        if self.change_type == CHANGE_TYPE_INITIAL and self.before is not None:
            raise ChangelogError(
                f"change_type '{CHANGE_TYPE_INITIAL}' requires before=None, got '{self.before}'"
            )
        if self.change_type != CHANGE_TYPE_INITIAL and self.before is None:
            raise ChangelogError(f"change_type '{self.change_type}' requires a non-None before id")

    def to_dict(self) -> dict[str, Any]:
        """Return this change event as a plain, JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "change_type": self.change_type,
            "changed_fields": list(self.changed_fields),
            "approver": self.approver,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfigChangeEvent:
        """Reconstruct a change event from a plain dict produced by `to_dict`."""
        return cls(
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            before=payload.get("before"),
            after=payload["after"],
            reason=payload["reason"],
            change_type=payload.get("change_type", CHANGE_TYPE_EXPLICIT),
            changed_fields=tuple(payload.get("changed_fields", ())),
            approver=payload.get("approver"),
        )


@dataclass
class ChangeLog:
    """An append-only log of `ConfigChangeEvent`s, resolvable back to any config id."""

    events: list[ConfigChangeEvent] = field(default_factory=list)

    def record(
        self,
        *,
        before: str | None,
        after: str,
        reason: str,
        change_type: str | None = None,
        changed_fields: Sequence[str] = (),
        approver: str | None = None,
        timestamp: datetime | None = None,
    ) -> ConfigChangeEvent:
        """Append a new change event to the log.

        Args:
            before: The `ensemble_config_id` in force before the change, or
                None if `after` is the first configuration ever adopted.
            after: The `ensemble_config_id` in force after the change.
            reason: Free-text explanation for why the configuration changed.
            change_type: One of `KNOWN_CHANGE_TYPES`. Defaults to
                `CHANGE_TYPE_INITIAL` when `before is None`, else
                `CHANGE_TYPE_EXPLICIT` -- the common cases infer correctly
                without the caller having to name the type explicitly; pass
                `CHANGE_TYPE_SENTINEL_DRIFT` explicitly for a drift trigger,
                since that is the one case `before == after`.
            changed_fields: Dot-flattened `Config.to_dict()` field paths that
                changed (see `diff_config_fields`).
            approver: Optional reviewer/approver id or pseudonym.
            timestamp: When the transition occurred; defaults to now (UTC).

        Returns:
            The `ConfigChangeEvent` that was appended.
        """
        resolved_type = change_type or (
            CHANGE_TYPE_INITIAL if before is None else CHANGE_TYPE_EXPLICIT
        )
        event = ConfigChangeEvent(
            timestamp=timestamp or datetime.now(UTC),
            before=before,
            after=after,
            reason=reason,
            change_type=resolved_type,
            changed_fields=tuple(changed_fields),
            approver=approver,
        )
        self.events.append(event)
        return event

    def history_of(self, config_id: str) -> list[ConfigChangeEvent]:
        """Resolve the chain of change events that led to `config_id`, oldest first.

        Walks the log backward from its most recent event, following each
        event's `before` link, until it reaches an event whose `before` is
        None (the first configuration ever adopted) or no matching event is
        found.

        Args:
            config_id: The `ensemble_config_id` to resolve the history of.

        Returns:
            The chain of events leading to `config_id`, ordered oldest first.
            Empty if `config_id` was never recorded as the `after` of an event.
        """
        chain: list[ConfigChangeEvent] = []
        target: str | None = config_id
        for event in reversed(self.events):
            if event.after == target:
                chain.append(event)
                target = event.before
                if target is None:
                    break
        chain.reverse()
        return chain

    def previous_config_id(self, config_id: str) -> str | None:
        """Return the config id immediately preceding `config_id`, if recorded.

        Args:
            config_id: The `ensemble_config_id` to look up.

        Returns:
            The `before` id of the most recent event whose `after` is
            `config_id`, or None if no such event is recorded.
        """
        for event in reversed(self.events):
            if event.after == config_id:
                return event.before
        return None

    def to_list(self) -> list[dict[str, Any]]:
        """Return every event as a plain, JSON-serializable list, oldest first."""
        return [event.to_dict() for event in self.events]

    @classmethod
    def from_list(cls, payload: Sequence[Mapping[str, Any]]) -> ChangeLog:
        """Reconstruct a changelog from a plain list produced by `to_list`."""
        return cls(events=[ConfigChangeEvent.from_dict(e) for e in payload])


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def diff_config_fields(before: Config | None, after: Config) -> list[str]:
    """List the dot-flattened `Config.to_dict()` field paths that differ between two configs.

    Nested mappings (e.g. `vendors.openai.model_version`) are flattened so a
    single vendor's field change is reported precisely rather than as an
    opaque top-level `"vendors"` diff.

    Args:
        before: The prior configuration, or None if `after` is the first
            configuration ever adopted (every field of `after` is then
            reported as changed).
        after: The new configuration.

    Returns:
        A sorted list of differing field paths.
    """
    after_flat = _flatten(after.to_dict())
    if before is None:
        return sorted(after_flat)
    before_flat = _flatten(before.to_dict())
    keys = set(before_flat) | set(after_flat)
    return sorted(key for key in keys if before_flat.get(key) != after_flat.get(key))


__all__ = [
    "CHANGE_TYPE_EXPLICIT",
    "CHANGE_TYPE_INITIAL",
    "CHANGE_TYPE_SENTINEL_DRIFT",
    "KNOWN_CHANGE_TYPES",
    "ChangeLog",
    "ChangelogError",
    "ConfigChangeEvent",
    "diff_config_fields",
]
