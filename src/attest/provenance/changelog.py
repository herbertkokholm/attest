"""Changelog of ensemble configuration transitions across epochs.

The changelog is the append-only record of *why* the ensemble configuration
changed from one epoch to the next. Given a config id, it can be walked
backward through recorded events to recover the lineage of configurations
that led to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConfigChangeEvent:
    """A single, immutable transition between two ensemble configurations.

    Attributes:
        timestamp: When the transition was recorded.
        before: The `ensemble_config_id` in force before the change, or None
            if this event records the first configuration ever adopted.
        after: The `ensemble_config_id` in force after the change.
        reason: Free-text explanation for why the configuration changed.
    """

    timestamp: datetime
    before: str | None
    after: str
    reason: str


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
        timestamp: datetime | None = None,
    ) -> ConfigChangeEvent:
        """Append a new change event to the log.

        Args:
            before: The `ensemble_config_id` in force before the change, or
                None if `after` is the first configuration ever adopted.
            after: The `ensemble_config_id` in force after the change.
            reason: Free-text explanation for why the configuration changed.
            timestamp: When the transition occurred; defaults to now (UTC).

        Returns:
            The `ConfigChangeEvent` that was appended.
        """
        event = ConfigChangeEvent(
            timestamp=timestamp or datetime.now(UTC),
            before=before,
            after=after,
            reason=reason,
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
