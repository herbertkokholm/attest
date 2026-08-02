"""Stable-epoch tracking: grouping decisions made under one unchanged ensemble configuration.

An epoch is a span during which the ensemble configuration is held fixed.
Statistics, validation records, and audits are only comparable within a
single epoch; any change to the configuration -- a new model version, a
different aggregation rule, a shifted tau -- ends the current epoch and
opens a new one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from attest.provenance.config import Config, compute_ensemble_config_id


@dataclass(frozen=True)
class Epoch:
    """A stable epoch: a span over which the ensemble configuration did not change.

    Attributes:
        id: Unique identifier for this epoch.
        ensemble_config_id: The content-derived id of the configuration in
            force for the duration of this epoch.
        opened_at: When this epoch was opened.
    """

    id: str
    ensemble_config_id: str
    opened_at: datetime


def open_epoch(config: Config, *, opened_at: datetime | None = None) -> Epoch:
    """Open a new epoch for `config`, unconditionally.

    Args:
        config: The ensemble configuration now in force.
        opened_at: Timestamp to record as the opening time; defaults to now (UTC).

    Returns:
        A freshly opened `Epoch` bound to `config`'s current content.
    """
    return Epoch(
        id=str(uuid.uuid4()),
        ensemble_config_id=compute_ensemble_config_id(config),
        opened_at=opened_at or datetime.now(UTC),
    )


def maybe_open_epoch(
    current: Epoch | None, config: Config, *, opened_at: datetime | None = None
) -> Epoch:
    """Return `current` unchanged, or open a new epoch if `config` has changed.

    A new epoch is opened whenever `config`'s content-derived id differs from
    the id recorded on `current` -- i.e. whenever any field of the
    configuration has changed since `current` was opened. If `current` is
    None, a new epoch is always opened.

    Args:
        current: The presently open epoch, or None if no epoch is open yet.
        config: The ensemble configuration to check against `current`.
        opened_at: Timestamp to record if a new epoch is opened; defaults to
            now (UTC).

    Returns:
        `current` if `config` is unchanged, otherwise a new `Epoch`.
    """
    new_id = compute_ensemble_config_id(config)
    if current is not None and current.ensemble_config_id == new_id:
        return current
    return open_epoch(config, opened_at=opened_at)
