"""Rater protocol, deterministic rater, and ensemble execution.

`Rater` is the interface every ensemble member implements, whether it calls
out to a live LLM vendor (see `vendors.providers`) or -- as `DeterministicRater`
does -- runs entirely locally with no network access. `run_ensemble` drives a
set of raters over a batch of records and returns the raw vote vectors,
stamped with the `ensemble_config_id` of the configuration that produced
them. Network access, if any, is confined to a `Rater.rate` implementation;
this module itself never makes a network call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from attest.contracts.input import Record
from attest.ensemble.votes import VALID_RATINGS, Vote, VoteVector
from attest.provenance.config import Config, compute_ensemble_config_id

DEFAULT_SCREENING_PROMPT = (
    "You are screening a record for a systematic review. Read the title and "
    "abstract and decide whether the record should be included. "
    "Respond with exactly one token: -1 to exclude, 0 if related but "
    "uncertain, or 1 to include."
)

_ORDINAL_TOKENS: dict[str, int] = {"-1": -1, "0": 0, "1": 1, "+1": 1}


class VendorResponseError(ValueError):
    """Raised when a vendor's raw response cannot be parsed into an ordinal rating."""


def parse_ordinal_response(text: str) -> int:
    """Parse a rater's free-text response into an ordinal rating.

    Scans whitespace-separated tokens in `text` (stripped of surrounding
    punctuation) for the first one that spells an ordinal rating, so replies
    like ``"1"``, ``"+1."``, or ``"Decision: -1"`` all parse correctly.

    Args:
        text: The rater's raw text response.

    Returns:
        The parsed ordinal rating: -1, 0, or 1.

    Raises:
        VendorResponseError: If no token in `text` spells a recognized
            ordinal rating.
    """
    for token in text.split():
        cleaned = token.strip().strip(".,:;()")
        if cleaned in _ORDINAL_TOKENS:
            return _ORDINAL_TOKENS[cleaned]
    raise VendorResponseError(f"could not parse an ordinal rating (-1/0/1) from response: {text!r}")


@runtime_checkable
class Rater(Protocol):
    """A single ensemble member capable of rating one record on the ordinal scale.

    Implementations may reach out to a network service (see
    `vendors.providers`) or be fully local, like `DeterministicRater`.
    `attest` never assumes network access is available; only a concrete
    rater's own `rate` implementation knows whether it needs it.

    Attributes:
        vendor: Name of the vendor this rater belongs to (e.g. "anthropic").
        model: Identifier of the model this rater uses.
    """

    vendor: str
    model: str

    def rate(self, record: Record) -> tuple[int, Any]:
        """Rate `record`, returning `(ordinal, raw_response)`.

        Args:
            record: The record to rate.

        Returns:
            A tuple of the ordinal rating (-1, 0, or 1) and the rater's raw,
            implementation-specific response. The raw response is retained
            for audit and debugging only; it is never part of the versioned
            vote contract (`attest.ensemble.votes`).
        """
        ...


@dataclass
class DeterministicRater:
    """A seedable, network-free rater for tests and frozen-vote reproduction.

    Produces the same ordinal rating for the same (seed, vendor, model,
    record id) every time, derived from a SHA-256 digest rather than
    Python's salted `hash()`, so results are reproducible across processes
    and interpreter runs -- not just within one.

    Attributes:
        vendor: Name to report as this rater's vendor.
        model: Name to report as this rater's model.
        seed: Seed distinguishing this rater's ratings from another
            `DeterministicRater` with the same vendor and model.
    """

    vendor: str
    model: str = "deterministic-v1"
    seed: int = 0

    def rate(self, record: Record) -> tuple[int, dict[str, Any]]:
        """Deterministically derive an ordinal rating from `record.id` and this rater's seed."""
        digest = hashlib.sha256(
            f"{self.seed}:{self.vendor}:{self.model}:{record.id}".encode()
        ).digest()
        ordinal = VALID_RATINGS[digest[0] % len(VALID_RATINGS)]
        raw_response = {
            "vendor": self.vendor,
            "model": self.model,
            "seed": self.seed,
            "record_id": record.id,
            "digest": digest.hex(),
        }
        return ordinal, raw_response


@dataclass(frozen=True)
class EnsembleRun:
    """The result of running a set of raters over a batch of records.

    Attributes:
        votes: One `VoteVector` per record, in input order, each stamped
            with the `ensemble_config_id` of the configuration that was run.
        raw_responses: Per-record, per-vendor raw rater responses, retained
            for audit and debugging. Not part of the versioned vote
            contract -- only `votes` is.
    """

    votes: list[VoteVector]
    raw_responses: dict[str, dict[str, Any]]


def run_ensemble(records: Iterable[Record], raters: Sequence[Rater], config: Config) -> EnsembleRun:
    """Run every rater over every record and collect the raw vote vectors.

    Each record is rated once by each rater in `raters`, in order; the
    resulting per-vendor votes are retained as a `VoteVector` stamped with
    the `ensemble_config_id` derived from `config`. No aggregation happens
    here -- that is `attest.ensemble.aggregate.g`'s job, applied later to the
    retained vote vectors.

    Args:
        records: The records to rate.
        raters: The ensemble members to run, in the order their votes are
            recorded into each record's vote vector.
        config: The ensemble configuration in force for this run, used to
            compute the `ensemble_config_id` every resulting vote vector is
            stamped with.

    Returns:
        An `EnsembleRun` holding one `VoteVector` per record and the raw,
        per-vendor responses that produced them.
    """
    ensemble_config_id = compute_ensemble_config_id(config)
    votes: list[VoteVector] = []
    raw_responses: dict[str, dict[str, Any]] = {}

    for record in records:
        record_votes: list[Vote] = []
        record_raw: dict[str, Any] = {}
        for rater in raters:
            ordinal, raw = rater.rate(record)
            record_votes.append(Vote(vendor=rater.vendor, rating=ordinal))
            record_raw[rater.vendor] = raw
        votes.append(
            VoteVector(
                record_id=record.id,
                ensemble_config_id=ensemble_config_id,
                votes=tuple(record_votes),
            )
        )
        raw_responses[record.id] = record_raw

    return EnsembleRun(votes=votes, raw_responses=raw_responses)


__all__ = [
    "DEFAULT_SCREENING_PROMPT",
    "DeterministicRater",
    "EnsembleRun",
    "Rater",
    "VendorResponseError",
    "parse_ordinal_response",
    "run_ensemble",
]
