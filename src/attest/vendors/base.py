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
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from attest.contracts.input import Record
from attest.ensemble.votes import VALID_RATINGS, Vote, VoteVector
from attest.provenance.config import (
    OUTPUT_CONTRACT_VERSION,
    Config,
    compute_ensemble_config_id,
)

# The generic, criteria-free task instruction used when a caller supplies no
# criteria of its own (`compose_system_prompt(None)`). Together with
# `OUTPUT_CONTRACT`, this reproduces the pre-split `DEFAULT_SCREENING_PROMPT`
# text exactly, so a rater that never receives criteria behaves identically
# to before the constant was split.
SCREENING_TASK_PREAMBLE = (
    "You are screening a record for a systematic review. Read the title and "
    "abstract and decide whether the record should be included."
)

# The kernel-owned output-format contract. Composed onto every screening
# prompt by `compose_system_prompt`, in exactly one place, so it can never be
# silently dropped by a caller-supplied criteria string the way it could be
# when it was baked into `DEFAULT_SCREENING_PROMPT`.
OUTPUT_CONTRACT = (
    "Respond with exactly one token: -1 to exclude, 0 if related but uncertain, or 1 to include."
)

_ORDINAL_TOKENS: dict[str, int] = {"-1": -1, "0": 0, "1": 1, "+1": 1}


class VendorResponseError(ValueError):
    """Raised when a vendor's raw response cannot be parsed into an ordinal rating."""


class ModelVersionDriftError(ValueError):
    """Raised when a vendor's response reports a different model version than configured.

    Distinct from `VendorResponseError`: a parse failure is about one
    record's reply; a version drift is about the vendor itself no longer
    matching the `VendorSpec.model_version` this ensemble configuration was
    pinned to and hashed under (e.g. a floating alias like "claude-sonnet-5"
    silently resolved to a newer snapshot) -- it invalidates every result in
    the run, not just one, so it is never caught and skipped the way an
    unparseable reply is.
    """


def check_model_version(
    *, vendor: str, model: str, expected_version: str, reported_version: str | None
) -> None:
    """Raise if a vendor's response reports a model version other than the one configured.

    Args:
        vendor: Vendor name, for the error message.
        model: The model identifier that was requested.
        expected_version: The `VendorSpec.model_version` this rater was
            configured with.
        reported_version: The vendor response's own account of which model
            version actually served the request, or `None` if this vendor's
            SDK/response does not expose that information -- in which case
            no check is possible and this is a silent no-op.

    Raises:
        ModelVersionDriftError: If `reported_version` is not `None` and
            differs from `expected_version`.
    """
    if reported_version is not None and reported_version != expected_version:
        raise ModelVersionDriftError(
            f"vendor '{vendor}' model '{model}' responded with version "
            f"'{reported_version}', but this configuration expects '{expected_version}'"
        )


def compose_system_prompt(criteria: str | None) -> str:
    """Build the final system message from caller-supplied criteria plus the output contract.

    This is the single composition point every rater path (sync providers,
    batch providers, and `DeterministicRater`) uses to turn criteria text
    into the message actually sent to a vendor, so `OUTPUT_CONTRACT` is
    appended in exactly one place and can never be forgotten by a caller
    that supplies its own criteria.

    Args:
        criteria: The task-specific screening criteria text, or None to fall
            back to the generic `SCREENING_TASK_PREAMBLE`.

    Returns:
        `(criteria or SCREENING_TASK_PREAMBLE) + "\\n\\n" + OUTPUT_CONTRACT`.

    Warns:
        UserWarning: If `criteria` already contains a copy of
            `OUTPUT_CONTRACT` -- a leftover from the pre-split convention of
            bundling the output-format sentence into the prompt by hand,
            which would otherwise be sent twice.
    """
    text = criteria if criteria is not None else SCREENING_TASK_PREAMBLE
    if OUTPUT_CONTRACT in text:
        warnings.warn(
            "criteria already contains the output-contract instruction "
            f"({OUTPUT_CONTRACT!r}); attest now appends "
            "attest.vendors.base.OUTPUT_CONTRACT to every composed prompt automatically, "
            "so remove the duplicated sentence from criteria to avoid sending it twice",
            stacklevel=2,
        )
    return f"{text}\n\n{OUTPUT_CONTRACT}"


def parse_ordinal_response(text: str) -> int:
    """Parse a rater's free-text response into an ordinal rating.

    Tries a strict match first: the whole reply (stripped of surrounding
    whitespace and punctuation) is a single ordinal token, or its last
    non-empty line is. Only if neither strict form matches does this fall
    back to scanning every whitespace-separated token in `text` for ordinal
    tokens, so replies like ``"1"``, ``"+1."``, or ``"Decision: -1"`` still
    parse correctly. If that scan finds tokens spelling more than one
    distinct rating (e.g. a reply that mentions both ``-1`` and ``1``), the
    response is genuinely ambiguous and this raises rather than silently
    picking the first one found.

    Args:
        text: The rater's raw text response.

    Returns:
        The parsed ordinal rating: -1, 0, or 1.

    Raises:
        VendorResponseError: If no token in `text` spells a recognized
            ordinal rating, or if tokens spelling more than one distinct
            ordinal rating are found.
    """
    stripped = text.strip()
    whole = stripped.strip(".,:;()")
    if whole in _ORDINAL_TOKENS:
        return _ORDINAL_TOKENS[whole]

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1].strip(".,:;()")
        if last_line in _ORDINAL_TOKENS:
            return _ORDINAL_TOKENS[last_line]

    found_ratings: list[int] = []
    for token in text.split():
        cleaned = token.strip().strip(".,:;()")
        if cleaned in _ORDINAL_TOKENS:
            found_ratings.append(_ORDINAL_TOKENS[cleaned])

    if not found_ratings:
        raise VendorResponseError(
            f"could not parse an ordinal rating (-1/0/1) from response: {text!r}"
        )
    distinct = sorted(set(found_ratings))
    if len(distinct) > 1:
        raise VendorResponseError(
            f"ambiguous response contains multiple distinct ordinal ratings {distinct}: {text!r}"
        )
    return found_ratings[0]


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

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, Any]:
        """Rate `record`, returning `(ordinal, raw_response)`.

        Args:
            record: The record to rate.
            prompt: Screening criteria text to use for this call, overriding
                this rater's own configured criteria. `None` (the default)
                means use this rater's own criteria, unchanged. Criteria
                only -- the output contract is appended once, by the
                implementation, via `compose_system_prompt`; callers must
                never bundle it into `prompt` themselves.

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

    def rate(self, record: Record, *, prompt: str | None = None) -> tuple[int, dict[str, Any]]:
        """Deterministically derive an ordinal rating from `record.id`, this rater's
        seed, and (if given) the composed system prompt built from `prompt` criteria.

        `prompt` only enters the digest when explicitly passed, so every
        pre-existing caller that never passed one (i.e. always called with
        `prompt=None`) gets the exact same ratings as before this parameter
        existed -- calibrated seeds in existing tests stay valid. When
        `prompt` is passed, it is routed through `compose_system_prompt`
        before entering the digest, exactly as a live provider would route
        it before sending, so sync and batch execution -- and this rater's
        sensitivity to `OUTPUT_CONTRACT` changes -- stay in step with the
        live raters it stands in for.
        """
        digest_input = f"{self.seed}:{self.vendor}:{self.model}:{record.id}"
        if prompt is not None:
            digest_input = f"{digest_input}:{compose_system_prompt(prompt)}"
        digest = hashlib.sha256(digest_input.encode()).digest()
        ordinal = VALID_RATINGS[digest[0] % len(VALID_RATINGS)]
        raw_response = {
            "vendor": self.vendor,
            "model": self.model,
            "seed": self.seed,
            "record_id": record.id,
            "digest": digest.hex(),
            "prompt": prompt,
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
            stamped with, and -- via `config.prompt_for_track` -- to resolve
            each record's screening prompt from its `track`.

    Returns:
        An `EnsembleRun` holding one `VoteVector` per record and the raw,
        per-vendor responses that produced them.
    """
    ensemble_config_id = compute_ensemble_config_id(config)
    votes: list[VoteVector] = []
    raw_responses: dict[str, dict[str, Any]] = {}

    for record in records:
        prompt = config.prompt_for_track(record.track)
        record_votes: list[Vote] = []
        record_raw: dict[str, Any] = {}
        for rater in raters:
            ordinal, raw = rater.rate(record, prompt=prompt)
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
    "OUTPUT_CONTRACT",
    "OUTPUT_CONTRACT_VERSION",
    "SCREENING_TASK_PREAMBLE",
    "DeterministicRater",
    "EnsembleRun",
    "ModelVersionDriftError",
    "Rater",
    "VendorResponseError",
    "check_model_version",
    "compose_system_prompt",
    "parse_ordinal_response",
    "run_ensemble",
]
