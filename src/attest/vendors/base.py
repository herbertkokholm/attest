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
import json
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from attest.contracts.input import Record
from attest.ensemble.votes import VALID_RATINGS, Vote, VoteVector
from attest.provenance.config import (
    BATCH_OUTPUT_CONTRACT_VERSION,
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
    "Respond with exactly one letter and nothing else: E to exclude, U if related but "
    "uncertain, or I to include."
)

# Single-token symbols, chosen so the screening decision is always exactly
# one token regardless of vendor tokenizer: some vendors split a numeral
# like "-1" into two tokens ("-" and "1") while others keep it as one, which
# used to make `attest.ensemble.confidence` systematically less confident
# for two-token vendors on the exact same underlying certainty. A bare
# letter tokenizes to one token everywhere in practice. Matching is
# case-insensitive (see `parse_ordinal_response`), so this dict is keyed by
# the upper-cased letter only.
_ORDINAL_TOKENS: dict[str, int] = {"E": -1, "U": 0, "I": 1}
_ORDINAL_TOKEN_TEXT: dict[int, str] = {ordinal: token for token, ordinal in _ORDINAL_TOKENS.items()}

# The kernel-owned output-format contract used in place of `OUTPUT_CONTRACT`
# whenever more than one record is packed into a single request (see
# `Config.batch_size`). Composed onto a batch prompt in exactly one place --
# `compose_batch_system_prompt` -- the same way `OUTPUT_CONTRACT` is for the
# single-record path. Never used when a chunk holds exactly one record: that
# case always routes through `compose_system_prompt`/`OUTPUT_CONTRACT`
# instead, so a `batch_size == 1` configuration never composes this text.
BATCH_OUTPUT_CONTRACT = (
    "You will be screening multiple records in this request, each given a unique id. For "
    "every record, decide whether it should be included using the rule above. Respond with "
    "exactly one JSON object and nothing else, mapping each record's id (as a string) to its "
    "decision letter: E to exclude, U if related but uncertain, or I to include. Include every "
    'given id exactly once. Example, for records with ids "1" and "2": {"1": "I", "2": "E"}'
)


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
    tokens, so replies like ``"I"``, ``"E."``, or ``"Decision: E"`` still
    parse correctly. Matching is case-insensitive. If that scan finds tokens
    spelling more than one distinct rating (e.g. a reply that mentions both
    ``E`` and ``I``), the response is genuinely ambiguous and this raises
    rather than silently picking the first one found.

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
    whole = stripped.strip(".,:;()").upper()
    if whole in _ORDINAL_TOKENS:
        return _ORDINAL_TOKENS[whole]

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1].strip(".,:;()").upper()
        if last_line in _ORDINAL_TOKENS:
            return _ORDINAL_TOKENS[last_line]

    found_ratings: list[int] = []
    for token in text.split():
        cleaned = token.strip().strip(".,:;()").upper()
        if cleaned in _ORDINAL_TOKENS:
            found_ratings.append(_ORDINAL_TOKENS[cleaned])

    if not found_ratings:
        raise VendorResponseError(
            f"could not parse an ordinal rating (E/U/I) from response: {text!r}"
        )
    distinct = sorted(set(found_ratings))
    if len(distinct) > 1:
        raise VendorResponseError(
            f"ambiguous response contains multiple distinct ordinal ratings {distinct}: {text!r}"
        )
    return found_ratings[0]


def compose_batch_system_prompt(criteria: str | None) -> str:
    """Build the final system message for a multi-record (`batch_size > 1`) request.

    The multi-record counterpart of `compose_system_prompt`: same criteria
    fallback, but appends `BATCH_OUTPUT_CONTRACT` (a per-id JSON mapping)
    instead of `OUTPUT_CONTRACT` (a single letter), since a multi-record
    request must return one decision per packed record, not one for the
    whole request. Never used for a chunk of exactly one record -- that case
    always uses `compose_system_prompt`, so a `batch_size == 1`
    configuration never composes this text.

    Args:
        criteria: The task-specific screening criteria text, or None to fall
            back to the generic `SCREENING_TASK_PREAMBLE`.

    Returns:
        `(criteria or SCREENING_TASK_PREAMBLE) + "\\n\\n" + BATCH_OUTPUT_CONTRACT`.
    """
    text = criteria if criteria is not None else SCREENING_TASK_PREAMBLE
    return f"{text}\n\n{BATCH_OUTPUT_CONTRACT}"


def compose_batch_user_message(records: Sequence[Record]) -> str:
    """Build the user message listing every record in a multi-record request, by id.

    Args:
        records: The records packed into this request, in the order their
            ids should appear.

    Returns:
        One "Record <id>: Title: ... Abstract: ..." block per record,
        blank-line separated.
    """
    return "\n\n".join(
        f"Record {record.id}:\nTitle: {record.title}\nAbstract: {record.abstract}"
        for record in records
    )


def parse_batch_response(text: str, record_ids: Sequence[str]) -> dict[str, int]:
    """Parse a multi-record reply into a mapping of record id to ordinal rating.

    Mirrors `parse_ordinal_response`'s failure posture for the multi-record
    case: a record in `record_ids` with no parseable rating in `text` is
    never silently dropped or defaulted -- this raises instead, the same
    disposition an unparseable single-record reply already gets today (see
    `parse_ordinal_response`). An id present in `text` but not in
    `record_ids` -- extra, duplicate, or hallucinated -- is simply ignored.

    Args:
        text: The rater's raw text response, expected to be a JSON object
            mapping record id (string) to a decision letter or ordinal.
        record_ids: The ids every returned rating must cover.

    Returns:
        Mapping of record id to ordinal rating (-1, 0, or 1), one entry per
        id in `record_ids`.

    Raises:
        VendorResponseError: If `text` is not valid JSON, is not a JSON
            object, or has no parseable rating for some id in `record_ids`.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VendorResponseError(
            f"could not parse batch response as JSON: {text!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise VendorResponseError(f"batch response JSON must be an object, got: {text!r}")

    results: dict[str, int] = {}
    for record_id in record_ids:
        value = payload.get(record_id)
        if isinstance(value, str):
            results[record_id] = parse_ordinal_response(value)
        elif isinstance(value, int) and not isinstance(value, bool) and value in VALID_RATINGS:
            results[record_id] = value
        else:
            raise VendorResponseError(
                f"batch response has no parseable rating for record id {record_id!r}: {text!r}"
            )
    return results


def chunk_records(
    records: Sequence[Record], group_key: Callable[[Record], Any], batch_size: int
) -> list[list[Record]]:
    """Split `records` into request-sized chunks, grouped first by `group_key`.

    Records are grouped by `group_key(record)` -- preserving each group's
    relative order and the order groups first appear in `records` -- and
    every group is then split into consecutive pieces of at most
    `batch_size` records; a final undersized piece is valid, not an error.
    Two records with different `group_key` values are never placed in the
    same chunk, so callers pass a key that captures whatever must not be
    mixed within one request (e.g. a record's resolved screening prompt).

    Args:
        records: The records to chunk.
        group_key: Function computing the grouping key for one record.
        batch_size: Maximum number of records per chunk.

    Returns:
        The chunks, in group-then-position order.
    """
    groups: dict[Any, list[Record]] = {}
    order: list[Any] = []
    for record in records:
        key = group_key(record)
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            order.append(key)
        bucket.append(record)

    chunks: list[list[Record]] = []
    for key in order:
        group = groups[key]
        for i in range(0, len(group), batch_size):
            chunks.append(group[i : i + batch_size])
    return chunks


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

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, Any]]:
        """Rate every record in `records` together, in one request.

        Called by `run_ensemble` once per chunk of more than one record (see
        `Config.batch_size`); a chunk of exactly one record always routes
        through `rate` instead, so a `batch_size == 1` configuration never
        calls this method and stays byte-identical to before it existed.

        Args:
            records: The records to rate together, all sharing the same
                resolved screening prompt (see `Config.prompt_for_track`).
            prompt: Screening criteria text to use for this call, overriding
                this rater's own configured criteria -- identical in meaning
                to `rate`'s `prompt` parameter.

        Returns:
            One `(ordinal, raw_response)` pair per record in `records`, in
            the same order.

        Raises:
            NotImplementedError: If this rater does not support packing more
                than one record into a request. Never a silent per-record
                loop: that would send one request per record while the
                configuration's hashed `batch_size` claims more, misrepresenting
                the instrument that actually ran.
            VendorResponseError: If some record in `records` has no
                parseable rating in the response -- the multi-record
                counterpart of `rate`'s own unparseable-response failure;
                never silently dropped or defaulted.
        """
        ...


class SingleRecordOnlyRateMany:
    """Mixin `rate_many` for a `Rater` that has not yet been converted to true
    multi-record packing (see `Config.batch_size`).

    Supports a chunk of exactly one record -- routed through the class's own
    `rate` -- and raises for anything larger, rather than silently issuing
    one request per record while the configuration's hashed `batch_size`
    claims more. `run_ensemble` itself never calls `rate_many` for a
    singleton chunk (it calls `rate` directly), so in practice this mixin's
    `rate_many` is only ever invoked with `len(records) > 1`, and always
    raises; the `len(records) == 1` branch exists for direct callers (e.g.
    tests) and to keep the contract honest on its own terms.
    """

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, Any]]:
        if len(records) > 1:
            raise NotImplementedError(
                f"{type(self).__name__}.rate_many does not support packing more than one "
                f"record into a request (got {len(records)} records); this provider has not "
                "yet been converted to true multi-record packing -- see Config.batch_size"
            )
        return [self.rate(records[0], prompt=prompt)]  # type: ignore[attr-defined]


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
        request_logprobs: If True, also derive a deterministic, seeded fake
            logprob from the same digest and attach it under
            `raw_response["logprobs"]` in the OpenAI-compatible shape
            `attest.ensemble.confidence._openai_compatible_probability`
            parses. This does not simulate any real vendor's actual
            behavior -- it exists solely so `--request-logprobs` (and
            everything downstream of it: `attest.ensemble.confidence`,
            confidence-stratified audit draws, confidence-driven
            active-learning selection) can be exercised network-free in
            tests, exactly as `request_logprobs=False` (the default)
            reproduces this rater's pre-logprobs behavior identically.
    """

    vendor: str
    model: str = "deterministic-v1"
    seed: int = 0
    request_logprobs: bool = False

    def _rate_one(
        self, record: Record, *, prompt: str | None, peers: tuple[str, ...]
    ) -> tuple[int, dict[str, Any]]:
        """Shared digest/rating computation behind `rate` and `rate_many`.

        `peers` is the sorted tuple of the *other* record ids sharing this
        record's request chunk (empty for `rate`'s single-record calls and
        for a `rate_many` chunk of one). It only enters the digest when
        non-empty, so a singleton chunk's digest -- and thus rating and raw
        response -- reduces exactly to what `rate` alone would produce,
        preserving `batch_size == 1` reproducibility while making
        `rate_many` genuinely sensitive to a record's chunk peers (see
        `Config.batch_size`).
        """
        digest_input = f"{self.seed}:{self.vendor}:{self.model}:{record.id}"
        if prompt is not None:
            digest_input = f"{digest_input}:{compose_system_prompt(prompt)}"
        if peers:
            digest_input = f"{digest_input}:{','.join(peers)}"
        digest = hashlib.sha256(digest_input.encode()).digest()
        ordinal = VALID_RATINGS[digest[0] % len(VALID_RATINGS)]
        raw_response: dict[str, Any] = {
            "vendor": self.vendor,
            "model": self.model,
            "seed": self.seed,
            "record_id": record.id,
            "digest": digest.hex(),
            "prompt": prompt,
        }
        if self.request_logprobs:
            fake_logprob = -(digest[1] / 255) * 3.0
            raw_response["logprobs"] = {
                "content": [{"token": _ORDINAL_TOKEN_TEXT[ordinal], "logprob": fake_logprob}]
            }
        return ordinal, raw_response

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
        return self._rate_one(record, prompt=prompt, peers=())

    def rate_many(
        self, records: Sequence[Record], *, prompt: str | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Deterministically derive one ordinal rating per record, each sensitive to
        its chunk peers (see `Config.batch_size`).

        Every record's digest includes the sorted tuple of the *other*
        record ids in `records` -- its chunk peers -- alongside the same
        seed/vendor/model/id/prompt inputs `rate` uses, so this rater's
        output genuinely depends on how records are packed together, not
        just on batch_size counted but ignored. For a singleton `records`
        (one record, no peers), this reduces exactly to `rate`'s own output
        (see `_rate_one`), so `batch_size == 1` reproducibility is
        unaffected by this method's existence.
        """
        ids = [record.id for record in records]
        results: list[tuple[int, dict[str, Any]]] = []
        for record in records:
            peers = tuple(sorted(rid for rid in ids if rid != record.id))
            results.append(self._rate_one(record, prompt=prompt, peers=peers))
        return results


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

    Each record is rated once by each rater in `raters`; the resulting
    per-vendor votes are retained as a `VoteVector`, in `records` order,
    stamped with the `ensemble_config_id` derived from `config`. No
    aggregation happens here -- that is `attest.ensemble.aggregate.g`'s job,
    applied later to the retained vote vectors.

    At `config.batch_size == 1` (the default), each rater is called once per
    record via `Rater.rate`, in `records` order -- the exact code path this
    function has always used, so results stay byte-identical regardless of
    `batch_size`'s existence. At `batch_size > 1`, records are grouped by
    resolved prompt (`config.prompt_for_track`, preserving each group's
    input order; two records with different resolved prompts are never
    packed together) and split into chunks of at most `batch_size` records
    (a final undersized chunk is valid); each rater is called once per chunk
    via `Rater.rate_many`. Chunking is internal to this function -- nothing
    downstream (planes, stats, ablation, `attest.io.store`, the validation
    record) is aware of it; every record still gets exactly one `VoteVector`,
    in its original `records` position.

    Args:
        records: The records to rate.
        raters: The ensemble members to run, in the order their votes are
            recorded into each record's vote vector.
        config: The ensemble configuration in force for this run, used to
            compute the `ensemble_config_id` every resulting vote vector is
            stamped with, to resolve each record's screening prompt and
            chunk-grouping key (`config.prompt_for_track`), and to size
            chunks (`config.batch_size`).

    Returns:
        An `EnsembleRun` holding one `VoteVector` per record, in `records`
        order, and the raw, per-vendor responses that produced them.
    """
    records = list(records)
    ensemble_config_id = compute_ensemble_config_id(config)

    if config.batch_size == 1:
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

    votes_by_id: dict[str, list[Vote]] = {record.id: [] for record in records}
    raw_by_id: dict[str, dict[str, Any]] = {record.id: {} for record in records}
    chunks = chunk_records(
        records, lambda r: config.prompt_for_track(r.track), config.batch_size
    )
    for chunk in chunks:
        prompt = config.prompt_for_track(chunk[0].track)
        for rater in raters:
            if len(chunk) == 1:
                results = [rater.rate(chunk[0], prompt=prompt)]
            else:
                results = rater.rate_many(chunk, prompt=prompt)
            for record, (ordinal, raw) in zip(chunk, results, strict=True):
                votes_by_id[record.id].append(Vote(vendor=rater.vendor, rating=ordinal))
                raw_by_id[record.id][rater.vendor] = raw

    votes = [
        VoteVector(
            record_id=record.id,
            ensemble_config_id=ensemble_config_id,
            votes=tuple(votes_by_id[record.id]),
        )
        for record in records
    ]
    raw_responses = {record.id: raw_by_id[record.id] for record in records}
    return EnsembleRun(votes=votes, raw_responses=raw_responses)


__all__ = [
    "BATCH_OUTPUT_CONTRACT",
    "BATCH_OUTPUT_CONTRACT_VERSION",
    "OUTPUT_CONTRACT",
    "OUTPUT_CONTRACT_VERSION",
    "SCREENING_TASK_PREAMBLE",
    "DeterministicRater",
    "EnsembleRun",
    "ModelVersionDriftError",
    "Rater",
    "SingleRecordOnlyRateMany",
    "VendorResponseError",
    "check_model_version",
    "chunk_records",
    "compose_batch_system_prompt",
    "compose_batch_user_message",
    "compose_system_prompt",
    "parse_batch_response",
    "parse_ordinal_response",
    "run_ensemble",
]
