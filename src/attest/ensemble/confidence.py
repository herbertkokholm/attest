"""Per-vote confidence derived from vendor logprobs, coverage-gated into a per-record figure.

Implements the decision recorded in `docs/logprob_support.md`
("Decision: coverage-aware confidence for audit-draw stratification"):

- **Exclusion, not imputation.** A vendor that logs no logprob for a vote
  (permanently, like Anthropic; or for this particular call, like
  `request_logprobs=False`, or an unparseable/unconfirmed payload, like
  Google's still-unverified shape) contributes nothing to the confidence
  signal. `vote_confidence` returns `None` in every such case, deliberately
  conflating them: the caller must not try to tell "vendor never supports
  this" apart from "shape didn't parse this time," since either way there
  is simply no confidence figure for that vote. The vote's ordinal rating
  itself is untouched by any of this -- it still counts fully in
  `attest.ensemble.tau`/`attest.ensemble.aggregate`.
- **Minimum coverage of 3 supporting votes.** Below three, no central
  tendency statistic exists that is robust to a single miscalibrated
  vendor (a median only diverges from the plain mean starting at n=3); see
  the docs section above for the full argument. `record_confidence` marks
  a record `scored=False` below that threshold rather than returning a
  number that looks meaningful but isn't.

This module is pure: it only reads already-collected `raw_response`
payloads (`attest.vendors.base.EnsembleRun.raw_responses`) and
`attest.ensemble.votes.VoteVector`; it makes no network calls itself.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from attest.ensemble.votes import VoteVector

MIN_SUPPORTING_VOTES = 3
UNSCORED_TIER = "unscored"

TIER_LOW = "low"
TIER_HIGH = "high"


class ConfidenceError(ValueError):
    """Raised when a confidence computation is given a nonsensical input."""


def _openai_compatible_probability(logprobs: Any) -> float | None:
    """Extract P(the emitted text) from an OpenAI-compatible `ChoiceLogprobs` payload.

    Shared by openai, mistral, fireworks (sync), and together: all four
    request logprobs through an OpenAI-shaped Chat Completions endpoint and
    retain the same `{"content": [{"token", "logprob", ...}, ...]}` shape
    (see `attest.vendors.providers.{openai,mistral,fireworks,together}`).

    By the chain rule, `P(full text) = exp(sum of each token's conditional
    logprob)` -- summing every entry in `content` is robust to however many
    tokens the ordinal reply happened to tokenize into (e.g. "-1" splitting
    into "-" and "1"), without needing to guess which token "is" the
    answer.
    """
    if not isinstance(logprobs, Mapping):
        return None
    content = logprobs.get("content")
    if not content:
        return None
    total = 0.0
    for entry in content:
        if not isinstance(entry, Mapping):
            return None
        logprob = entry.get("logprob")
        if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            return None
        total += logprob
    return math.exp(total)


def _google_probability(logprobs: Any) -> float | None:
    """Best-effort P(the emitted text) from Gemini's `logprobs_result`.

    Unconfirmed live, per `docs/logprob_support.md`:
    `attest.vendors.providers.google._serialize_logprobs` falls back to
    `str()` when neither `to_dict` nor `model_dump` exists on the SDK's
    protobuf-backed response object, and even a successfully-serialized
    dict's key casing (`chosenCandidates` vs `chosen_candidates`) isn't
    pinned down without a live call via `tools/vendor_logprob_probe.py`.
    This tries the shapes Gemini's REST/SDK docs describe and returns
    `None` -- never raises -- on anything else (including the bare-`str()`
    fallback), so an unconfirmed or drifted shape degrades to "no
    confidence for this vote" rather than corrupting the aggregate.
    """
    if not isinstance(logprobs, Mapping):
        return None
    candidates = logprobs.get("chosenCandidates", logprobs.get("chosen_candidates"))
    if not candidates:
        return None
    total = 0.0
    for entry in candidates:
        if not isinstance(entry, Mapping):
            return None
        logprob = entry.get("logProbability", entry.get("log_probability"))
        if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            return None
        total += logprob
    return math.exp(total)


# No entry for "anthropic": AnthropicRater/AnthropicBatchRater never
# populate raw_response["logprobs"] (the Messages API has no equivalent,
# see attest.vendors.providers.anthropic), so vote_confidence returns None
# for it via the plain `logprobs is None` check below without ever needing
# a per-vendor extractor. Same for any vendor/mode not requesting logprobs
# (e.g. FireworksBatchRater, or any rater built with request_logprobs=False).
_EXTRACTORS: dict[str, Callable[[Any], float | None]] = {
    "openai": _openai_compatible_probability,
    "mistral": _openai_compatible_probability,
    "fireworks": _openai_compatible_probability,
    "together": _openai_compatible_probability,
    "google": _google_probability,
}


def vote_confidence(vendor: str, raw_response: Any) -> float | None:
    """P(the token(s) this vendor actually emitted) for one vote, or None if unavailable.

    Args:
        vendor: The voting vendor's name (`attest.ensemble.votes.Vote.vendor`).
        raw_response: That vendor's raw response for this record, i.e.
            `attest.vendors.base.EnsembleRun.raw_responses[record_id][vendor]`.

    Returns:
        A probability in `[0, 1]`, or `None` if no confidence figure is
        available -- covering, without distinguishing them: the vendor
        never supports logprobs, this call didn't request them, or the
        response omitted/malformed them despite being asked. See the
        module docstring for why these are deliberately not distinguished.
    """
    extractor = _EXTRACTORS.get(vendor)
    if extractor is None or not isinstance(raw_response, Mapping):
        return None
    logprobs = raw_response.get("logprobs")
    if logprobs is None:
        return None
    probability = extractor(logprobs)
    if probability is None:
        return None
    # Clamp away float noise that could push exp(sum(logprob)) a hair past
    # 1.0 for a near-certain multi-token sequence; logprobs are always <= 0
    # in exact arithmetic, so the true probability is always <= 1.0.
    return min(1.0, probability)


@dataclass(frozen=True)
class RecordConfidence:
    """One record's coverage-gated ensemble confidence.

    Attributes:
        record_id: Id of the scored record.
        median_probability: Median per-vote `P(emitted token)` across
            supporting vendors only. Meaningless (fixed at 0.0) when
            `scored` is False -- callers must check `scored` first.
        n_supporting: Number of votes a confidence figure was available for.
        n_total: Total number of votes cast for this record, for context.
        scored: True iff `n_supporting >= MIN_SUPPORTING_VOTES`.
    """

    record_id: str
    median_probability: float
    n_supporting: int
    n_total: int
    scored: bool


def record_confidence(
    vote_vector: VoteVector, raw_responses: Mapping[str, Any]
) -> RecordConfidence:
    """Compute one record's coverage-gated confidence from its votes and raw responses.

    Args:
        vote_vector: The record's cast votes.
        raw_responses: This record's per-vendor raw responses, i.e.
            `attest.vendors.base.EnsembleRun.raw_responses[record_id]`.

    Returns:
        A `RecordConfidence` whose `median_probability` is the median of
        `vote_confidence` across only the votes it was available for.
        `scored` is False whenever fewer than `MIN_SUPPORTING_VOTES` votes
        yielded a confidence figure -- see the module docstring for why 3
        is the minimum below which no statistic here would be robust to a
        single miscalibrated vendor.
    """
    probabilities = [
        probability
        for vote in vote_vector.votes
        if (probability := vote_confidence(vote.vendor, raw_responses.get(vote.vendor))) is not None
    ]
    n_supporting = len(probabilities)
    scored = n_supporting >= MIN_SUPPORTING_VOTES
    return RecordConfidence(
        record_id=vote_vector.record_id,
        median_probability=statistics.median(probabilities) if scored else 0.0,
        n_supporting=n_supporting,
        n_total=len(vote_vector.votes),
        scored=scored,
    )


DEFAULT_LOW_THRESHOLD = 0.5


def confidence_tier(
    confidence: RecordConfidence, *, low_threshold: float = DEFAULT_LOW_THRESHOLD
) -> str:
    """Bucket a record's confidence into a stratification tier.

    Args:
        confidence: The record's `RecordConfidence` (see `record_confidence`).
        low_threshold: Median probability at or below which a record is
            `"low"` confidence. Defaults to 0.5 -- the midpoint of the
            probability domain, the least arbitrary "no information" fallback
            available -- but this is a policy choice, not a statistical
            derivation: it belongs in a run's recorded provenance, the same
            way `tau` is captured via `attest.ensemble.tau.TauReport`, not
            silently left implicit. It is NOT hash-versioned into
            `attest.provenance.config.Config`/`ensemble_config_id` the way
            `tau`/`temperature`/`model_version` are, because it never changes
            what a vendor samples or the ensemble's own aggregate decision
            (`attest.ensemble.aggregate.g`) -- it only rebuckets an
            already-fixed excluded population for audit stratification,
            structurally the same "downstream of the sample, not part of it"
            category as `request_logprobs` (see
            `attest.vendors.providers.openai.OpenAIRater.request_logprobs`).

    Returns:
        `UNSCORED_TIER` if `confidence.scored` is False (insufficient
        vendor coverage -- see `MIN_SUPPORTING_VOTES`); otherwise `"low"`
        if `median_probability <= low_threshold`, else `"high"`.

    Raises:
        ConfidenceError: If `low_threshold` is outside `[0, 1]`.
    """
    if not (0.0 <= low_threshold <= 1.0):
        raise ConfidenceError(f"low_threshold must be in [0, 1], got {low_threshold}")
    if not confidence.scored:
        return UNSCORED_TIER
    return TIER_LOW if confidence.median_probability <= low_threshold else TIER_HIGH


__all__ = [
    "DEFAULT_LOW_THRESHOLD",
    "MIN_SUPPORTING_VOTES",
    "TIER_HIGH",
    "TIER_LOW",
    "UNSCORED_TIER",
    "ConfidenceError",
    "RecordConfidence",
    "confidence_tier",
    "record_confidence",
    "vote_confidence",
]
