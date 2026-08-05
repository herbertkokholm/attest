"""Tests for attest.ensemble.confidence: per-vote logprob extraction and coverage-gated scoring."""

from __future__ import annotations

import math

import pytest

from attest.ensemble.confidence import (
    MIN_SUPPORTING_VOTES,
    UNSCORED_TIER,
    ConfidenceError,
    confidence_tier,
    record_confidence,
    vote_confidence,
)
from attest.ensemble.votes import build_vote_vector

_CONFIG_ID = "config-abc"


def _openai_shaped(*logprobs: float) -> dict:
    return {"logprobs": {"content": [{"token": "x", "logprob": lp} for lp in logprobs]}}


def _google_shaped(*logprobs: float, camel: bool = True) -> dict:
    key = "chosenCandidates" if camel else "chosen_candidates"
    field = "logProbability" if camel else "log_probability"
    return {"logprobs": {key: [{"token": "x", field: lp} for lp in logprobs]}}


# --- vote_confidence: openai-compatible shape (openai/mistral/fireworks/together) -------


@pytest.mark.parametrize("vendor", ["openai", "mistral", "fireworks", "together"])
def test_vote_confidence_single_token_openai_compatible(vendor: str) -> None:
    raw = _openai_shaped(-0.05)

    probability = vote_confidence(vendor, raw)

    assert probability == pytest.approx(math.exp(-0.05))


def test_vote_confidence_geometric_mean_across_multiple_tokens() -> None:
    # A reply that (still) tokenizes into more than one token is scored by
    # the geometric mean of its per-token probabilities, not their sum --
    # see attest.ensemble.confidence._geometric_mean_probability for why a
    # sum would systematically penalize vendors that split the same
    # single-letter answer into more tokens.
    raw = _openai_shaped(-0.02, -0.01)

    probability = vote_confidence("openai", raw)

    assert probability == pytest.approx(math.exp((-0.02 + -0.01) / 2))


def test_vote_confidence_skips_framing_tokens() -> None:
    # A leading whitespace token some vendors emit before the answer is
    # framing, not part of the answer, and must not dilute the mean.
    raw = {
        "logprobs": {
            "content": [
                {"token": " ", "logprob": -5.0},
                {"token": "I", "logprob": -0.05},
            ]
        }
    }

    assert vote_confidence("openai", raw) == pytest.approx(math.exp(-0.05))


def test_vote_confidence_all_framing_tokens_returns_none() -> None:
    raw = {
        "logprobs": {
            "content": [
                {"token": " ", "logprob": -1.0},
                {"token": "\n", "logprob": -2.0},
            ]
        }
    }

    assert vote_confidence("openai", raw) is None


def test_vote_confidence_missing_content_returns_none() -> None:
    assert vote_confidence("openai", {"logprobs": {"content": []}}) is None
    assert vote_confidence("openai", {"logprobs": {}}) is None


def test_vote_confidence_malformed_logprob_value_returns_none() -> None:
    raw = {"logprobs": {"content": [{"token": "x", "logprob": "not-a-number"}]}}

    assert vote_confidence("openai", raw) is None


def test_vote_confidence_missing_logprobs_key_returns_none() -> None:
    assert vote_confidence("openai", {"text": "1"}) is None


def test_vote_confidence_non_mapping_raw_response_returns_none() -> None:
    assert vote_confidence("openai", None) is None
    assert vote_confidence("openai", "not-a-dict") is None


def test_vote_confidence_clamps_to_one() -> None:
    # Two near-zero (i.e. near-certain) logprobs could sum fractionally above
    # 0.0 due to float noise; exp() of that must still clamp to 1.0.
    raw = _openai_shaped(1e-16, 1e-16)

    assert vote_confidence("openai", raw) == 1.0


# --- vote_confidence: google (best-effort, unconfirmed shape) --------------------------


def test_vote_confidence_google_camel_case() -> None:
    raw = _google_shaped(-0.1, camel=True)

    assert vote_confidence("google", raw) == pytest.approx(math.exp(-0.1))


def test_vote_confidence_google_snake_case() -> None:
    raw = _google_shaped(-0.1, camel=False)

    assert vote_confidence("google", raw) == pytest.approx(math.exp(-0.1))


def test_vote_confidence_google_str_fallback_returns_none() -> None:
    # _serialize_logprobs (attest.vendors.providers.google) falls back to
    # str() when the SDK object has neither to_dict nor model_dump.
    raw = {"logprobs": "<GeminiLogprobsResult object at 0x...>"}

    assert vote_confidence("google", raw) is None


def test_vote_confidence_google_unrecognized_shape_returns_none() -> None:
    raw = {"logprobs": {"somethingElse": []}}

    assert vote_confidence("google", raw) is None


# --- vote_confidence: vendors with no confidence signal ---------------------------------


def test_vote_confidence_anthropic_always_none() -> None:
    # AnthropicRater never populates raw_response["logprobs"] -- but even a
    # raw_response that somehow carried one must still return None, since
    # anthropic has no registered extractor at all.
    assert vote_confidence("anthropic", {"text": "1"}) is None
    assert vote_confidence("anthropic", _openai_shaped(-0.01)) is None


def test_vote_confidence_unknown_vendor_returns_none() -> None:
    assert vote_confidence("some-future-vendor", _openai_shaped(-0.01)) is None


def test_vote_confidence_request_logprobs_false_returns_none() -> None:
    # No "logprobs" key at all is what every provider leaves in raw_response
    # when request_logprobs=False.
    assert vote_confidence("openai", {"text": "1", "id": "abc", "model": "gpt-5"}) is None


# --- record_confidence: coverage gate ----------------------------------------------------


def _raw_responses(**per_vendor: dict | None) -> dict:
    return per_vendor


def test_record_confidence_scored_at_exactly_three_supporting_votes() -> None:
    votes = build_vote_vector("r1", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1})
    raw = _raw_responses(
        openai=_openai_shaped(-0.1),
        mistral=_openai_shaped(-0.2),
        together=_openai_shaped(-0.3),
    )

    confidence = record_confidence(votes, raw)

    assert confidence.scored is True
    assert confidence.n_supporting == 3
    assert confidence.n_total == 3
    assert confidence.median_probability == pytest.approx(math.exp(-0.2))


def test_record_confidence_unscored_below_minimum_coverage() -> None:
    assert MIN_SUPPORTING_VOTES == 3
    votes = build_vote_vector("r2", _CONFIG_ID, {"openai": 1, "mistral": 1})
    raw = _raw_responses(openai=_openai_shaped(-0.1), mistral=_openai_shaped(-0.2))

    confidence = record_confidence(votes, raw)

    assert confidence.scored is False
    assert confidence.n_supporting == 2
    assert confidence.median_probability == 0.0


def test_record_confidence_excludes_non_supporting_vendor_but_counts_it_in_n_total() -> None:
    # anthropic contributes an ordinal vote (still counted in n_total /
    # tau/aggregate elsewhere) but no confidence figure.
    votes = build_vote_vector(
        "r3", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1, "anthropic": 1}
    )
    raw = _raw_responses(
        openai=_openai_shaped(-0.1),
        mistral=_openai_shaped(-0.2),
        together=_openai_shaped(-0.3),
        anthropic={"text": "1"},
    )

    confidence = record_confidence(votes, raw)

    assert confidence.scored is True
    assert confidence.n_supporting == 3
    assert confidence.n_total == 4


def test_record_confidence_median_is_robust_to_one_miscalibrated_vendor() -> None:
    # A single wildly-outlying vendor (near-zero confidence) must not drag
    # the median the way it would drag a mean -- this is the whole point of
    # requiring >=3 supporting votes (see docs/logprob_support.md).
    votes = build_vote_vector("r4", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1})
    raw = _raw_responses(
        openai=_openai_shaped(math.log(0.95)),
        mistral=_openai_shaped(math.log(0.90)),
        together=_openai_shaped(math.log(0.01)),  # the outlier
    )

    confidence = record_confidence(votes, raw)

    mean_probability = (0.95 + 0.90 + 0.01) / 3
    assert confidence.median_probability == pytest.approx(0.90)
    assert confidence.median_probability > mean_probability


def test_record_confidence_invariant_to_tokenization_of_the_same_per_token_probability() -> None:
    # The whole point of moving OUTPUT_CONTRACT to single-token E/U/I
    # symbols: a vendor whose tokenizer happens to split the reply into
    # more tokens must land on the same confidence as one that emits it as
    # a single token, given the same underlying per-token certainty. This
    # is the regression guard against the tier boundary drifting the way
    # "-1" splitting into two tokens used to drift it under the old
    # sum-based extractor.
    per_token_logprob = math.log(0.8)
    votes = build_vote_vector(
        "r-tokenization", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1}
    )
    single_token_raw = _raw_responses(
        openai=_openai_shaped(per_token_logprob),
        mistral=_openai_shaped(per_token_logprob),
        together=_openai_shaped(per_token_logprob),
    )
    two_token_raw = _raw_responses(
        openai=_openai_shaped(per_token_logprob, per_token_logprob),
        mistral=_openai_shaped(per_token_logprob, per_token_logprob),
        together=_openai_shaped(per_token_logprob, per_token_logprob),
    )

    single_token_confidence = record_confidence(votes, single_token_raw)
    two_token_confidence = record_confidence(votes, two_token_raw)

    assert single_token_confidence.median_probability == pytest.approx(
        two_token_confidence.median_probability
    )
    assert single_token_confidence.scored is True
    assert single_token_confidence.n_supporting == two_token_confidence.n_supporting == 3


def test_record_confidence_missing_raw_response_entry_treated_as_non_supporting() -> None:
    votes = build_vote_vector("r5", _CONFIG_ID, {"openai": 1, "mistral": 1})
    raw = _raw_responses(openai=_openai_shaped(-0.1))  # mistral entry absent

    confidence = record_confidence(votes, raw)

    assert confidence.n_supporting == 1
    assert confidence.scored is False


# --- confidence_tier ----------------------------------------------------------------------


def test_confidence_tier_unscored_regardless_of_threshold() -> None:
    votes = build_vote_vector("r6", _CONFIG_ID, {"openai": 1, "mistral": 1})
    raw = _raw_responses(openai=_openai_shaped(-0.01), mistral=_openai_shaped(-0.01))
    confidence = record_confidence(votes, raw)

    assert confidence_tier(confidence, low_threshold=0.99) == UNSCORED_TIER
    assert confidence_tier(confidence, low_threshold=0.0) == UNSCORED_TIER


def test_confidence_tier_low_and_high() -> None:
    votes = build_vote_vector("r7", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1})
    raw = _raw_responses(
        openai=_openai_shaped(math.log(0.2)),
        mistral=_openai_shaped(math.log(0.2)),
        together=_openai_shaped(math.log(0.2)),
    )
    confidence = record_confidence(votes, raw)
    assert confidence.median_probability == pytest.approx(0.2)

    assert confidence_tier(confidence, low_threshold=0.5) == "low"
    assert confidence_tier(confidence, low_threshold=0.1) == "high"


def test_confidence_tier_default_threshold_is_the_probability_midpoint() -> None:
    votes = build_vote_vector("r7b", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1})
    below = record_confidence(
        votes,
        _raw_responses(
            openai=_openai_shaped(math.log(0.4)),
            mistral=_openai_shaped(math.log(0.4)),
            together=_openai_shaped(math.log(0.4)),
        ),
    )
    above = record_confidence(
        votes,
        _raw_responses(
            openai=_openai_shaped(math.log(0.6)),
            mistral=_openai_shaped(math.log(0.6)),
            together=_openai_shaped(math.log(0.6)),
        ),
    )

    assert confidence_tier(below) == "low"
    assert confidence_tier(above) == "high"


def test_confidence_tier_boundary_is_inclusive_on_low_side() -> None:
    votes = build_vote_vector("r8", _CONFIG_ID, {"openai": 1, "mistral": 1, "together": 1})
    raw = _raw_responses(
        openai=_openai_shaped(math.log(0.5)),
        mistral=_openai_shaped(math.log(0.5)),
        together=_openai_shaped(math.log(0.5)),
    )
    confidence = record_confidence(votes, raw)

    assert confidence_tier(confidence, low_threshold=0.5) == "low"


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_confidence_tier_rejects_threshold_outside_unit_interval(threshold: float) -> None:
    votes = build_vote_vector("r9", _CONFIG_ID, {"openai": 1})
    confidence = record_confidence(votes, _raw_responses(openai=_openai_shaped(-0.1)))

    with pytest.raises(ConfidenceError):
        confidence_tier(confidence, low_threshold=threshold)
