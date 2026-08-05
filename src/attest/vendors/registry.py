"""Builds the set of raters for an ensemble configuration.

Maps each vendor name in a `Config` to a `Rater` or `BatchRater`, importing
provider adapter modules lazily inside each factory so that constructing a
registry -- or building raters for a configuration that only uses a subset
of vendors -- never requires an extra that is not installed for a vendor not
in use. Factories receive the vendor's whole `VendorSpec`, not just its
`model`, so `model_version` and `temperature` reach the constructed rater
and are actually applied by it (see `attest.vendors.providers.*`) rather
than being versioned/hashed metadata the rater never sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from attest.provenance.config import Config, VendorSpec
from attest.vendors.base import Rater
from attest.vendors.batch import BatchRater


class RaterFactory(Protocol):
    """A vendor's `Rater` factory: `VendorSpec` in, optional logprobs request in, `Rater` out."""

    def __call__(self, spec: VendorSpec, *, request_logprobs: bool = False) -> Rater: ...


class BatchRaterFactory(Protocol):
    """A vendor's `BatchRater` factory: `VendorSpec` and an optional logprobs request in,
    `BatchRater` out.
    """

    def __call__(self, spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater: ...


def _build_anthropic(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.anthropic import AnthropicRater

    # request_logprobs is accepted for signature parity with the other
    # factories but never forwarded: AnthropicRater has no such field, since
    # the Messages API has no logprobs equivalent to apply it to (see
    # AnthropicRater's docstring). Adding an unused kwarg here would be
    # exactly the decorative-field pattern this codebase avoids elsewhere.
    return AnthropicRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_openai(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.openai import OpenAIRater

    return OpenAIRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_google(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.google import GoogleRater

    return GoogleRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_openmodel(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.openmodel import OpenModelRater

    # request_logprobs is accepted for signature parity but not forwarded:
    # self-hosted OpenAI-compatible servers vary in logprobs support, and
    # OpenModelRater does not yet expose the field -- see
    # AnthropicRater/_build_anthropic for why an unforwarded kwarg here is
    # preferable to a field that may not be honored.
    return OpenModelRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_mistral(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.mistral import MistralRater

    return MistralRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_fireworks(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.fireworks import FireworksRater

    return FireworksRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_together(spec: VendorSpec, *, request_logprobs: bool = False) -> Rater:
    from attest.vendors.providers.together import TogetherRater

    return TogetherRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


_PROVIDER_FACTORIES: dict[str, RaterFactory] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
    "google": _build_google,
    "openmodel": _build_openmodel,
    "mistral": _build_mistral,
    "fireworks": _build_fireworks,
    "together": _build_together,
}


def _build_anthropic_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    # See _build_anthropic: request_logprobs is accepted for signature
    # parity but never forwarded, since AnthropicBatchRater has no such
    # field.
    return AnthropicBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_openai_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.openai import OpenAIBatchRater

    return OpenAIBatchRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_google_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.google import GoogleBatchRater

    return GoogleBatchRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_mistral_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.mistral import MistralBatchRater

    return MistralBatchRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


def _build_fireworks_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.fireworks import FireworksBatchRater

    # request_logprobs accepted for signature parity, not yet forwarded.
    # Unlike the sync path, Fireworks' Batch Inference API has a genuinely
    # different, only partly-confirmed row schema (see
    # attest.vendors.providers.fireworks module docstring) -- adding
    # logprobs here means guessing at an already-unconfirmed output shape,
    # so this is deliberately deferred until that schema itself is verified
    # against a live batch run. See docs/logprob_support.md.
    return FireworksBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_together_batch(spec: VendorSpec, *, request_logprobs: bool = False) -> BatchRater:
    from attest.vendors.providers.together import TogetherBatchRater

    return TogetherBatchRater(
        model=spec.model,
        model_version=spec.model_version,
        temperature=spec.temperature,
        request_logprobs=request_logprobs,
    )


_BATCH_PROVIDER_FACTORIES: dict[str, BatchRaterFactory] = {
    "anthropic": _build_anthropic_batch,
    "openai": _build_openai_batch,
    "google": _build_google_batch,
    "mistral": _build_mistral_batch,
    "fireworks": _build_fireworks_batch,
    "together": _build_together_batch,
}


def build_raters(
    config: Config,
    *,
    factories: Mapping[str, RaterFactory] | None = None,
    request_logprobs: bool = False,
) -> list[Rater]:
    """Build one rater per vendor named in `config.vendors`.

    Args:
        config: The ensemble configuration naming which vendors participate
            and each one's `VendorSpec` (model, model_version, prompt_version,
            temperature).
        factories: Optional mapping of vendor name to a rater factory taking
            that vendor's `VendorSpec`, overriding the built-in provider
            factories for matching vendor names. Chiefly useful in tests, to
            substitute a `DeterministicRater` for a live provider adapter
            without needing that vendor's SDK installed.
        request_logprobs: If True, ask every factory that supports it to
            build a rater with per-token log probabilities requested (see
            `attest.vendors.providers.openai.OpenAIRater.request_logprobs`).
            Forwarded only when True, so a custom `factories` override that
            takes just `(spec)` -- as every pre-existing caller's did --
            keeps working unchanged at the default. Not every vendor
            supports this: Anthropic's factory accepts and ignores it (see
            `_build_anthropic`), since the Messages API has no logprobs
            equivalent to apply it to.

    Returns:
        One `Rater` per vendor in `config.vendors`, in that mapping's
        iteration order.

    Raises:
        KeyError: If a vendor named in `config.vendors` has no matching
            factory, built-in or supplied via `factories`.
    """
    merged: dict[str, RaterFactory] = {
        **_PROVIDER_FACTORIES,
        **(factories or {}),
    }
    raters: list[Rater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no rater factory for vendor '{vendor}'; known vendors: {known}")
        rater = (
            factory(spec, request_logprobs=request_logprobs) if request_logprobs else factory(spec)
        )
        raters.append(rater)
    return raters


def build_batch_raters(
    config: Config,
    *,
    factories: Mapping[str, BatchRaterFactory] | None = None,
    request_logprobs: bool = False,
) -> list[BatchRater]:
    """Build one `BatchRater` per vendor named in `config.vendors`.

    Args:
        config: The ensemble configuration naming which vendors participate
            and each one's `VendorSpec` (model, model_version, prompt_version,
            temperature).
        factories: Optional mapping of vendor name to a batch rater factory
            taking that vendor's `VendorSpec`, overriding the built-in
            provider factories for matching vendor names. Chiefly useful in
            tests, to substitute a `DeterministicBatchRater` for a live
            provider adapter without needing that vendor's SDK installed.
        request_logprobs: If True, ask every factory that supports it to
            build a batch rater with per-token log probabilities requested
            on every submitted request. See `build_raters`'s
            `request_logprobs` for why this is forwarded only when True, and
            why not every vendor honors it.

    Returns:
        One `BatchRater` per vendor in `config.vendors`, in that mapping's
        iteration order.

    Raises:
        KeyError: If a vendor named in `config.vendors` has no matching
            batch factory, built-in or supplied via `factories` -- e.g.
            "openmodel", which has no built-in batch adapter since
            self-hosted OpenAI-compatible servers do not generally expose a
            batch endpoint.
    """
    merged: dict[str, BatchRaterFactory] = {
        **_BATCH_PROVIDER_FACTORIES,
        **(factories or {}),
    }
    raters: list[BatchRater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no batch rater factory for vendor '{vendor}'; known vendors: {known}")
        rater = (
            factory(spec, request_logprobs=request_logprobs) if request_logprobs else factory(spec)
        )
        raters.append(rater)
    return raters
