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

from collections.abc import Callable, Mapping

from attest.provenance.config import Config, VendorSpec
from attest.vendors.base import Rater
from attest.vendors.batch import BatchRater


def _build_anthropic(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.anthropic import AnthropicRater

    return AnthropicRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_openai(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.openai import OpenAIRater

    return OpenAIRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_google(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.google import GoogleRater

    return GoogleRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_openmodel(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.openmodel import OpenModelRater

    return OpenModelRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_mistral(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.mistral import MistralRater

    return MistralRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_fireworks(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.fireworks import FireworksRater

    return FireworksRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_together(spec: VendorSpec) -> Rater:
    from attest.vendors.providers.together import TogetherRater

    return TogetherRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


_PROVIDER_FACTORIES: dict[str, Callable[[VendorSpec], Rater]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
    "google": _build_google,
    "openmodel": _build_openmodel,
    "mistral": _build_mistral,
    "fireworks": _build_fireworks,
    "together": _build_together,
}


def _build_anthropic_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    return AnthropicBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_openai_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.openai import OpenAIBatchRater

    return OpenAIBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_google_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.google import GoogleBatchRater

    return GoogleBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_mistral_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.mistral import MistralBatchRater

    return MistralBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_fireworks_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.fireworks import FireworksBatchRater

    return FireworksBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


def _build_together_batch(spec: VendorSpec) -> BatchRater:
    from attest.vendors.providers.together import TogetherBatchRater

    return TogetherBatchRater(
        model=spec.model, model_version=spec.model_version, temperature=spec.temperature
    )


_BATCH_PROVIDER_FACTORIES: dict[str, Callable[[VendorSpec], BatchRater]] = {
    "anthropic": _build_anthropic_batch,
    "openai": _build_openai_batch,
    "google": _build_google_batch,
    "mistral": _build_mistral_batch,
    "fireworks": _build_fireworks_batch,
    "together": _build_together_batch,
}


def build_raters(
    config: Config, *, factories: Mapping[str, Callable[[VendorSpec], Rater]] | None = None
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

    Returns:
        One `Rater` per vendor in `config.vendors`, in that mapping's
        iteration order.

    Raises:
        KeyError: If a vendor named in `config.vendors` has no matching
            factory, built-in or supplied via `factories`.
    """
    merged: dict[str, Callable[[VendorSpec], Rater]] = {
        **_PROVIDER_FACTORIES,
        **(factories or {}),
    }
    raters: list[Rater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no rater factory for vendor '{vendor}'; known vendors: {known}")
        raters.append(factory(spec))
    return raters


def build_batch_raters(
    config: Config, *, factories: Mapping[str, Callable[[VendorSpec], BatchRater]] | None = None
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
    merged: dict[str, Callable[[VendorSpec], BatchRater]] = {
        **_BATCH_PROVIDER_FACTORIES,
        **(factories or {}),
    }
    raters: list[BatchRater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no batch rater factory for vendor '{vendor}'; known vendors: {known}")
        raters.append(factory(spec))
    return raters
