"""Builds the set of raters for an ensemble configuration.

Maps each vendor name in a `Config` to a `Rater` or `BatchRater`, importing
provider adapter modules lazily inside each factory so that constructing a
registry -- or building raters for a configuration that only uses a subset
of vendors -- never requires an extra that is not installed for a vendor not
in use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from attest.provenance.config import Config
from attest.vendors.base import Rater
from attest.vendors.batch import BatchRater


def _build_anthropic(model: str) -> Rater:
    from attest.vendors.providers.anthropic import AnthropicRater

    return AnthropicRater(model=model)


def _build_openai(model: str) -> Rater:
    from attest.vendors.providers.openai import OpenAIRater

    return OpenAIRater(model=model)


def _build_google(model: str) -> Rater:
    from attest.vendors.providers.google import GoogleRater

    return GoogleRater(model=model)


def _build_openmodel(model: str) -> Rater:
    from attest.vendors.providers.openmodel import OpenModelRater

    return OpenModelRater(model=model)


_PROVIDER_FACTORIES: dict[str, Callable[[str], Rater]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
    "google": _build_google,
    "openmodel": _build_openmodel,
}


def _build_anthropic_batch(model: str) -> BatchRater:
    from attest.vendors.providers.anthropic import AnthropicBatchRater

    return AnthropicBatchRater(model=model)


def _build_openai_batch(model: str) -> BatchRater:
    from attest.vendors.providers.openai import OpenAIBatchRater

    return OpenAIBatchRater(model=model)


def _build_google_batch(model: str) -> BatchRater:
    from attest.vendors.providers.google import GoogleBatchRater

    return GoogleBatchRater(model=model)


_BATCH_PROVIDER_FACTORIES: dict[str, Callable[[str], BatchRater]] = {
    "anthropic": _build_anthropic_batch,
    "openai": _build_openai_batch,
    "google": _build_google_batch,
}


def build_raters(
    config: Config, *, factories: Mapping[str, Callable[[str], Rater]] | None = None
) -> list[Rater]:
    """Build one rater per vendor named in `config.vendors`.

    Args:
        config: The ensemble configuration naming which vendors participate
            and which model each one uses.
        factories: Optional mapping of vendor name to a rater factory,
            overriding the built-in provider factories for matching vendor
            names. Chiefly useful in tests, to substitute a
            `DeterministicRater` for a live provider adapter without
            needing that vendor's SDK installed.

    Returns:
        One `Rater` per vendor in `config.vendors`, in that mapping's
        iteration order.

    Raises:
        KeyError: If a vendor named in `config.vendors` has no matching
            factory, built-in or supplied via `factories`.
    """
    merged: dict[str, Callable[[str], Rater]] = {**_PROVIDER_FACTORIES, **(factories or {})}
    raters: list[Rater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no rater factory for vendor '{vendor}'; known vendors: {known}")
        raters.append(factory(spec.model))
    return raters


def build_batch_raters(
    config: Config, *, factories: Mapping[str, Callable[[str], BatchRater]] | None = None
) -> list[BatchRater]:
    """Build one `BatchRater` per vendor named in `config.vendors`.

    Args:
        config: The ensemble configuration naming which vendors participate
            and which model each one uses.
        factories: Optional mapping of vendor name to a batch rater factory,
            overriding the built-in provider factories for matching vendor
            names. Chiefly useful in tests, to substitute a
            `DeterministicBatchRater` for a live provider adapter without
            needing that vendor's SDK installed.

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
    merged: dict[str, Callable[[str], BatchRater]] = {
        **_BATCH_PROVIDER_FACTORIES,
        **(factories or {}),
    }
    raters: list[BatchRater] = []
    for vendor, spec in config.vendors.items():
        factory = merged.get(vendor)
        if factory is None:
            known = sorted(merged)
            raise KeyError(f"no batch rater factory for vendor '{vendor}'; known vendors: {known}")
        raters.append(factory(spec.model))
    return raters
