"""Immutable ensemble configuration and its content-derived ensemble_config_id.

An ensemble configuration is the full set of choices that determine how
records are screened: which vendors vote, which model and prompt version
each vendor uses, and how their votes are aggregated. The `ensemble_config_id`
is a content hash of that configuration, so two configurations with the same
content always resolve to the same id regardless of when or in what order
they were constructed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from attest.ensemble.aggregate import KNOWN_ZERO_POLICIES, ZERO_POLICY_ESCALATE
from attest.ensemble.confidence import DEFAULT_LOW_THRESHOLD

# Version of the kernel-owned output-format contract appended to every
# screening prompt (see `attest.vendors.base.OUTPUT_CONTRACT`). Defined here,
# not in `attest.vendors.base`, because `Config.to_dict` needs it for hashing
# and `attest.vendors.base` already imports from this module -- importing it
# back the other way would be circular. `attest.vendors.base` re-exports it.
OUTPUT_CONTRACT_VERSION = "1"

# Version of the kernel-owned multi-record output-format contract used when
# `batch_size > 1` packs more than one record into a single screening request
# (see `attest.vendors.base.BATCH_OUTPUT_CONTRACT`). Defined here for the same
# circularity reason as `OUTPUT_CONTRACT_VERSION`, and re-exported from
# `attest.vendors.base`. Included in `Config.to_dict` only when `batch_size >
# 1` -- see `Config.to_dict` -- so a `batch_size == 1` configuration hashes
# identically whether or not this constant exists, preserving its bytes.
BATCH_OUTPUT_CONTRACT_VERSION = "1"

# Convention (not enforced elsewhere) for a config field whose real value is
# not yet known -- e.g. a served model snapshot string that must be captured
# from a live vendor before a run is frozen. `VendorSpec.__post_init__`
# refuses to construct with one of these left in place, so a placeholder can
# never silently reach a live screening run: a content hash of "TODO:..." is
# provenance for a plan, not for the instrument that actually ran.
TODO_PLACEHOLDER_PREFIX = "TODO:"


@dataclass(frozen=True)
class VendorSpec:
    """One ensemble member's model, prompt, and sampling versioning.

    Attributes:
        model: Model identifier used by this vendor (e.g. "gpt-4o").
        model_version: Version string of the model.
        prompt_version: Version identifier of the screening prompt used with
            this vendor.
        temperature: Sampling temperature used with this vendor. Versioned
            exactly like `model_version` and `prompt_version`: it is
            hash-sensitive, so changing it yields a different
            `ensemble_config_id` and opens a new epoch, the same as a model
            or prompt change would.
        reasoning_effort: Forwarded to `attest.vendors.providers.openai`'s
            and `attest.vendors.providers.openmodel`'s raters as the Chat
            Completions API's `reasoning_effort` parameter when not `None`;
            ignored by every other provider. Exists because some
            current-generation OpenAI models reject an
            explicit non-default `temperature` outright (HTTP 400) unless
            paired with `reasoning_effort="none"` -- confirmed empirically
            against `gpt-5.6-terra` on 2026-08-24, not assumed from
            documentation (OpenAI's docs describe the coupling but not which
            models it applies to). `None` (the default) omits the parameter
            from the request entirely, reproducing every prior model's
            behavior exactly. Hash-versioned like `temperature` -- it changes
            what the vendor actually samples -- but omitted from `to_dict()`
            when `None`, so a config that never sets it hashes identically
            to before this field existed.
        send_temperature: When `False`, `attest.vendors.providers.anthropic`'s
            and `attest.vendors.providers.openmodel`'s raters omit
            `temperature` from the request entirely instead of sending
            `self.temperature` and letting the vendor reject it. Exists
            because Claude Sonnet 5 (and the rest of the Claude 4.6+
            generation) returns HTTP 400 on any explicit
            `temperature`/`top_p`/`top_k` value -- confirmed against
            Anthropic's own current model documentation, 2026-08-24 -- with
            no parameter analogous to OpenAI's `reasoning_effort="none"` that
            re-enables it; omission is the only way to avoid the error.
            Ignored by every other provider. Defaults to `True` (send
            `temperature` as before this field existed) and is omitted from
            `to_dict()` at that default, so an existing config hashes
            unchanged. `temperature` itself stays required on `VendorSpec`
            even when this is `False`, since it remains meaningful
            provenance (the value that was requested, for a model that
            silently cannot honor it) and other providers still consume it
            directly.
        base_url: Forwarded to `attest.vendors.providers.openmodel`'s raters
            as the OpenAI-compatible server to call; ignored by every other
            provider, whose endpoint is fixed by their own SDK. `None` (the
            default) leaves `OpenModelRater.base_url` at its own default
            (a local server), reproducing prior behavior exactly. Unlike
            `api_key_env`, this is hash-versioned: which server actually
            answers can mean different weights, quantization, or serving
            stack behind the same nominal `model` name, exactly the drift
            `model_version`/`check_model_version` exists to catch -- so
            pointing `openmodel` at a different `base_url` opens a new
            epoch. Omitted from `to_dict()` when `None`, so a config that
            never sets it hashes identically to before this field existed.
        api_key_env: Name of the environment variable to read this vendor's
            API key from, forwarded to
            `attest.vendors.providers.openmodel`'s raters; ignored by every
            other provider, each of which reads its own fixed env var
            (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) via its SDK's own
            default lookup. `openmodel` has no SDK and thus no fixed vendor
            identity to hang a fixed env var name off of, so the variable
            name itself is named here instead. Deliberately never included
            in `to_dict()` -- unlike every other field on this class, it is
            not provenance: it names where a secret lives, not a value that
            changes what the vendor samples, and hashing it would open a
            phantom epoch merely by moving the same key to a differently
            named variable. The key's actual value is never read into this
            frozen, JSON-serializable, hash-versioned config -- only looked
            up from the environment at rater-construction time, the same as
            every other provider's credential.
    """

    model: str
    model_version: str
    prompt_version: str
    temperature: float
    reasoning_effort: str | None = None
    send_temperature: bool = True
    base_url: str | None = None
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("model", self.model),
            ("model_version", self.model_version),
            ("prompt_version", self.prompt_version),
        ):
            if value.startswith(TODO_PLACEHOLDER_PREFIX):
                raise ValueError(
                    f"VendorSpec.{field_name} is an unresolved placeholder ('{value}'): "
                    "resolve it to the real value before this configuration can be used -- "
                    "a TODO-prefixed field must never reach a live screening run"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return this vendor spec as a plain dict.

        `reasoning_effort`, `send_temperature`, and `base_url` are included
        only when set to something other than their no-op default (`None`,
        `True`, and `None` respectively), so a `VendorSpec` that never
        touches any of them hashes identically to one constructed before
        these fields existed. `api_key_env` is never included, at any value
        -- see its docstring for why it is deliberately not provenance.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.send_temperature is not True:
            payload["send_temperature"] = self.send_temperature
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        return payload


@dataclass
class Config:
    """An ensemble configuration descriptor.

    Attributes:
        vendors: Mapping of vendor name to that vendor's model and prompt
            versioning. The ensemble size `x` is derived from this mapping,
            never set independently.
        aggregation: Name of the aggregation rule applied to ensemble votes.
        tau: Decision threshold used by the aggregation rule.
        default_prompt: Screening prompt text sent to every vendor for a
            record whose track has no entry in `track_prompts`. `None`
            (the default) means every rater falls back to its own
            hardcoded `attest.vendors.base.DEFAULT_SCREENING_PROMPT`,
            exactly as before this field existed.
        track_prompts: Mapping of record track (stringified) to the
            screening prompt text to use for records on that track,
            overriding `default_prompt`. Lets one ensemble configuration
            screen records from several systematic reviews in a single run,
            each against its own published eligibility criteria, while
            still voting with the same vendors/models/aggregation on all of
            them.
        zero_policy: Disposition of a would-be `auto_label == 0` decision
            (see `attest.ensemble.aggregate.g`). `ZERO_POLICY_ESCALATE` (the
            default) routes it to a human via escalation; `ZERO_POLICY_INCLUDE`
            folds it into `+1`. Validated at construction; only these two
            values are accepted -- there is deliberately no "exclude" option.
        batch_size: `b_e` -- the maximum number of records packed into a
            single vendor screening request (see manuscript Eq. 1 and
            Section 2.6). A control knob set from the runbook, not a
            provenance-only claim: `attest.vendors.base.run_ensemble` and
            `attest.vendors.batch.submit_batch` actually chunk records into
            requests of at most this size (grouped by resolved prompt first,
            so records on different tracks/criteria are never packed
            together; a final undersized chunk is valid). Defaults to `1`
            -- one record per request, this kernel's original behavior.
            Hashed unconditionally, on par with `vendors`/`aggregation`/
            `tau`/`x`, because per-request packing changes measured
            screening performance even under an unchanged model name: a
            `batch_size` change opens a new epoch the same as a model or
            prompt change would. Never checked against `len(kept)` or any
            other record count -- it is a packing width, not a corpus-size
            assertion.
        confidence_threshold: Default `low_threshold` (see
            `attest.ensemble.confidence.confidence_tier`) a confidence-
            stratified `audit-draw` uses when no `--confidence-threshold`
            override is given on that command's own invocation. Sourced
            from the same config file as `tau` for the same ergonomic
            reason -- a runbook setting the caller shouldn't have to retype
            identically on every invocation -- but, unlike `tau`,
            deliberately excluded from `to_dict()`/`ensemble_config_id`:
            see `attest.io.store._config_to_dict` for where it is instead
            persisted, and `docs/logprob_support.md` for why it must never
            be hash-versioned (it changes only how an already-fixed
            excluded population is stratified for audit, never what a
            vendor samples or the ensemble's own aggregate decision).
    """

    vendors: dict[str, VendorSpec] = field(default_factory=dict)
    aggregation: str = ""
    tau: float = 0.0
    batch_size: int = 1
    default_prompt: str | None = None
    track_prompts: dict[str, str] = field(default_factory=dict)
    zero_policy: str = ZERO_POLICY_ESCALATE
    confidence_threshold: float = DEFAULT_LOW_THRESHOLD

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.zero_policy not in KNOWN_ZERO_POLICIES:
            raise ValueError(
                f"unknown zero_policy '{self.zero_policy}': expected one of {KNOWN_ZERO_POLICIES}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @property
    def x(self) -> int:
        """Ensemble size, derived from the number of participating vendors."""
        return len(self.vendors)

    def prompt_for_track(self, track: int | str) -> str | None:
        """Resolve the screening prompt to use for a record on `track`.

        Args:
            track: A record's `track` field (see `attest.contracts.input.Record`).

        Returns:
            `track_prompts[str(track)]` if set, else `default_prompt`, else
            `None` -- meaning the rater should fall back to its own
            hardcoded default prompt.
        """
        return self.track_prompts.get(str(track), self.default_prompt)

    def to_dict(self) -> dict[str, Any]:
        """Return this configuration as a plain dict, including the derived `x`.

        `default_prompt`/`track_prompts` are omitted when unset, so a
        configuration that does not use them serializes -- and hashes into
        `compute_ensemble_config_id` -- identically to one built before
        these fields existed. `output_contract_version` is always included,
        regardless: the kernel-appended output contract (see
        `attest.vendors.base.compose_system_prompt`) is appended to *every*
        composed prompt, including the no-criteria fallback, so its version
        is always sensitive to the text actually sent to a vendor on this
        config's behalf. Bumping `OUTPUT_CONTRACT_VERSION` therefore opens a
        new epoch for every config, not just those supplying criteria.
        `zero_policy` is omitted when it is the default `ZERO_POLICY_ESCALATE`,
        so a config that never sets it hashes
        identically to one built before the field existed; a config that
        opts into `ZERO_POLICY_INCLUDE` picks up a different id, correctly
        opening a new epoch for that real behavior change.
        `confidence_threshold` is never included here -- unlike every other
        field on this class, it is not part of the hashed/versioned
        contract at all (see its attribute docstring above), so it is
        persisted separately by `attest.io.store._config_to_dict` instead
        of through this method.
        `batch_size` is always included, unconditionally, on par with
        `vendors`/`aggregation`/`tau`/`x`: it is `b_e` in the manuscript's
        `C_e` tuple (Eq. 1), not an optional add-on like `zero_policy`, so
        every change to it is hash-sensitive and opens a new epoch.
        `batch_output_contract_version` is included only when `batch_size >
        1`: the multi-record output contract it versions (see
        `attest.vendors.base.BATCH_OUTPUT_CONTRACT`) is only ever composed
        onto a request when more than one record is packed into it, so a
        `batch_size == 1` configuration -- which never touches that contract
        -- hashes identically whether or not this constant exists, keeping
        its `ensemble_config_id` byte-for-byte stable.
        """
        payload: dict[str, Any] = {
            "vendors": {name: spec.to_dict() for name, spec in self.vendors.items()},
            "aggregation": self.aggregation,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "x": self.x,
        }
        if self.default_prompt is not None:
            payload["default_prompt"] = self.default_prompt
        if self.track_prompts:
            payload["track_prompts"] = dict(self.track_prompts)
        payload["output_contract_version"] = OUTPUT_CONTRACT_VERSION
        if self.batch_size > 1:
            payload["batch_output_contract_version"] = BATCH_OUTPUT_CONTRACT_VERSION
        if self.zero_policy != ZERO_POLICY_ESCALATE:
            payload["zero_policy"] = self.zero_policy
        return payload


def compute_ensemble_config_id(config: Config) -> str:
    """Derive a stable, content-based identifier for an ensemble configuration.

    The id is a SHA-256 hex digest of the configuration's canonical JSON
    serialization: keys sorted recursively and compactly separated, so the
    same configuration content always yields the same id regardless of
    construction order or Python dict iteration order, and any change to any
    field yields a different id.

    Args:
        config: The ensemble configuration to identify.

    Returns:
        A hex-encoded SHA-256 digest of the configuration's canonical form.
    """
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
