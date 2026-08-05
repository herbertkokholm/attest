"""Live probe: does vendor X, model Y actually return logprobs -- in sync and in batch?

Empirical, not assumed: vendor APIs commonly ignore an unsupported request
parameter instead of erroring, so the only reliable check is inspecting the
actual response payload for a populated `logprobs` field. Sync and batch are
checked separately and never assumed to have parity -- batch endpoints
commonly lag sync endpoints in parameter support.

Unlike `tools/sentinel_drift_probe.py` (pure offline, safe to run anytime),
this script makes live, billed API calls against each vendor's real endpoint
and needs that vendor's API key set in the environment (see each provider's
`_client` in `attest.vendors.providers.*` for which env var). It is not
wired into pytest/CI -- run it manually, and only against a model you intend
to actually configure a `VendorSpec` with.

`anthropic` is accepted as a `--vendor` value and always reports "not
supported" rather than being silently rejected: the Messages API has no
logprobs equivalent as of this writing (see
`attest.vendors.providers.anthropic.AnthropicRater`'s docstring).

Vendor batch APIs are asynchronous and can take anywhere from minutes to
(per some vendors' stated SLA) 24 hours to complete, so this script does not
block waiting for one. `batch-submit` submits a job and persists a
`BatchHandle` to `--handle-file`; a later `batch-fetch` invocation (same
`--vendor`/`--model`/`--model-version`, so it reconstructs an equivalent
rater) polls that handle once and reports whether it's done yet.

Usage:
    python tools/vendor_logprob_probe.py sync --vendor openai --model gpt-4o
    python tools/vendor_logprob_probe.py batch-submit --vendor openai \\
        --model gpt-4o --handle-file /tmp/probe-openai.json
    python tools/vendor_logprob_probe.py batch-fetch --vendor openai \\
        --model gpt-4o --handle-file /tmp/probe-openai.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attest.contracts.input import Record  # noqa: E402
from attest.vendors.batch import BatchHandle  # noqa: E402

_PROBE_RECORD = Record(
    id="logprob-probe-1",
    title="A randomized controlled trial of a novel intervention",
    abstract="This study reports outcomes from a randomized controlled trial.",
    track="probe",
)

_SYNC_RATER_CLASSES = {
    "openai": ("attest.vendors.providers.openai", "OpenAIRater"),
    "mistral": ("attest.vendors.providers.mistral", "MistralRater"),
    "google": ("attest.vendors.providers.google", "GoogleRater"),
    "fireworks": ("attest.vendors.providers.fireworks", "FireworksRater"),
    "together": ("attest.vendors.providers.together", "TogetherRater"),
}

_BATCH_RATER_CLASSES = {
    "openai": ("attest.vendors.providers.openai", "OpenAIBatchRater"),
    "mistral": ("attest.vendors.providers.mistral", "MistralBatchRater"),
    "google": ("attest.vendors.providers.google", "GoogleBatchRater"),
    "together": ("attest.vendors.providers.together", "TogetherBatchRater"),
}

# Vendors valid as --vendor but with no logprobs-capable batch rater
# (permanently, or -- like fireworks -- for now): reported explicitly by
# cmd_batch_submit/cmd_batch_fetch rather than raising a raw KeyError.
_BATCH_UNSUPPORTED_REASONS = {
    "anthropic": "Messages API has no logprobs equivalent",
    "fireworks": (
        "FireworksBatchRater has no request_logprobs field yet -- its own row "
        "schema is unconfirmed, see docs/logprob_support.md"
    ),
}


def _load_class(module_name: str, class_name: str) -> Any:
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def _report(label: str, logprobs: Any, extra: str = "") -> None:
    if logprobs is None:
        print(f"[{label}] logprobs NOT present in response{extra}")
    else:
        print(f"[{label}] logprobs present{extra}")
        print(f"  {logprobs!r}")


def cmd_sync(args: argparse.Namespace) -> int:
    if args.vendor == "anthropic":
        print(f"[{args.vendor}/sync] not supported: Messages API has no logprobs equivalent")
        return 0

    rater_cls = _load_class(*_SYNC_RATER_CLASSES[args.vendor])
    rater = rater_cls(
        model=args.model,
        model_version=args.model_version or args.model,
        temperature=0.0,
        request_logprobs=True,
        top_logprobs=args.top_logprobs,
    )
    try:
        ordinal, raw = rater.rate(_PROBE_RECORD)
    except Exception as exc:  # noqa: BLE001 -- diagnostic tool: report and exit, don't crash
        print(f"[{args.vendor}/sync] call failed: {exc!r}")
        return 1

    extra = f" (ordinal={ordinal}, text={raw.get('text')!r})"
    _report(f"{args.vendor}/sync", raw.get("logprobs"), extra)
    return 0


def cmd_batch_submit(args: argparse.Namespace) -> int:
    if args.vendor in _BATCH_UNSUPPORTED_REASONS:
        print(f"[{args.vendor}/batch] not supported: {_BATCH_UNSUPPORTED_REASONS[args.vendor]}")
        return 0

    rater_cls = _load_class(*_BATCH_RATER_CLASSES[args.vendor])
    rater = rater_cls(
        model=args.model,
        model_version=args.model_version or args.model,
        temperature=0.0,
        request_logprobs=True,
        top_logprobs=args.top_logprobs,
    )
    handle = rater.submit([_PROBE_RECORD], ensemble_config_id="logprob-probe")
    Path(args.handle_file).write_text(json.dumps(handle.to_dict()))
    print(
        f"[{args.vendor}/batch] submitted {handle.provider_batch_id!r}, "
        f"handle written to {args.handle_file}"
    )
    print(f"[{args.vendor}/batch] re-run 'batch-fetch' with the same --model later to check it")
    return 0


def cmd_batch_fetch(args: argparse.Namespace) -> int:
    if args.vendor in _BATCH_UNSUPPORTED_REASONS:
        print(f"[{args.vendor}/batch] not supported: {_BATCH_UNSUPPORTED_REASONS[args.vendor]}")
        return 0

    handle = BatchHandle.from_dict(json.loads(Path(args.handle_file).read_text()))
    rater_cls = _load_class(*_BATCH_RATER_CLASSES[args.vendor])
    rater = rater_cls(
        model=args.model,
        model_version=args.model_version or args.model,
        temperature=0.0,
        request_logprobs=True,
        top_logprobs=args.top_logprobs,
    )
    status = rater.poll(handle)
    if status != "completed":
        print(f"[{args.vendor}/batch] not done yet (status={status}); re-run batch-fetch later")
        return 0
    results = rater.fetch(handle)
    result = results.get(_PROBE_RECORD.id)
    if result is None:
        print(f"[{args.vendor}/batch] completed, but no result for the probe record")
        return 1
    ordinal, raw = result
    logprobs = raw.get("logprobs") if isinstance(raw, dict) else None
    extra = f" (ordinal={ordinal})"
    _report(f"{args.vendor}/batch", logprobs, extra)
    return 0


def _add_common_args(parser: argparse.ArgumentParser, *, needs_top_logprobs: bool = True) -> None:
    parser.add_argument(
        "--vendor",
        required=True,
        choices=sorted({*_SYNC_RATER_CLASSES, "anthropic"}),
    )
    parser.add_argument("--model", required=True, help="Vendor model identifier, e.g. 'gpt-4o'.")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Expected resolved model version; defaults to --model. A mismatch "
        "raises ModelVersionDriftError before logprobs can be inspected -- "
        "pass the exact resolved snapshot id if --model is a floating alias.",
    )
    if needs_top_logprobs:
        parser.add_argument("--top-logprobs", type=int, default=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="One-shot sync call, checked immediately.")
    _add_common_args(sync_parser)
    sync_parser.set_defaults(handler=cmd_sync)

    submit_parser = subparsers.add_parser("batch-submit", help="Submit a 1-record batch job.")
    _add_common_args(submit_parser)
    submit_parser.add_argument(
        "--handle-file", required=True, help="Path to persist the batch handle to."
    )
    submit_parser.set_defaults(handler=cmd_batch_submit)

    fetch_parser = subparsers.add_parser(
        "batch-fetch", help="Poll and, if done, fetch a submitted batch job."
    )
    _add_common_args(fetch_parser)
    fetch_parser.add_argument(
        "--handle-file", required=True, help="Path a prior batch-submit wrote to."
    )
    fetch_parser.set_defaults(handler=cmd_batch_fetch)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
