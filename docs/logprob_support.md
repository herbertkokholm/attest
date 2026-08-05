# Per-vote logprobs: support matrix and the open fallback-value decision

Each ensemble member rates a record with a single ordinal token in
`{-1, 0, +1}` (`attest.vendors.base.OUTPUT_CONTRACT`). A per-vote logprob on
that token is a *within*-vendor confidence signal -- how sure this one
vendor was of its own answer -- complementary to the ensemble's existing
*between*-vendor dispersion measure (`tau`, in `attest.ensemble.tau`).
Candidate downstream uses (not implemented yet): stratifying `audit-draw`
toward low-confidence records, and enriching the latent-vendor-drift
sentinel (`docs/sentinel_drift_rule.md`) with a confidence delta alongside
label agreement.

## Design choices already made

- **Not a `VendorSpec` field.** `VendorSpec` (`attest.provenance.config`) is
  the hash-versioned hook for things that change what a vendor *samples*
  (`temperature`, `model_version`, `prompt_version`). Requesting logprobs
  changes only what side-channel metadata comes back with an unchanged
  sample, so it is a plain `request_logprobs: bool` constructor field on
  each supporting `Rater`/`BatchRater`, threaded through
  `attest.vendors.registry.build_raters`/`build_batch_raters`. Toggling it
  never opens a new ensemble-config epoch.
- **Un-normalized, vendor-native shape.** Each provider stashes its own raw
  logprob structure into `raw_response["logprobs"]`, matching how
  `raw_response` already varies slightly per vendor. `raw_response` is
  explicitly outside the versioned vote contract and retained for audit and
  debugging only (`attest.vendors.base.Rater.rate`), and
  `RunStore.write_raw_responses` already persists it to
  `raw_responses.json` -- so real logprob values are logged as soon as a
  rater is built with `request_logprobs=True`, no further plumbing needed.
- **Anthropic has no field, not a silent no-op field.** The Messages API
  exposes no per-token logprob equivalent as of this writing. Adding an
  unused `request_logprobs` field to `AnthropicRater`/`AnthropicBatchRater`
  would be a decorative config field that can never be applied, so neither
  class has one; `attest.vendors.registry._build_anthropic{,_batch}` accept
  and ignore the flag for factory-signature parity only.
- **Fireworks AI and Together AI: sync and `TogetherBatchRater` wired; `FireworksBatchRater` deliberately not.**
  `FireworksRater`/`TogetherRater` (sync, OpenAI-compatible Chat Completions)
  and `TogetherBatchRater` (OpenAI-shaped batch) now have the same
  `request_logprobs`/`top_logprobs` fields and `logprobs=True,
  top_logprobs=N` treatment as `OpenAIRater` -- support is high-confidence
  (OpenAI-compatible shape) but still empirically unverified; confirm with
  `tools/vendor_logprob_probe.py` before relying on it. `FireworksBatchRater`
  still has no such field: its Batch Inference API has a genuinely
  different, only partly-confirmed per-row schema (see that module's
  docstring) -- adding logprobs there means guessing at an
  already-unconfirmed output shape, so it stays deferred until that base
  schema itself is verified against a live batch run.
  `attest.vendors.registry._build_fireworks_batch` still accepts and
  ignores `request_logprobs`, the same signature-parity pattern Anthropic's
  factories use.

## Support matrix

Two separate questions, only one of which needs a live call:

- **Does `attest` request it?** Known now, from the code -- the two columns
  below reflect the current state of `attest.vendors.providers.*` directly,
  no API key required.
- **Does the vendor's API actually honor the request for a given model?**
  An empirical question, per model family (and not assumed to match between
  sync and batch even for the same vendor) -- unfilled below, since it
  needs live credentials and a real, billed call via
  `tools/vendor_logprob_probe.py sync`/`batch-submit`/`batch-fetch`. Add the
  model and outcome to a row once you've run it against a model you
  actually configure a `VendorSpec` with.

| Vendor    | Requests logprobs (sync) | Requests logprobs (batch) | Confirmed live (model, outcome) | Notes |
|-----------|---------------------------|----------------------------|----------------------------------|-------|
| openai    | Yes | Yes | not yet run | |
| mistral   | Yes | Yes | not yet run | |
| google    | Yes | Yes | not yet run | Gemini logprobs support is model-family-dependent |
| fireworks | Yes | No (`FireworksBatchRater` has no field) | not yet run (sync only) | batch row schema itself unconfirmed, see that module's docstring |
| together  | Yes | Yes | not yet run | |
| anthropic | No (no field) | No (no field) | N/A | Messages API has no logprobs equivalent |

## Open decision: no fallback value for Anthropic (deferred, not resolved here)

Once `request_logprobs=True` is used, three of four vendors log a real
per-vote logprob and Anthropic logs none -- `raw_response["logprobs"]` is
simply absent for Anthropic's votes. Any analytical use spanning the whole
ensemble (audit stratification, tau weighting, drift-sentinel enrichment)
will hit this asymmetry and must decide how to handle it: tolerate `None`/
absence for Anthropic's votes, or substitute some fallback value (e.g. an
imputed/synthetic logprob) so every vote carries a comparable confidence
figure.

**This decision is intentionally not made here.** Whatever the fallback
would be is a modeling choice with real consequences for anything built on
top of it (e.g. it would shape which records an audit-stratification scheme
prioritizes), and picking one silently -- rather than as a deliberate,
reviewed choice -- risks baking in an assumption nobody actually signed off
on. Before any of the downstream analytical uses above are built, this
section should be revisited and either: (a) record the decision to exclude
Anthropic votes from any logprob-weighted computation, or (b) record the
chosen fallback value/method and the reasoning behind it.
