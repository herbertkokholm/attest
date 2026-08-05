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
- **Fireworks AI and Together AI are stubbed, not yet wired.** Both vendors
  were added to `attest.vendors` after this document was first written.
  `attest.vendors.registry._build_fireworks{,_batch}`/`_build_together{,_batch}`
  accept and ignore `request_logprobs` today, the same signature-parity
  pattern as Anthropic's factories -- but unlike Anthropic, this is a
  temporary stub, not a permanent API gap: `FireworksRater`/`TogetherRater`
  (sync) and `TogetherBatchRater` call OpenAI-compatible endpoints and are
  plausible near-term candidates for the same `logprobs=True,
  top_logprobs=N` treatment `attest.vendors.providers.openai.OpenAIRater`
  already has. `FireworksBatchRater` is a separate case: its Batch
  Inference API has a genuinely different, only partly-confirmed per-row
  schema (see that module's docstring) -- adding logprobs there means
  guessing at an already-unconfirmed output shape, so it should wait until
  that schema itself is verified against a live batch run, not be
  implemented speculatively alongside the other three.

## Support matrix

Fill in by running `tools/vendor_logprob_probe.py` against each model you
actually configure a `VendorSpec` with -- request-parameter presence in an
SDK does not guarantee the API honors it for a given model family, and
sync/batch support are never assumed to match.

| Vendor     | Model | Sync | Batch | Notes |
|------------|-------|------|-------|-------|
| openai     |       |      |       |       |
| mistral    |       |      |       |       |
| google     |       |      |       |       |
| fireworks  | --    | Not wired | Not wired | `request_logprobs` accepted but ignored (stub); sync is a plausible near-term add, batch should wait for the row schema itself to be confirmed |
| together   | --    | Not wired | Not wired | `request_logprobs` accepted but ignored (stub); both sync and batch are plausible near-term adds (OpenAI-compatible shape) |
| anthropic  | (any) | No   | No    | Messages API has no logprobs equivalent |

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
