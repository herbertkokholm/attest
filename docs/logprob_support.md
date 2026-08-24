# Per-vote logprobs: support matrix and the open fallback-value decision

Each ensemble member rates a record with a single ordinal token in
`{-1, 0, +1}` (`attest.vendors.base.OUTPUT_CONTRACT`). A per-vote logprob on
that token is a *within*-vendor confidence signal -- how sure this one
vendor was of its own answer -- complementary to the ensemble's existing
*between*-vendor dispersion measure (`tau`, in `attest.ensemble.tau`).
`attest.ensemble.confidence` implements this signal (extraction, the
coverage-gated median, and tiering; see "Decision" below) and it is wired
into two of the four planes (`attest.planes`), each reusing the exact same
`record_confidence`/`confidence_tier` figure, never a signal recomputed
per-plane:

- **Recall audit** (`attest.planes.recall_audit`): `draw_audit_sample`'s
  `stratify_by_confidence` stratifies the audit draw by confidence tier
  instead of track.
- **Active learning** (`attest.planes.active_learning`): `select_for_review`
  additionally selects a vote vector whenever its confidence is scored and
  at or below `confidence_threshold` -- independently of dispersion/
  boundary, so it catches the class of unanimous-but-shaky vote vectors
  (e.g. `(1, 1, 1)` with every vendor's own confidence weak) that the
  dispersion/boundary signal alone cannot see, since dispersion is zero for
  any unanimous vote regardless of how confident each vendor actually was.
- **Adjudication** (`attest.planes.adjudication`): deliberately not wired
  in. Its queue already contains only already-escalated records; confidence
  could only reorder that queue's triage priority, not change which records
  appear in it, so this was judged not worth building without a concrete
  need (see the "left open" Neyman-allocation note below for the same
  judgment applied elsewhere).

The latent-vendor-drift sentinel (`docs/sentinel_drift_rule.md`) enrichment
remains a distinct, not-yet-addressed candidate use -- see the "Decision"
section below for why it is explicitly out of scope here.

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

## Decision: coverage-aware confidence for audit-draw stratification

Resolved for the audit-stratification candidate use above
(`attest.planes.recall_audit.draw_audit_sample`/`build_strata`). The
drift-sentinel enrichment (`docs/sentinel_drift_rule.md`) is a separate
downstream use and is not addressed by this decision -- revisit it
separately before building that.

Once `request_logprobs=True` is used, some vendors log a real per-vote
logprob and others (Anthropic always; Fireworks batch, see above) log none
-- `raw_response["logprobs"]` is simply absent for those votes. Any
analytical use spanning the whole ensemble must decide how to handle this
asymmetry.

**Per-vendor exclusion, not imputation.** A vendor that logs no logprob for
a vote contributes no confidence figure for that vote. Its ordinal vote
still counts fully toward `tau`/ensemble aggregation
(`attest.ensemble.tau`, `attest.ensemble.aggregate`) -- only the *derived
confidence signal* excludes it. No fallback/imputed logprob value is
introduced; picking one silently was the risk this section originally
flagged, and exclusion avoids baking in an unreviewed assumption about what
a missing vendor's confidence "would have been."

**Minimum coverage: 3 supporting votes.** A record's confidence score is
computed only when at least 3 of its ensemble members support logprobs.
Below that, the record is marked unscored for confidence purposes and
falls back to whatever non-confidence stratification key is in use (e.g.
track-only, or the pooled stratum) -- it is never silently dropped from
the audit population.

Why 3, not 2: at N=1 the score is one vendor's raw, unaveraged noise. At
N=2, averaging shrinks standard error somewhat, but no statistic distinct
from the plain mean exists to identify which of the two points is the
outlier if they disagree -- a single miscalibrated vendor sways the mean
exactly as much as a well-calibrated one would. N=3 is the smallest N at
which a robust central tendency (median / majority) first diverges from
the plain mean, giving the confidence figure resistance to exactly one
badly-calibrated vendor -- the same "which one's the odd one out"
reasoning `tau`'s dispersion rule already applies to the votes themselves,
applied here to the derived confidence figure instead.

**Computation** (`attest.ensemble.confidence`; wired into
`attest.planes.recall_audit.draw_audit_sample`/`ExcludedRecord` via the new
`stratify_by_confidence` flag and `confidence_tier` field):

1. Per vote, from `raw_response["logprobs"]` (vendor-native shape, joined
   against that vendor's entry in the record's `VoteVector` --
   `attest.ensemble.votes.Vote` itself carries no logprob field, so this is
   a read against `raw_responses.json`, not a new field on `Vote`), extract
   the logprob of the *actually emitted* ordinal token and convert to a
   probability via `exp(logprob)`. This is comparable across vendors
   because it is always "P(the token this vendor emitted)," regardless of
   each vendor's native logprobs shape.
2. Per record, take the median of that probability across only the
   supporting vendors' votes (this is exactly why the >=3 coverage rule
   above matters -- below it, "median" is a 1- or 2-point degenerate
   statistic, not a robust one).
3. Bucket records into confidence tiers (e.g. low/high, or finer bins) by a
   threshold on that median. This is a new stratification key, structurally
   identical to `track` today -- `attest.stats.recall.Stratum`'s docstring
   already anticipates it ("e.g. a track or dispersion band").
4. Feed the tier into `draw_audit_sample`'s existing per-stratum
   proportional allocation and `build_strata`'s per-stratum n/m/population
   accounting exactly as `track` is fed in today.
   `attest.stats.recall.stratified_recall` needs no change: it is already
   agnostic to what a stratum name represents, only consuming n/m/population
   per stratum.
5. Records below the 3-vote coverage threshold get their own explicit
   stratum (e.g. `"unscored"`), never folded into a tier they have no
   evidence for and never dropped from the audit population.

**Wired end-to-end through `attest.cli`/`attest.io.store`, not library-only.**
An earlier version of this section stopped at the library API; that left the
whole pipeline unreachable in practice, since `screen` never actually
requested logprobs and `audit-draw`/`validate` never actually computed a
tier. Now:

- `screen --request-logprobs` (and `batch-fetch --request-logprobs`, which
  must be passed identically to whatever `screen --mode batch` submitted
  with -- not itself persisted in `BatchHandle`, the same way
  `--deterministic-seed` isn't) thread `request_logprobs` into
  `attest.vendors.registry.build_raters`/`build_batch_raters`, so
  `raw_responses.json` actually gains a `"logprobs"` entry per vote.
  `DeterministicRater`/`DeterministicBatchRater` grew a matching
  `request_logprobs` field emitting a deterministic, seeded fake logprob in
  the same OpenAI-compatible shape (never a simulation of any real vendor's
  behavior) purely so this whole path is testable offline.
- `audit-draw --stratify-by-confidence` computes each screen-excluded
  record's tier from the run's already-stored `votes.json`/
  `raw_responses.json` (`attest.cli._attach_confidence_tiers`, reusing
  `record_confidence`/`confidence_tier` unchanged) before drawing, and
  persists the `low_threshold` actually used to `confidence_policy.json`
  via `RunStore.write_confidence_policy` -- the "gemmes som bevis" record
  this section's earlier draft only described in a docstring without ever
  writing to disk.
- The `low_threshold` `--stratify-by-confidence` uses comes from
  `Config.confidence_threshold`, a field on `attest.provenance.config.Config`
  sourced from the same `--config` JSON file `tau`/`vendors` come from -- a
  runbook setting, not something retyped identically on every invocation.
  There is deliberately no `--confidence-threshold` CLI flag on `audit-draw`
  at all, mirroring `tau`, which has no CLI override either: one source of
  truth, no drift between what `config.json` declares and what a given draw
  actually used (an earlier draft of this wiring added such a flag as a
  convenience for quick experimentation; it was removed once it became
  clear the asymmetry with `tau` was more confusing than the convenience was
  worth). Unlike `tau`, `confidence_threshold` is deliberately excluded from
  `Config.to_dict()` (see that method's docstring) and therefore never
  affects `ensemble_config_id`; `attest.io.store._config_to_dict` persists
  it into the run's stored `config.json` alongside, but outside, the hashed
  payload, so it still round-trips through `RunStore` like every other
  `Config` field.
- `validate` reads `confidence_policy.json` (its mere presence signals this
  run's audit draw was confidence-stratified), recomputes the same tiers to
  rebuild matching population sizes for `attest.stats.recall`, and surfaces
  the policy in its output payload as `confidence_policy` -- the same
  provenance treatment `tau_report` already gets.

**Left open, not resolved here:** whether allocation across confidence
tiers should stay population-proportional (as `_allocate_proportionally`
does today) or move to disproportionate/Neyman-style oversampling of the
low-confidence tier -- which is what "stratifying audit-draw toward
low-confidence records" (this doc's own candidate use, above) actually
implies. Proportional allocation stratifies but does not prioritize; that
choice should get the same deliberate treatment this section just did
before it's built.
