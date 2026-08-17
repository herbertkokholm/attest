# attest

`attest` is a standalone, dependency-light Python library: the
screening-and-self-validation kernel for an LLM-based evidence-screening
method used in systematic-review title and abstract screening.

An ensemble of LLMs from different vendors rates each record on an ordinal
scale (exclude / uncertain / include); disagreement escalates to a human;
three statistically separated planes route human labels so that only one of
them is ever used to estimate recall; and every run produces a versioned
self-validation record reporting inter-rater agreement, error correlation,
escalation rate, and recall with its worst-case floor.

`attest` is the kernel only. A separate proprietary shell (e.g. ResearchWhat)
is expected to import it, fetch candidate records, schedule work, persist
results, and serve a UI. See [The boundary rule](#the-boundary-rule).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python tools/check_boundary.py
```

Try the CLI on the bundled example data (see [Running the CLI on the example
data](#running-the-cli-on-the-example-data) below), or use `attest` as a
library:

```python
from attest.contracts.input import validate_and_normalize
from attest.provenance.config import Config, VendorSpec
from attest.vendors.base import DeterministicRater, run_ensemble
from attest.ensemble.aggregate import g

normalized = validate_and_normalize(payload)  # a dict matching the input contract

config = Config(
    vendors={
        "v1": VendorSpec(
            model="deterministic-v1", model_version="1", prompt_version="p1", temperature=0.0
        ),
        "v2": VendorSpec(
            model="deterministic-v1", model_version="1", prompt_version="p1", temperature=0.0
        ),
    },
    aggregation="boundary_dispersion",
    tau=1.0,
)
raters = [DeterministicRater(vendor=name, seed=42) for name in config.vendors]
ensemble_run = run_ensemble(normalized.records, raters, config)
decisions = {
    vv.record_id: g(
        vv, aggregation=config.aggregation, tau=config.tau, zero_policy=config.zero_policy
    )
    for vv in ensemble_run.votes
}
```

By default every rater falls back to the kernel's own generic
`attest.vendors.base.SCREENING_TASK_PREAMBLE`, which carries no
review-specific eligibility criteria; `compose_system_prompt` appends the
kernel-owned output-format instruction (`OUTPUT_CONTRACT`) to whatever
criteria text is in force, so callers/config supply criteria only and never
need to restate the output contract themselves. To screen against a
review's actual published criteria, set `Config.default_prompt` (applied to
every record) and/or `Config.track_prompts` (a mapping of `record.track` to
criteria text, overriding `default_prompt` for that track) -- `run_ensemble`
and `submit_batch` resolve the right criteria per record automatically via
`Config.prompt_for_track`, so one ensemble configuration can screen several
reviews' records in a single run, each against its own criteria, while still
voting with the same vendors/models/aggregation on all of them. Both fields
are part of the hashed configuration content, so changing either changes
`ensemble_config_id`.

`g`'s boundary+dispersion rule never auto-commits the uncertain ordinal (0)
as a final label -- a vote vector whose mean is exactly 0 either escalates
to a human (`Config.zero_policy = "escalate"`, the default) or is folded
into `+1` (`zero_policy = "include"`). There is deliberately no "exclude"
option: auto-excluding on ensemble uncertainty is the one disposition that
would silently destroy recall. `zero_policy` is part of the hashed
configuration content, like `default_prompt`/`track_prompts` above.

`tau` is a threshold on a quantity (sample standard deviation over the
ordinal domain `{-1, 0, +1}`) that only attains a finite, enumerable set of
values for a given ensemble size `x` -- so most `tau` values are
behaviorally identical to a canonical one, and `tau` is not comparable
across different `x`. `attest.ensemble.tau` proves this and makes it
self-documenting: `describe_tau`/`validate_tau` produce a `TauReport`
(reachable dispersion values, the canonical interval `tau` falls in,
warnings for a suspicious `tau`), and `resolve_tau` computes a safe `tau`
from a declared policy instead of a hand-picked decimal. `attest screen`
validates `config.tau` at epoch-open time and persists the report to
`tau_report.json` in the run directory; `attest validate` surfaces it
alongside the validation record.

`tau`/dispersion is a *between*-vendor signal: how much the ensemble
disagreed. `attest.ensemble.confidence` adds an orthogonal *within*-vendor
one: per-vote logprobs, requested via `screen --request-logprobs`
(`Config`/`VendorSpec` are untouched by this -- it changes only what
side-channel metadata comes back with an unchanged sample, never
`ensemble_config_id`). A record's confidence is the median `P(the token a
vendor emitted)` across only the vendors that returned one, computed only
once at least `attest.ensemble.confidence.MIN_SUPPORTING_VOTES` (3) vendors
did -- below that there is no central-tendency statistic robust to a single
miscalibrated vendor, so the record is tagged `"unscored"` rather than
scored on thin evidence. Not every vendor can contribute: Anthropic's
Messages API has no logprobs equivalent at all (see
[`docs/logprob_support.md`](docs/logprob_support.md) for the full support
matrix), so an ensemble that includes Anthropic needs at least four vendors
total before three can ever support the confidence signal -- `attest`'s own
bundled `data/example_config.json` is deliberately four vendors
(anthropic, openai, mistral, together) for exactly this reason, not two.
This confidence signal feeds two of the three planes:
`audit-draw --stratify-by-confidence` stratifies the recall audit by tier
instead of track (see [Confidence-stratified
auditing](#confidence-stratified-auditing-logprobs) below), and
`attest.planes.active_learning.select_for_review` additionally selects a
record whenever it is scored and at or below `confidence_threshold`, which
catches unanimous-but-shaky vote vectors (e.g. `(1, 1, 1)` where every
vendor was individually unsure) that dispersion alone can never see, since
dispersion is zero for any unanimous vote regardless of how confident each
vendor actually was. `confidence_threshold` (default 0.5) is sourced from
`Config`/`config.json` exactly like `tau` -- a runbook setting the caller
shouldn't have to retype identically on every `audit-draw` invocation -- but
unlike `tau` it is deliberately excluded from `Config.to_dict()` and never
affects `ensemble_config_id`: it changes only how an already-fixed
excluded population is stratified for audit, never what a vendor samples or
the ensemble's own aggregate decision. `attest audit-draw` records whichever
threshold it actually used to `confidence_policy.json`, the same provenance
treatment `tau_report.json` gets, and `attest validate` reads it back to
reconstruct matching population sizes and surfaces it in the validation
payload as `confidence_policy`.

## The two stable contracts

`attest` exposes two versioned wire contracts. Treat both as frozen
interfaces: change them only via an explicit version bump, never in place.

- **`attest.contracts.input`** (`SCHEMA_VERSION = "1.0"`) — the shape of
  records submitted to attest for screening: id, title, abstract, track,
  external ids, and an optional gold label.
- **`attest.contracts.validation_record`** (`SCHEMA_VERSION = "1.1"`) — one
  self-validation record per stable ensemble-configuration epoch: the
  ensemble config (including `zero_policy`), inter-rater agreement, error
  correlation, escalation rate, recall (point estimate *and* rule-of-three
  worst-case floor, reported together, never the point estimate alone),
  confusion matrix, and PRISMA flow counts.

## The boundary rule

A product shell may import `attest`; `attest` must never import or know
anything about a product shell.

The kernel never imports or references collectors, schedulers, HTTP,
S3/object storage, databases, multi-tenancy, web frameworks, or UI. It may
call out to an LLM vendor API, but only through `attest.vendors` (see
`attest.vendors.providers`); every other module is network-free. Persistence
is confined to `attest.io.store`, a local JSON run directory — no database,
no object storage.

This is enforced both culturally and mechanically: `tools/check_boundary.py`
statically scans every module under `src/attest` and fails if a module
outside `attest.vendors` imports a network-client library (`socket`,
`urllib.request`, `requests`, `httpx`, ...), or if any kernel module —
including `attest.vendors` — imports a scheduler (`celery`, `apscheduler`,
...), a database/object-storage client (`sqlite3`, `boto3`, `sqlalchemy`,
...), or a web framework (`flask`, `fastapi`, `django`, ...). It runs in CI
(`.github/workflows/lint.yml`) and locally via:

```bash
python tools/check_boundary.py
```

`attest` is dependency-light: the kernel itself depends only on
`krippendorff` and `scipy`/`statsmodels` for statistics (`attest.stats`),
plus the Python standard library. Vendor SDKs (`anthropic`, `openai`,
`google-generativeai`, `mistralai`, `fireworks-ai`, `together`) are optional
extras, imported lazily and only inside `attest.vendors.providers`, so
installing `attest` without them still imports cleanly. Six vendor families
are supported out of the box — Anthropic, OpenAI, Google, Mistral, Fireworks
AI, and Together AI — each a genuinely distinct model family (Fireworks and
Together additionally give access to third-party open-weight models behind
an OpenAI-compatible API), which is the point: it strengthens the
inter-vendor independence argument behind the ensemble.

## Architecture at a glance

- **Prefilter** (`attest.prefilter`) — deterministic, source-agnostic dedup
  and exclusion rules, injected by the caller; owns the upstream PRISMA
  counts.
- **Ensemble** (`attest.ensemble`) — runs each configured vendor's `Rater`
  over a record and aggregates the vote vector into an auto-label or an
  escalation (`attest.ensemble.aggregate.g`); `attest.ensemble.tau` proves
  and exploits `tau`'s step-function structure to validate, describe, and
  derive it instead of leaving it a hand-picked decimal.
- **Three statistically separated planes** (`attest.planes`) — form a
  firewall so that recall is never estimated from a biased sample:
  - `adjudication` — resolves escalated (disagreement) records with an
    authoritative human label.
  - `active_learning` — routes high-disagreement records, or unanimous
    records with low within-vendor confidence (see
    `attest.ensemble.confidence`), to human review to improve the ensemble;
    never a probability sample.
  - `recall_audit` — the *only* plane recall may be estimated from: a random
    probability sample of the screen-excluded population, gold-checked by a
    human auditor.
- **Provenance** (`attest.provenance`) — three separately versioned
  descriptors and the machinery around them:
  - `config` — the screening ensemble configuration and its immutable,
    content-hashed `ensemble_config_id`.
  - `epochs` — stable spans over which that configuration (or, for a
    sentinel-drift trigger, the vendor behavior it hashes) is held fixed.
  - `changelog` — the append-only, persisted record of *why* an epoch
    changed: `initial_config`, `explicit_config_change` (with a
    machine-readable field diff), or `sentinel_drift`. See [Configuration
    governance](#configuration-governance-protocol-and-run-manifest) below.
  - `protocol` — the validation-protocol descriptor (audit design,
    adjudication protocol, sentinel thresholds, reporting spec), hashed into
    its own `protocol_id`, deliberately disjoint from `ensemble_config_id`.
  - `manifest` — a run manifest: software version, input hash, seeds, and a
    SHA-256 per persisted artifact, for offline integrity verification.
  - `sentinel` — the latent-vendor-drift detector (see [Latent-vendor-drift
    sentinel](#latent-vendor-drift-sentinel) below).
  - `runs` — per-run provenance records (what ran, on what track, under
    which epoch).
- **Stats** (`attest.stats`) — inter-rater agreement (Krippendorff's alpha),
  error correlation, and stratified recall with a rule-of-three floor. See
  [`docs/sentinel_drift_rule.md`](docs/sentinel_drift_rule.md) for the
  threshold-rule rationale behind the latent-vendor-drift sentinel
  (`attest.provenance.sentinel`, implemented -- see below), reusing this
  module's alpha; reproducible probe at
  [`tools/sentinel_drift_probe.py`](tools/sentinel_drift_probe.py).
- **I/O** (`attest.io.store`) — the only module that touches a filesystem: a
  local, idempotent JSON run directory.
- **Vendors** (`attest.vendors`) — the `Rater` protocol, a network-free
  `DeterministicRater` for tests, and live provider adapters
  (`attest.vendors.providers`) — the sole path to the network.
  `attest.vendors.batch` adds a parallel `BatchRater` protocol driving
  vendors' asynchronous Batch APIs (roughly 50% cheaper, without per-minute
  rate-limit fragility): `run_ensemble_batch` submits, polls with backoff,
  fetches, and assembles the exact same vote vectors `run_ensemble` would,
  stamped with the same `ensemble_config_id`, so everything downstream of
  `screen` is unaware of which execution strategy produced them.
  `request_logprobs` optionally requests per-token log probabilities on the
  ordinal decision from the vendors wired for it today (OpenAI, Mistral,
  Google, Fireworks, Together); `FireworksBatchRater` is the one exception
  -- its batch row schema is itself unconfirmed, so logprobs there are
  deliberately deferred -- and Anthropic's Messages API has no logprobs
  equivalent at all. See [`docs/logprob_support.md`](docs/logprob_support.md)
  for the support matrix and an open design decision on that gap. Verify
  actual vendor/model support with
  [`tools/vendor_logprob_probe.py`](tools/vendor_logprob_probe.py) rather
  than assuming it.
- **CLI** (`attest.cli`) — file-based subcommands over a run directory:
  `screen`, `batch-fetch`, `adjudicate`, `audit-draw`, `audit-apply`,
  `validate`, `ablate`, `protocol`, `manifest`, `verify`, `sentinel-init`,
  `sentinel-check`, `active-learning-select`, `active-learning-review`.

## Running the CLI on the example data

`data/example_gold_set.json` (input contract) and `data/example_config.json`
(a four-vendor ensemble config -- anthropic, openai, mistral, together) are
bundled for a quick, network-free run using `--deterministic-seed`, which
swaps in seeded `DeterministicRater`s instead of live vendor adapters:

```bash
attest screen \
  --input data/example_gold_set.json \
  --config data/example_config.json \
  --run-dir /tmp/attest-demo \
  --deterministic-seed 42
```

This prints a summary (PRISMA counts, escalation count) and persists votes,
decisions, config, a `tau_report.json` self-documenting what `config.tau`
does at this ensemble size (see `attest.ensemble.tau` above), and a run
record under `--run-dir`. List and resolve any escalated records:

```bash
attest adjudicate --run-dir /tmp/attest-demo
attest adjudicate --run-dir /tmp/attest-demo --record-id <id> --label <-1|0|1>
```

Draw and apply a random recall-audit sample from the screen-excluded
population:

```bash
attest audit-draw --run-dir /tmp/attest-demo \
  --input data/example_gold_set.json --size 2 --seed 7

echo '{"<drawn-id-1>": 1, "<drawn-id-2>": -1}' > /tmp/labels.json
attest audit-apply --run-dir /tmp/attest-demo --labels /tmp/labels.json
```

`--size all` draws the entire screen-excluded population instead of a
sample, for exact (not floored) recall when human-labeling the draw is
free -- e.g. scoring against an already-published gold set.

Assemble the self-validation record for the epoch:

```bash
attest validate --run-dir /tmp/attest-demo --input data/example_gold_set.json
```

Or run a controlled ablation sweep over ensemble size on stored votes
against a frozen gold set:

```bash
attest ablate --run-dir /tmp/attest-demo --input data/example_gold_set.json
```

`--tau` and `--zero-policy` are held fixed across every swept ensemble size
`x'` in this command (they describe the sweep call, not something the sweep
varies) -- each result's `tau_report` documents what the fixed `tau`
actually means at that result's own `x`.

Every subcommand except `screen` and `batch-fetch` runs entirely offline
over files already written to the run directory; `screen` and `batch-fetch`
are the only ones that may reach the network, and only through a `Rater` or
`BatchRater` built by `attest.vendors` (bypassed here via
`--deterministic-seed`).

### Batch mode

For large jobs, `screen --mode batch` submits one vendor batch per rater
instead of rating records synchronously, at each vendor's cheaper batch
rate. `--wait` polls every batch to completion before persisting votes, just
like `--mode sync` does synchronously; without it, `screen` exits right
after submission and a later `batch-fetch` call resumes from the persisted
batch handles:

```bash
attest screen \
  --input data/example_gold_set.json \
  --config data/example_config.json \
  --run-dir /tmp/attest-batch-demo \
  --deterministic-seed 42 \
  --mode batch

# ... later, possibly a different process invocation ...

attest batch-fetch \
  --run-dir /tmp/attest-batch-demo \
  --input data/example_gold_set.json \
  --deterministic-seed 42
```

This produces byte-for-byte the same `votes.json` and `decisions.json` as
the synchronous path over the same input, config, and seed.

### Confidence-stratified auditing (logprobs)

`--request-logprobs` asks every vendor that supports it for per-vote
logprobs -- openai, mistral, together here; Anthropic never does (see
[`docs/logprob_support.md`](docs/logprob_support.md)) -- and
`audit-draw --stratify-by-confidence` stratifies the recall audit by the
resulting tier instead of track:

```bash
attest screen \
  --input data/example_gold_set.json \
  --config data/example_config.json \
  --run-dir /tmp/attest-logprobs-demo \
  --deterministic-seed 37 \
  --request-logprobs

attest audit-draw --run-dir /tmp/attest-logprobs-demo \
  --input data/example_gold_set.json --size all --stratify-by-confidence
```

For this seed nothing escalates, and the screen-excluded population is
exactly two records landing in different tiers: `rec-001` (median
confidence 0.34, scored from openai/mistral/together -- Anthropic's vote
still counted toward the `boundary_dispersion` decision above, just not
toward this figure) draws `"low"`; `rec-002` (median confidence 0.58) draws
`"high"`. The threshold that produced that split
(`Config.confidence_threshold`, 0.5 by default) is recorded to
`confidence_policy.json`:

```json
{"low_threshold": 0.5, "min_supporting_votes": 3}
```

Apply gold-check labels and validate exactly as before:

```bash
echo '{"rec-001": 1, "rec-002": -1}' > /tmp/labels.json
attest audit-apply --run-dir /tmp/attest-logprobs-demo --labels /tmp/labels.json
attest validate --run-dir /tmp/attest-logprobs-demo --input data/example_gold_set.json
```

`rec-001`'s gold label is `1` -- a truly relevant record screening excluded
anyway -- while `rec-002`'s is `-1`, correctly excluded; the low-confidence
stratum (`rec-001` alone, stratum population 1) is exactly the one carrying
the miss, the concrete illustration of what stratifying toward
low-confidence records is for. `attest validate`'s output now also carries
a `confidence_policy` key alongside `tau_report`, the same provenance
treatment `tau` already gets.

## Configuration governance, protocol, and run manifest

Three provenance levels are kept deliberately separate (see
`attest.provenance.protocol`'s module docstring):

- **Screening config** (`attest.provenance.config.Config`) -- vendors,
  models, prompts, aggregation, tau, zero policy. The only thing
  `ensemble_config_id` hashes.
- **Validation protocol** (`attest.provenance.protocol.ValidationProtocol`)
  -- audit-strata/design, the adjudication protocol, the sentinel-drift
  rule's thresholds, and the reporting/analysis spec. Hashed into its own
  `protocol_id`, never into `ensemble_config_id`: a change to the audit
  budget or the sentinel's advisory threshold is an analysis-plan revision,
  not a new measurement instrument.
- **Run manifest** (`attest.provenance.manifest.RunManifest`) -- one
  execution's software version, the hash of the input file it screened,
  every random seed used, and a SHA-256 per artifact the run directory holds
  at manifest-build time.

`attest screen`'s first invocation over a run directory logs a changelog
event (`changelog.json`, append-only -- `attest.io.store.RunStore.write_changelog`
refuses any write that would rewrite or shrink already-stored history): an
`initial_config` event by default, or an `explicit_config_change` event --
with a machine-readable, dot-flattened field diff
(`attest.provenance.changelog.diff_config_fields`) -- when
`--previous-run-dir`/`--change-reason` are given.

A run directory is locked to one configuration/epoch forever
(`RunStore.write_config`/`write_epoch` both refuse a second, different
value), so a deliberate config change -- a bumped model version, a
different prompt, a raised `tau` -- always means a *new* run directory, the
"fresh `RUN_DIR` per epoch" convention the runbook already follows (e.g.
`RUN_DIR=data/run_epoch2`). `--previous-run-dir` is how that new run
directory's first `screen` call records *why* it exists instead of looking
like an unrelated fresh start: it names the run directory of the epoch
being replaced, so attest reads that predecessor's own persisted
`config.json` back (not a hand-passed file, which could go stale) to
compute the `before` id and the field diff.

`/tmp/attest-demo` (screened in [Running the CLI on the example
data](#running-the-cli-on-the-example-data) above with
`data/example_config.json`) is epoch 1. Later, a deliberate change --
`openai`'s `model_version` bumped after a vendor snapshot change -- always
means a *new* run directory, e.g. `RUN_DIR=data/run_epoch2` in the runbook's
own naming, or here:

```bash
# data/example_config_epoch2.json is identical to data/example_config.json
# except for that one field -- a static, hand-authored example alongside
# example_config.json and example_gold_set.json, not something attest
# derives at run time.
attest screen \
  --input data/example_gold_set.json \
  --config data/example_config_epoch2.json \
  --run-dir /tmp/attest-epoch2 \
  --deterministic-seed 42 \
  --previous-run-dir /tmp/attest-demo \
  --change-reason "bumped openai model_version after a vendor snapshot change" \
  --approver reviewer-a
```

`/tmp/attest-epoch2/changelog.json` then records one `explicit_config_change`
event: `before` is `/tmp/attest-demo`'s actual `ensemble_config_id`, `after`
is `/tmp/attest-epoch2`'s, `changed_fields` is
`["vendors.openai.model_version"]`, and `approver` is `"reviewer-a"` --
`--approver` requires `--previous-run-dir`, since there is nothing to
approve on an ordinary first run.

Build and persist the protocol descriptor and the run manifest once a run
directory's artifacts are in the state you want to freeze (typically after
`validate`), then verify integrity any time afterward, entirely offline:

```bash
attest protocol --run-dir /tmp/attest-demo --stratify-by track --audit-size-policy "n=600"
attest manifest --run-dir /tmp/attest-demo --input data/example_gold_set.json --seed screen=42
attest verify --run-dir /tmp/attest-demo
```

`verify` recomputes the SHA-256 of every artifact the manifest recorded --
config, protocol descriptor, votes, raw responses, decisions, epoch,
changelog, the audit-draw snapshot, the audit-labels snapshot, and the
validation-record snapshot (`RunStore.MANIFEST_ARTIFACTS`) -- and reports
exactly which one is missing or has changed; it exits `1` (without printing
a Python traceback) when anything fails, so a CI step can gate on it
directly. The input file itself is hashed, not copied into the run
directory, since it can be large (e.g. a multi-review SYNERGY gold set).

## Latent-vendor-drift sentinel

Implements the hybrid rule recommended in
[`docs/sentinel_drift_rule.md`](docs/sentinel_drift_rule.md):
`attest.provenance.sentinel` runs the same raters (or `DeterministicRater`s)
that back `screen`, offline and network-free when seeded. A hard trigger
(>= 2 one-directional polarity crossings per vendor, `baseline in {0, +1} ->
current == -1`, on a frozen sentinel set) opens a new epoch and appends a
`sentinel_drift` changelog event; a looser Krippendorff's-alpha bound is
logged as an advisory-only signal that never opens an epoch on its own.

```bash
attest sentinel-init --run-dir /tmp/attest-demo \
  --sentinel-input data/example_sentinel_set.json --deterministic-seed 42

# ... periodically, e.g. from the runbook's own schedule ...
attest sentinel-check --run-dir /tmp/attest-demo \
  --sentinel-input data/example_sentinel_set.json --deterministic-seed 42
```

On a hard trigger, `sentinel-check` appends the `sentinel_drift` event to
this run directory's changelog and prints the newly opened epoch; like an
explicit config change, this run directory keeps its original epoch's
artifacts, and continued screening under the new epoch belongs in a fresh
run directory (the same convention `--previous-run-dir` follows for an
explicit change). Sentinel *scheduling* (cadence, baseline persistence
across runbook invocations) is deliberately left to the runbook -- see the
doc's "Split with the runbook" section.

## Adjudication and active-learning provenance

`adjudicate --reviewer ID --protocol-id ID` and `active-learning-review
--reviewer ID --protocol-id ID --notes TEXT` attach reviewer/selection
provenance -- who resolved or reviewed a record, when, under which
protocol, and why it was routed there in the first place
(`selection_reason`: `boundary`, `dispersion`, `tie`, or, for active
learning, `low_confidence`) -- to `adjudication_records.json` and
`active_learning_reviews.json`. This is provenance only: attest never
auto-tunes a prompt or retrains anything from either plane. A reviewer's
notes may motivate a later, human-initiated `attest screen --previous-run-dir
... --approver ...` call; recording the review and making that change are
always two separate, explicit steps.

`active-learning-select` persists `select_for_review`'s output
(`active_learning_selections.json`) so a later `active-learning-review` call
can look a record's selection reason back up without recomputing it.

## Implemented, planned, and out of scope

**Implemented:**

- The three statistically separated planes, the boundary+dispersion
  aggregation rule, tau/confidence provenance, versioned config/protocol/
  manifest, the append-only changelog, the latent-vendor-drift sentinel
  (hard trigger + advisory alpha), offline manifest-based artifact
  integrity verification, and reviewer/selection provenance on adjudication
  and active-learning records.
- Screening recall (sensitivity), estimated *only* from the random
  `recall_audit` plane's probability sample of the screen-excluded
  population -- this logic is unchanged by any of the above; adjudication
  and active-learning selections remain statistically firewalled out of the
  recall estimator (`attest.planes.recall_audit.build_strata` refuses any
  row not tagged `PLANE_RECALL_AUDIT`).

**Aggregation rules:** `attest.ensemble.aggregate.AGGREGATION_BOUNDARY_DISPERSION`
is the only aggregation rule with an implementation. `AGGREGATION_MAJORITY`
and `AGGREGATION_UNANIMITY` are recognized names (`Config.aggregation`
accepts them without raising) but `attest.ensemble.aggregate.g` raises
`NotImplementedError` if either is actually selected. Do not configure a
screening run with them expecting a working aggregation rule.

**Planned (not implemented):**

- **Inclusion audit.** `attest.contracts.validation_record.Recall`'s `TPc`
  (true positives) is currently taken directly from the confusion matrix
  against gold labels, which requires every included/escalated record to
  already carry ground truth -- true for the planned SYNERGY-based empirical
  evaluation, but not for a live deployment without full gold coverage. The
  manuscript's inclusion-audit design (§2.9: a separate random/track-stratified
  sample of the include-and-escalate set, its own confidence interval,
  mirroring the exclusion audit) is a future live-deployment module, not
  implemented here.
- Automated prompt tuning or retraining from active-learning or adjudication
  output. Both planes are provenance-recording only (see above); a config
  change motivated by a review is always a separate, explicit,
  human-initiated `attest screen --previous-run-dir ...` call.

**Out of scope for this kernel:**

- **Search recall / search completeness** (capture-recapture, relative
  recall). Screening recall answers a different question than search
  completeness and is reported separately, never multiplied together (see
  the manuscript's Discussion); attest implements neither capture-recapture
  nor relative recall.
- Sentinel *scheduling* (cadence, cross-run baseline persistence) --
  `attest.provenance.sentinel` is a pure, offline detector; wiring it into a
  recurring job belongs to the runbook (see [`docs/sentinel_drift_rule.md`](docs/sentinel_drift_rule.md)'s
  "Split with the runbook" section).
- Everything the boundary rule already excludes: scheduling, collectors,
  databases/object storage, and a UI. See [The boundary
  rule](#the-boundary-rule).

## Running the tests

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python tools/check_boundary.py
```

## License

attest is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See LICENSE.

Third-party dependency licenses: see [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).

Copyright (C) 2026 Thomas Herbert Kokholm.

A commercial license is available on request for use that the AGPL's terms do
not permit (for example, building a proprietary product or network service on
top of attest without releasing the corresponding source). See
CONTRIBUTING.md for the contribution licensing terms.
