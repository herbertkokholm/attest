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
        "v1": VendorSpec(model="deterministic-v1", model_version="1", prompt_version="p1"),
        "v2": VendorSpec(model="deterministic-v1", model_version="1", prompt_version="p1"),
    },
    aggregation="boundary_dispersion",
    tau=1.0,
)
raters = [DeterministicRater(vendor=name, seed=42) for name in config.vendors]
ensemble_run = run_ensemble(normalized.records, raters, config)
decisions = {
    vv.record_id: g(vv, aggregation=config.aggregation, tau=config.tau) for vv in ensemble_run.votes
}
```

## The two stable contracts

`attest` exposes two versioned wire contracts. Treat both as frozen
interfaces: change them only via an explicit version bump, never in place.

- **`attest.contracts.input`** (`SCHEMA_VERSION = "1.0"`) — the shape of
  records submitted to attest for screening: id, title, abstract, track,
  external ids, and an optional gold label.
- **`attest.contracts.validation_record`** (`SCHEMA_VERSION = "1.0"`) — one
  self-validation record per stable ensemble-configuration epoch: the
  ensemble config, inter-rater agreement, error correlation, escalation
  rate, recall (point estimate *and* rule-of-three worst-case floor,
  reported together, never the point estimate alone), confusion matrix, and
  PRISMA flow counts.

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
`google-generativeai`, `mistralai`) are optional extras, imported lazily and
only inside `attest.vendors.providers`, so installing `attest` without them
still imports cleanly. Four vendor families are supported out of the box —
Anthropic, OpenAI, Google, and Mistral — each a genuinely distinct model
family, which is the point: it strengthens the inter-vendor independence
argument behind the ensemble.

## Architecture at a glance

- **Prefilter** (`attest.prefilter`) — deterministic, source-agnostic dedup
  and exclusion rules, injected by the caller; owns the upstream PRISMA
  counts.
- **Ensemble** (`attest.ensemble`) — runs each configured vendor's `Rater`
  over a record and aggregates the vote vector into an auto-label or an
  escalation (`attest.ensemble.aggregate.g`).
- **Three statistically separated planes** (`attest.planes`) — form a
  firewall so that recall is never estimated from a biased sample:
  - `adjudication` — resolves escalated (disagreement) records with an
    authoritative human label.
  - `active_learning` — routes high-disagreement records to human review to
    improve the ensemble; never a probability sample.
  - `recall_audit` — the *only* plane recall may be estimated from: a random
    probability sample of the screen-excluded population, gold-checked by a
    human auditor.
- **Provenance** (`attest.provenance`) — immutable, content-hashed
  `ensemble_config_id`, stable epochs, and run records.
- **Stats** (`attest.stats`) — inter-rater agreement (Krippendorff's alpha),
  error correlation, and stratified recall with a rule-of-three floor.
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
- **CLI** (`attest.cli`) — `screen`, `batch-fetch`, `adjudicate`,
  `audit-draw`, `audit-apply`, `validate`, `ablate`: file-based subcommands
  over a run directory.

## Running the CLI on the example data

`data/example_gold_set.json` (input contract) and `data/example_config.json`
(a two-vendor ensemble config) are bundled for a quick, network-free run
using `--deterministic-seed`, which swaps in seeded `DeterministicRater`s
instead of live vendor adapters:

```bash
attest screen \
  --input data/example_gold_set.json \
  --config data/example_config.json \
  --run-dir /tmp/attest-demo \
  --deterministic-seed 42
```

This prints a summary (PRISMA counts, escalation count) and persists votes,
decisions, config, and a run record under `--run-dir`. List and resolve any
escalated records:

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

Assemble the self-validation record for the epoch:

```bash
attest validate --run-dir /tmp/attest-demo --input data/example_gold_set.json
```

Or run a controlled ablation sweep over ensemble size on stored votes
against a frozen gold set:

```bash
attest ablate --run-dir /tmp/attest-demo --input data/example_gold_set.json
```

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

Copyright (C) 2026 Thomas Herbert Kokholm.

A commercial license is available on request for use that the AGPL's terms do
not permit (for example, building a proprietary product or network service on
top of attest without releasing the corresponding source). See
CONTRIBUTING.md for the contribution licensing terms.
