# attest

`attest` is a standalone, dependency-light Python library: the
screening-and-self-validation kernel for an LLM-based evidence-screening
method used in systematic-review title and abstract screening.

## The boundary rule

A product shell may import `attest`; `attest` must never import or know
anything about a product shell.

The kernel never imports or references collectors, schedulers, HTTP,
S3/object storage, databases, multi-tenancy, web frameworks, or UI. If a
concern belongs to "the running product," it does not belong here. This is
enforced culturally as of this commit; a later commit may add a CI
import-linter to enforce it mechanically.

`attest` is dependency-light: as of this commit it uses only the Python
standard library. Statistical dependencies (`krippendorff`, `scipy` /
`statsmodels`) are expected to arrive in a later commit, scoped narrowly to
the `attest.stats` package.

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

Everything else in the package — the prefilter framework, ensemble
aggregation, the statistically separated planes, provenance tracking, and
statistics — may be refactored freely and is not yet fully implemented (see
`src/attest/*` stub modules).

## Domain notes

- Each record is rated on an ordinal scale: exclude (−1), related/uncertain
  (0), include (+1).
- Screening uses an ensemble of LLMs from different vendors; ensemble size
  `x` is dynamic.
- Three statistically separated planes — adjudication, active learning, and
  random recall audit — form a firewall: only random audit samples estimate
  recall.
- Every decision is stamped with an immutable `ensemble_config_id` and
  reported per stable epoch.
- The deterministic prefilter runs before the ensemble and owns the
  upstream PRISMA counts; its framework lives in the kernel
  (`attest.prefilter.framework`), but concrete source-specific dedup and
  exclusion rules are injected by the caller.

## Running the tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
mypy src
pytest
```

## License

attest is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See LICENSE.

Copyright (C) 2026 Thomas Herbert Kokholm.

A commercial license is available on request for use that the AGPL's terms do
not permit (for example, building a proprietary product or network service on
top of attest without releasing the corresponding source).
