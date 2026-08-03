# Latent-vendor-drift sentinel: threshold rule

Characterizes three candidate rules for deciding when a vendor's rating
behavior has drifted from its own baseline on a frozen sentinel set, and
recommends one. This is the rationale-of-record for the rule; the manuscript
names it in one sentence and does not repeat this analysis.

Reproducible probe: `examples/sentinel_drift_probe.py`. Imports only
`attest.stats.agreement` and the standard library -- no scheduler, no
storage, no network, matching the kernel's own boundary
(`tools/check_boundary.py`) even though this path is not itself scanned by
it. Run it directly to regenerate every number below.

## Semantics

Per vendor, per sentinel record: `baseline` is that vendor's first-epoch
rating; `current` is its latest rating on the same frozen record, ordinal
domain `{-1, 0, +1}`. Evaluated independently per vendor -- one vendor
drifting must be able to trigger on its own; drift is never averaged across
a stable ensemble. No statistic below is reimplemented: alpha is
`attest.stats.agreement.krippendorff_alpha` over the two-row
`{baseline, current}` reliability matrix, exactly as it is used anywhere
else in the kernel.

**Assumption, stated plainly:** the benign-flip-rate figures below assume
temperature pinned identically at baseline- and current-time, per vendor.
`VendorSpec.temperature` (`attest.provenance.config`) is a hashed field --
changing it already yields a different `ensemble_config_id` and opens a new
epoch through the ordinary config-change path, the same as a model or
prompt-version change. The sentinel comparison inherits that guarantee; it
does not separately enforce it. Temperature 0 is not assumed bit-identical
across calls -- the benign flip rates `p` below are exactly the
nondeterminism budget this analysis is pricing in.

Not to be confused with `attest.vendors.base.ModelVersionDriftError`: an
immediate, per-call check that a vendor's self-reported model version
matches the configured one, catching a silent alias/snapshot swap
mid-epoch. That is a metadata check on the model identity; the sentinel
here is a periodic, statistical check on a vendor's rating *behavior* --
orthogonal, and does not subsume or get subsumed by it.

## The three rules

1. **Absolute** (zero tolerance): `any(current[i] != baseline[i])`.
2. **Alpha threshold**: `krippendorff_alpha([baseline, current]) < tau`, candidate `tau = 0.95`.
3. **Critical-polarity crossing**: fires when a record crosses `baseline in {0, +1}` to
   `current == -1` -- a record the vendor used to keep, now excluded. The reverse crossing
   (`-1 -> {0, +1}`) does not count (see below for why).

Sentinel: n=100 per vendor, composition `{-1: 90, 0: 3, 1: 7}` -- a stated,
literature-typical stand-in for a heavily exclude-dominated screening track,
not read from any one review's data.

## Benign false-epoch rate (no real drift)

Per-record flip probability `p` models temperature-0 nondeterminism; ratings
flip independently and uniformly to a different domain value.

| p | absolute | alpha < 0.95 | polarity, single crossing | polarity, ≥2 crossings | polarity, both directions |
|---|---|---|---|---|---|
| 0.5% | 0.394 (`1-(1-p)^100`) | 0.248 | 0.027 | 0.0003 | 0.390 |
| 1.0% | 0.632 | 0.446 | 0.048 | 0.0006 | 0.617 |
| 2.0% | 0.871 | 0.720 | 0.097 | 0.006 | 0.855 |

Absolute is unusable at any plausible `p`: `1-(1-p)^100` alone rules it out.
Alpha at `tau=0.95` is not much better at this `n`. Both-directions polarity
collapses toward the absolute rule's behavior, because 90% of the sentinel
is baseline-`-1`, so most benign flips originate there and cross regardless
of direction -- counting the reverse crossing buys back exactly the
instability rule 3 exists to avoid. One-directional polarity is the only
rule with a workable false-epoch rate on its own; requiring **≥2** crossings
instead of 1 drives it to noise level, at negligible cost to true-drift
sensitivity since a real vendor regression is a correlated failure across
records, not an isolated one.

## Direction sensitivity

`k` injected flips, `+1 -> -1` (recall-critical) vs `0 -> +1` (benign swing):

| k | `+1→-1` alpha | absolute / polarity | `0→+1` alpha | absolute / polarity |
|---|---|---|---|---|
| 1 | 0.939 | fires / fires | 0.999 | fires / silent |
| 2 | 0.871 | fires / fires | 0.999 | fires / silent |
| 3 | 0.797 | fires / fires | 0.998 | fires / silent |

Absolute fires identically on both -- it cannot distinguish a catastrophic
crossing from harmless noise. Alpha moves on both (it is sensitive to
ordinal distance, not to which direction is safe), so an alpha reading alone
does not tell a reader which case occurred. Polarity crossing is
direction-aware by construction.

## Imbalance: the high-agreement / low-alpha paradox (Feinstein & Cicchetti)

At 90%-exclude imbalance, raw agreement barely moves while alpha swings
sharply (`-1 -> +1` flips):

| flips | raw agreement | alpha | delta from 1.0 |
|---|---|---|---|
| 1 | 0.990 | 0.944 | 0.056 |
| 2 | 0.980 | 0.893 | 0.108 |
| 3 | 0.970 | 0.845 | 0.155 |

One flip in 100 (99% raw agreement) already drops alpha under the candidate
0.95 cutoff. At this `n` and this imbalance, a fixed 0.95 threshold is not a
calibrated bound -- it is close to "one flip away," and which flip matters
more than how many.

## Summary

| | false-epoch rate, p=1%, n=100 | catches `+1→-1` | direction-aware | stable under imbalance |
|---|---|---|---|---|
| 1. Absolute | 63% -- unusable | yes | no | no |
| 2. Alpha < 0.95 | 45% -- unusable at this n | yes | no | no (paradox) |
| 3. Polarity, single crossing | 4.8% | yes | yes | yes |
| 3′. Polarity, ≥2 crossings | 0.06% | yes (correlated failure) | yes | yes |

## Recommendation

Hybrid, per vendor:

- **Hard trigger, opens an epoch:** one-directional polarity crossing
  (`baseline in {0, +1}`, `current == -1`), **≥2 crossings** on the sentinel
  set. Only rule that is both direction-aware and stable at this `n` and
  imbalance, and it measures exactly the failure mode that threatens
  recall -- correlated false negatives on the relevant class -- rather than
  disagreement in general.
- **Soft signal, logged, not epoch-opening:** alpha between baseline and
  current, reused from `attest.stats.agreement.krippendorff_alpha` alongside
  `raw_agreement` (per that module's own convention of never reporting alpha
  without it), flagged for human review below an advisory bound (looser than
  the shown-unstable 0.95, since it is advisory only). Catches drift outside
  the recall-critical direction that the hard trigger is deliberately blind
  to; not trusted alone, given the paradox above.

Rule 1 is not recommended in any form.

## Wiring

- Evaluated per vendor; one vendor's trigger opens an epoch for the whole
  config via the existing `attest.provenance.epochs.open_epoch` /
  `maybe_open_epoch` -- drift is a vendor behavior change, not necessarily a
  `Config` field change, so `maybe_open_epoch`'s content-hash check does not
  by itself see it; the caller opens explicitly on a hard-trigger firing.
- The transition is appended via `attest.provenance.changelog.ChangeLog.record(before=...,
  after=..., reason="sentinel drift: vendor <v>, <k> polarity crossings on sentinel set <hash>")`,
  the same event chain any other config-change trigger writes to.
- The drift check itself -- rule id, threshold, sentinel-set hash, and each
  vendor's result -- is persisted as provenance alongside the epoch/changelog
  entry, so a reader sees why an epoch opened without re-deriving it, the
  same role `tau_report.json` plays for `tau`.

## Split with the runbook

Everything above is a boundary-clean, offline statistical characterization:
no scheduler, no baseline store, no network. Sentinel scheduling, baseline
persistence, cadence, and the ≥2-crossing decision loop are operational
concerns and belong to the runbook, not to attest -- see the runbook for
that half.
