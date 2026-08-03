"""Reproducible probe backing `docs/sentinel_drift_rule.md`.

Characterizes three candidate latent-vendor-drift threshold rules against a
synthetic per-vendor sentinel set on the ordinal domain `{-1, 0, +1}`, at a
class balance representative of a heavily exclude-dominated systematic-review
track. Reuses `attest.stats.agreement.krippendorff_alpha` /
`raw_agreement` for the only statistic any rule needs; no statistic here is
reimplemented.

This module has no sentinel harness, no baseline store, and no scheduler --
it is a pure, offline characterization of rule *behavior*, run once to
produce the numbers cited in the doc. It imports only
`attest.stats.agreement` and the standard library, matching the kernel's own
boundary (see `tools/check_boundary.py`), even though this path sits outside
`src/attest` and is not itself scanned by that guard.

Run directly: `python examples/sentinel_drift_probe.py`
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attest.stats.agreement import krippendorff_alpha  # noqa: E402

N = 100
# Heavily exclude-dominated, representative of a typical systematic-review
# screening track. Not read from a specific review's data -- a stated,
# literature-typical stand-in for reproducibility outside any one dataset.
BASELINE_COMPOSITION = {-1: 90, 0: 3, 1: 7}
DOMAIN = (-1, 0, 1)


def make_baseline(n: int, composition: dict[int, int], seed: int = 0) -> list[int]:
    assert sum(composition.values()) == n
    values = [v for v, count in composition.items() for _ in range(count)]
    random.Random(seed).shuffle(values)
    return values


def alpha_of(baseline: list[int], current: list[int]) -> float:
    return krippendorff_alpha([[float(v) for v in baseline], [float(v) for v in current]])


def rule_absolute(baseline: list[int], current: list[int]) -> bool:
    return any(b != c for b, c in zip(baseline, current, strict=True))


def rule_alpha(baseline: list[int], current: list[int], tau: float = 0.95) -> bool:
    return alpha_of(baseline, current) < tau


def rule_polarity(baseline: list[int], current: list[int], *, both_directions: bool) -> bool:
    for b, c in zip(baseline, current, strict=True):
        if b in (0, 1) and c == -1:
            return True
        if both_directions and b == -1 and c in (0, 1):
            return True
    return False


def crossing_count(baseline: list[int], current: list[int]) -> int:
    return sum(1 for b, c in zip(baseline, current, strict=True) if b in (0, 1) and c == -1)


def benign_flip_trial(baseline: list[int], p: float, rng: random.Random) -> list[int]:
    """One draw of independent per-record benign flip noise at rate `p`.

    Models temperature-0 nondeterminism (backend/batching/version-pinning
    noise), not real drift: each record independently flips to a uniformly
    random *different* domain value with probability `p`. See the doc for
    why `p` is meaningful only once temperature is a hashed config field.
    """
    current = list(baseline)
    for i, v in enumerate(current):
        if rng.random() < p:
            current[i] = rng.choice([d for d in DOMAIN if d != v])
    return current


def false_epoch_rate(
    baseline: list[int],
    p: float,
    rule: Callable[[list[int], list[int]], bool],
    trials: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    fires = sum(1 for _ in range(trials) if rule(baseline, benign_flip_trial(baseline, p, rng)))
    return fires / trials


def inject_flips(baseline: list[int], frm: int, to: int, k: int) -> list[int]:
    current = list(baseline)
    for i in [i for i, v in enumerate(current) if v == frm][:k]:
        current[i] = to
    return current


def main() -> None:
    baseline = make_baseline(N, BASELINE_COMPOSITION)
    print(f"sentinel: n={N}, composition={BASELINE_COMPOSITION}\n")

    print("benign false-epoch rate (no real drift), by rule:")
    print(
        f"{'p':>6} {'absolute':>10} {'alpha<0.95':>11} {'polarity(1x)':>13} "
        f"{'polarity(>=2,1dir)':>19} {'polarity(2dir)':>15}"
    )
    for p in (0.005, 0.01, 0.02):
        r_abs = false_epoch_rate(baseline, p, rule_absolute, 4000, seed=1)
        r_alpha = false_epoch_rate(baseline, p, rule_alpha, 4000, seed=2)
        r_pol1 = false_epoch_rate(
            baseline, p, lambda b, c: rule_polarity(b, c, both_directions=False), 8000, seed=3
        )
        r_pol_k2 = false_epoch_rate(
            baseline, p, lambda b, c: crossing_count(b, c) >= 2, 8000, seed=4
        )
        r_pol2 = false_epoch_rate(
            baseline, p, lambda b, c: rule_polarity(b, c, both_directions=True), 4000, seed=5
        )
        analytic_abs = 1 - (1 - p) ** N
        print(
            f"{p:6.3f} {r_abs:10.4f} {r_alpha:11.4f} {r_pol1:13.4f} {r_pol_k2:19.4f} "
            f"{r_pol2:15.4f}  (analytic absolute: {analytic_abs:.4f})"
        )

    print("\ndirection sensitivity (k injected flips):")
    print(
        f"{'k':>3}  {'+1->-1 alpha':>13} {'abs/polarity':>13}   "
        f"{'0->+1 alpha':>12} {'abs/polarity':>13}"
    )
    for k in (1, 2, 3, 5, 10):
        bad = inject_flips(baseline, frm=1, to=-1, k=k)
        benign = inject_flips(baseline, frm=0, to=1, k=k)
        bad_flags = (
            f"{rule_absolute(baseline, bad)!s:>5}/"
            f"{rule_polarity(baseline, bad, both_directions=False)!s:<5}"
        )
        benign_flags = (
            f"{rule_absolute(baseline, benign)!s:>5}/"
            f"{rule_polarity(baseline, benign, both_directions=False)!s:<5}"
        )
        print(
            f"{k:3d}  {alpha_of(baseline, bad):13.4f} {bad_flags}   "
            f"{alpha_of(baseline, benign):12.4f} {benign_flags}"
        )

    print("\nimbalance sensitivity: alpha delta from 1.0 per flip (-1 -> +1):")
    for k in (1, 2, 3):
        cur = inject_flips(baseline, frm=-1, to=1, k=k)
        a = alpha_of(baseline, cur)
        raw = sum(1 for b, c in zip(baseline, cur, strict=True) if b == c) / N
        print(f"  k={k}: alpha={a:.4f} (delta={1 - a:.4f}), raw_agreement={raw:.3f}")


if __name__ == "__main__":
    main()
