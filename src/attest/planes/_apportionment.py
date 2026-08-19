"""Shared largest-remainder apportionment, used by both audit-draw planes.

Extracted from `attest.planes.recall_audit` so `attest.planes.inclusion_audit`
draws against the exact same allocation rule rather than a second copy of it
-- both directions of the audit firewall should allocate a stratified draw
identically, not merely similarly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def allocate_proportionally(n: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Allocate n draws across strata proportional to size, via largest-remainder apportionment.

    Guarantees each stratum's allocation never exceeds its size, and the
    allocations sum exactly to n.
    """
    total = sum(sizes.values())
    raw = {name: n * size / total for name, size in sizes.items()}
    floors = {name: math.floor(value) for name, value in raw.items()}
    remainder = n - sum(floors.values())
    by_largest_fraction = sorted(sizes, key=lambda name: raw[name] - floors[name], reverse=True)
    for name in by_largest_fraction[:remainder]:
        floors[name] += 1
    return floors
