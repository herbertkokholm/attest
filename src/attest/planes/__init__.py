"""Statistically separated planes: adjudication, active learning, and random recall audit.

Every row produced by a plane module is stamped with one of the constants
below, via a dataclass field declared `init=False` so the tag cannot be
overridden by a caller. This is the firewall's foundation: `recall_audit`'s
`build_strata` refuses any row whose `plane` is not `PLANE_RECALL_AUDIT`
before it can contribute to a recall estimate.
"""

from __future__ import annotations

PLANE_ADJUDICATION = "adjudication"
PLANE_ACTIVE_LEARNING = "active_learning"
PLANE_RECALL_AUDIT = "recall_audit"

__all__ = ["PLANE_ADJUDICATION", "PLANE_ACTIVE_LEARNING", "PLANE_RECALL_AUDIT"]
