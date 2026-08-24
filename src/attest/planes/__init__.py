"""Statistically separated planes: adjudication, active learning, and the two random audits.

Every row produced by a plane module is stamped with one of the constants
below, via a dataclass field declared `init=False` so the tag cannot be
overridden by a caller. This is the firewall's foundation: `recall_audit`'s
`build_strata` refuses any row whose `plane` is not `PLANE_RECALL_AUDIT`
before it can contribute a false-negative count to a recall estimate, and
`inclusion_audit`'s `build_inclusion_strata` enforces the same refusal for
`PLANE_INCLUSION_AUDIT` on the true-positive side. The two audits sample
disjoint populations (screen-excluded vs. include-and-escalate) by
construction, which makes their samples independent -- though the
Bonferroni combination `stratified_recall_with_audited_tp` uses to pair
their one-sided error budgets does not actually require that independence
to be valid; see that function's docstring for why, and for the tighter
Sidak alternative the independence would license instead.
"""

from __future__ import annotations

PLANE_ADJUDICATION = "adjudication"
PLANE_ACTIVE_LEARNING = "active_learning"
PLANE_RECALL_AUDIT = "recall_audit"
PLANE_INCLUSION_AUDIT = "inclusion_audit"

__all__ = [
    "PLANE_ADJUDICATION",
    "PLANE_ACTIVE_LEARNING",
    "PLANE_RECALL_AUDIT",
    "PLANE_INCLUSION_AUDIT",
]
