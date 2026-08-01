"""Recall estimation from random audit samples: sensitivity with a rule-of-three worst-case floor.

Screening recall is TP/(TP+FN). Because it can only be estimated from the
random recall audit plane, sample sizes are typically small; the point
estimate must always be reported alongside a rule-of-three worst-case floor,
never alone.
"""


def recall_with_floor(true_positives: int, false_negatives: int) -> tuple[float, float]:
    """Compute the recall point estimate and its rule-of-three floor (not yet implemented)."""
    raise NotImplementedError
