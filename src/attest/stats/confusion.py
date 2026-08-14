"""Confusion-matrix counts (TP/FP/FN/TN) of predicted labels against gold truths.

Shared by every caller that scores a set of final labels against gold labels
on the binary include/exclude distinction -- `attest.io.store` (the stored,
per-epoch validation record) and `attest.ablation.xsweep` (the offline,
controlled sweep) both need exactly this count, computed the same way, so it
lives here once rather than being re-derived per caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

RELEVANT_LABEL = 1


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion-matrix counts on the relevant/not-relevant distinction.

    Attributes:
        tp: Predicted relevant, truly relevant.
        fp: Predicted relevant, truly not relevant.
        fn: Predicted not relevant, truly relevant.
        tn: Predicted not relevant, truly not relevant.
    """

    tp: int
    fp: int
    fn: int
    tn: int


def confusion_matrix(
    predictions: Mapping[str, int],
    truths: Mapping[str, int],
    *,
    relevant_label: int = RELEVANT_LABEL,
) -> ConfusionMatrix:
    """Compute TP/FP/FN/TN of `predictions` against `truths`.

    Only record ids present in both `predictions` and `truths` are counted;
    a prediction with no gold label (or vice versa) is silently skipped
    rather than raising, since callers routinely pass a `predictions` set
    that is a subset of a larger `truths` mapping (e.g. only escalated or
    already-resolved records).

    Args:
        predictions: Mapping of record id to final ordinal label.
        truths: Mapping of record id to gold ordinal label.
        relevant_label: The ordinal label counted as "relevant" on both
            sides of the comparison. Defaults to `RELEVANT_LABEL` (+1).

    Returns:
        The `ConfusionMatrix` over the shared record ids.
    """
    tp = fp = fn = tn = 0
    for record_id, label in predictions.items():
        if record_id not in truths:
            continue
        predicted_positive = label == relevant_label
        actual_positive = truths[record_id] == relevant_label
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


__all__ = ["RELEVANT_LABEL", "ConfusionMatrix", "confusion_matrix"]
