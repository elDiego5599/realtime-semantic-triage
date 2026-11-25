"""Clustering metrics for predicted and gold advisory labels."""

from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scores:
    """Precision, recall, and F1 values."""

    precision: float
    recall: float
    f1: float


def _validate_and_contingency(
    pred: Mapping[str, Hashable],
    gold: Mapping[str, Hashable],
) -> tuple[Counter[tuple[Hashable, Hashable]], Counter[Hashable], Counter[Hashable], int]:
    """Validate mappings and build their contingency table."""
    if not pred or not gold:
        raise ValueError("pred and gold mappings must not be empty")
    if set(pred) != set(gold):
        raise ValueError("pred and gold mappings must contain the same item IDs")

    contingency: Counter[tuple[Hashable, Hashable]] = Counter()
    pred_totals: Counter[Hashable] = Counter()
    gold_totals: Counter[Hashable] = Counter()
    for item_id, pred_label in pred.items():
        gold_label = gold[item_id]
        contingency[pred_label, gold_label] += 1
        pred_totals[pred_label] += 1
        gold_totals[gold_label] += 1
    return contingency, pred_totals, gold_totals, len(pred)


def _scores(precision: float, recall: float) -> Scores:
    """Build scores and calculate their harmonic mean."""
    f1 = (
        0.0 if precision == 0.0 and recall == 0.0 else 2 * precision * recall / (precision + recall)
    )
    return Scores(precision=precision, recall=recall, f1=f1)


def pairwise_scores(
    pred: Mapping[str, Hashable],
    gold: Mapping[str, Hashable],
) -> Scores:
    """Calculate pairwise precision, recall, and F1."""
    contingency, pred_totals, gold_totals, _ = _validate_and_contingency(pred, gold)

    predicted_pairs = sum(count * (count - 1) // 2 for count in pred_totals.values())
    gold_pairs = sum(count * (count - 1) // 2 for count in gold_totals.values())
    true_pairs = sum(count * (count - 1) // 2 for count in contingency.values())

    precision = 1.0 if predicted_pairs == 0 else true_pairs / predicted_pairs
    recall = 1.0 if gold_pairs == 0 else true_pairs / gold_pairs
    return _scores(precision, recall)


def bcubed_scores(
    pred: Mapping[str, Hashable],
    gold: Mapping[str, Hashable],
) -> Scores:
    """Calculate B-cubed precision, recall, and F1."""
    contingency, pred_totals, gold_totals, item_count = _validate_and_contingency(
        pred,
        gold,
    )

    precision_sum = sum(
        count * count / pred_totals[pred_label] for (pred_label, _), count in contingency.items()
    )
    recall_sum = sum(
        count * count / gold_totals[gold_label] for (_, gold_label), count in contingency.items()
    )
    precision = precision_sum / item_count
    recall = recall_sum / item_count
    return _scores(precision, recall)
