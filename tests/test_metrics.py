import time

import pytest

from semtriage.metrics import Scores, bcubed_scores, pairwise_scores

GOLD = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 2}
PRED = {"a": "x", "b": "x", "c": "y", "d": "y", "e": "z"}


def approx(scores: Scores, precision: float, recall: float, f1: float) -> None:
    assert scores.precision == pytest.approx(precision)
    assert scores.recall == pytest.approx(recall)
    assert scores.f1 == pytest.approx(f1)


def test_scores_is_immutable():
    scores = Scores(precision=1.0, recall=1.0, f1=1.0)
    with pytest.raises(AttributeError):
        scores.f1 = 0.0  # type: ignore[misc]


def test_pairwise_hand_computed():
    # gold pairs: ab ac bc de, pred pairs: ab cd, true positive: ab
    approx(pairwise_scores(PRED, GOLD), 0.5, 0.25, 1 / 3)


def test_bcubed_hand_computed():
    approx(bcubed_scores(PRED, GOLD), 0.8, 8 / 15, 0.64)


@pytest.mark.parametrize("score", [pairwise_scores, bcubed_scores])
def test_perfect_clustering(score):
    pred = {k: f"p{v}" for k, v in GOLD.items()}
    approx(score(pred, GOLD), 1.0, 1.0, 1.0)


def test_label_values_are_arbitrary():
    pred = {"a": None, "b": None, "c": (1, 2), "d": (1, 2), "e": 3.5}
    approx(pairwise_scores(pred, GOLD), 0.5, 0.25, 1 / 3)


def test_pairwise_all_singletons_predicted():
    pred = {k: k for k in GOLD}
    approx(pairwise_scores(pred, GOLD), 1.0, 0.0, 0.0)


def test_pairwise_everything_merged():
    pred = {k: 0 for k in GOLD}
    # 10 predicted pairs, 4 gold pairs, all gold pairs found
    approx(pairwise_scores(pred, GOLD), 0.4, 1.0, 2 * 0.4 / 1.4)


def test_pairwise_no_pairs_anywhere():
    items = {"a": 1, "b": 2, "c": 3}
    approx(pairwise_scores(items, items), 1.0, 1.0, 1.0)


def test_bcubed_all_singletons_predicted():
    pred = {k: k for k in GOLD}
    # recall per item is 1 / gold cluster size: 1/3 three times, 1/2 twice
    approx(bcubed_scores(pred, GOLD), 1.0, 2 / 5, 2 * 0.4 / 1.4)


def test_bcubed_everything_merged():
    pred = {k: 0 for k in GOLD}
    # precision per item is gold cluster size / 5: 3/5 three times, 2/5 twice
    approx(bcubed_scores(pred, GOLD), 13 / 25, 1.0, 2 * (13 / 25) / (1 + 13 / 25))


@pytest.mark.parametrize("score", [pairwise_scores, bcubed_scores])
def test_mismatched_items_raise(score):
    with pytest.raises(ValueError):
        score({"a": 1, "b": 1}, {"a": 1, "c": 1})


@pytest.mark.parametrize("score", [pairwise_scores, bcubed_scores])
def test_empty_input_raises(score):
    with pytest.raises(ValueError):
        score({}, {})


@pytest.mark.parametrize("score", [pairwise_scores, bcubed_scores])
def test_scales_linearly(score):
    n = 200_000
    gold = {str(i): i // 3 for i in range(n)}
    pred = {str(i): i // 2 for i in range(n)}
    start = time.perf_counter()
    result = score(pred, gold)
    assert time.perf_counter() - start < 2.0
    assert 0.0 < result.f1 < 1.0
