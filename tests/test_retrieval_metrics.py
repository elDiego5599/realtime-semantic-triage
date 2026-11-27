import math

import pytest

from semtriage.retrieval_metrics import (
    RetrievalReport,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# recall_at_k


def test_recall_hand_computed():
    # relevant {a, c, e}; first 3 results hold a and c
    assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "c", "e"}, 3) == pytest.approx(2 / 3)


def test_recall_k_cutoff_is_inclusive():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0


def test_recall_k_larger_than_ranking():
    assert recall_at_k(["a", "b"], {"b", "z"}, 10) == pytest.approx(0.5)


def test_recall_divides_by_relevant_count():
    # 4 results, 2 relevant ids, 1 found
    assert recall_at_k(["a", "b", "c", "d"], {"a", "z"}, 4) == pytest.approx(0.5)


def test_recall_empty_ranking():
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_recall_empty_relevant_raises():
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), 5)


def test_recall_graded_mapping_counts_positive_gains_only():
    # c has gain 0, so only a and b are relevant; a is found
    assert recall_at_k(["a", "c"], {"a": 2.0, "b": 1.0, "c": 0.0}, 5) == pytest.approx(0.5)


def test_recall_mapping_with_only_zero_gains_raises():
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], {"a": 0.0}, 5)


# reciprocal_rank


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_is_one_based():
    assert reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_reciprocal_rank_empty_cases():
    assert reciprocal_rank([], {"a"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


def test_reciprocal_rank_skips_zero_gain():
    assert reciprocal_rank(["a", "b"], {"a": 0.0, "b": 1.0}) == pytest.approx(0.5)


# ndcg_at_k


def test_ndcg_binary_uses_log2_discount():
    # DCG = 1 / log2(3) = 0.6309; ideal DCG = 1 / log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(0.6309297535714575)


def test_ndcg_graded_hand_computed():
    # ranking a, b, c with gains 1, 0, 3 (d is past k, e is never retrieved)
    # DCG@3 = 1/log2(2) + 0/log2(3) + 3/log2(4) = 1 + 0 + 1.5 = 2.5
    # ideal gains sorted 3, 2, 1:
    # IDCG@3 = 3/1 + 2/log2(3) + 1/2 = 3.5 + 1.2618595 = 4.7618595
    # nDCG@3 = 2.5 / 4.7618595 = 0.5250050
    relevant = {"a": 1.0, "c": 3.0, "e": 2.0}
    assert ndcg_at_k(["a", "b", "c", "d"], relevant, 3) == pytest.approx(0.5250049893849101)


def test_ndcg_ideal_dcg_is_cut_at_k():
    # three relevant ids but k = 1: ideal DCG is 1, a hit at rank 1 scores 1
    assert ndcg_at_k(["a"], {"a", "b", "c"}, 1) == 1.0


def test_ndcg_k_cutoff():
    # a is at rank 3, outside k = 2
    assert ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(["b", "a", "c"], {"a": 1.0, "b": 2.0}, 3) == pytest.approx(1.0)


def test_ndcg_ties_in_order_are_discounted_by_position():
    # a and b have equal gain: either order is ideal
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 1.0, "b": 1.0}, 2) == pytest.approx(1.0)
    # a tied pair pushed down one position: DCG = 1/log2(3) + 1/log2(4)
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(expected)


def test_ndcg_k_larger_than_ranking():
    # DCG = 1 (a at rank 1); ideal over two relevant ids = 1 + 1/log2(3)
    expected = 1 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a"], {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_zero_ideal_dcg():
    assert ndcg_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], {"a": 0.0}, 5) == 0.0


def test_ndcg_empty_ranking():
    assert ndcg_at_k([], {"a"}, 5) == 0.0


@pytest.mark.parametrize("gain", [-1.0, math.nan, math.inf])
def test_ndcg_bad_gain_raises(gain):
    with pytest.raises(ValueError, match="gain"):
        ndcg_at_k(["a"], {"a": gain}, 1)


@pytest.mark.parametrize("gain", ["1", True, None])
def test_ndcg_non_numeric_gain_raises(gain):
    with pytest.raises(TypeError, match="gain"):
        ndcg_at_k(["a"], {"a": gain}, 1)


# input validation shared by the metrics


def _calls(ranked, relevant, k):
    return [
        lambda: recall_at_k(ranked, relevant, k),
        lambda: ndcg_at_k(ranked, relevant, k),
    ]


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(k):
    for call in _calls(["a"], {"a"}, k):
        with pytest.raises(ValueError, match="k"):
            call()


@pytest.mark.parametrize("k", [1.0, True, "1"])
def test_non_int_k_raises(k):
    for call in _calls(["a"], {"a"}, k):
        with pytest.raises(TypeError, match="k"):
            call()


def test_duplicate_ids_in_ranking_raise():
    for call in [*_calls(["a", "b", "a"], {"a"}, 3), lambda: reciprocal_rank(["a", "a"], {"a"})]:
        with pytest.raises(ValueError, match="duplicate"):
            call()


def test_duplicate_beyond_k_still_raises():
    with pytest.raises(ValueError, match="duplicate"):
        recall_at_k(["a", "b", "b"], {"a"}, 1)


@pytest.mark.parametrize(
    ("ranked", "relevant"),
    [
        ("ab", {"a"}),
        (["a", 1], {"a"}),
        (["a"], "a"),
        (["a"], {"a", 1}),
        (["a"], {1: 1.0}),
    ],
)
def test_non_string_ids_raise(ranked, relevant):
    calls = [*_calls(ranked, relevant, 1), lambda: reciprocal_rank(ranked, relevant)]
    for call in calls:
        with pytest.raises(TypeError):
            call()


# evaluate


def test_evaluate_means_over_judged_queries():
    runs = {
        "q1": ["a", "b", "c"],  # relevant {b}: recall@1 0, recall@2 1, rr 1/2
        "q2": ["x", "y"],  # relevant {y, z}: recall@1 0, recall@2 1/2, rr 1/2
    }
    qrels = {"q1": {"b"}, "q2": {"y", "z"}}
    report = evaluate(runs, qrels, ks=(1, 2))
    assert isinstance(report, RetrievalReport)
    assert report.evaluated == 2
    assert report.skipped == 0
    assert report.no_relevant == 0
    assert report.means["recall@1"] == 0.0
    assert report.means["recall@2"] == pytest.approx(0.75)
    assert report.means["mrr"] == pytest.approx(0.5)
    # q1 ndcg@2 = 1/log2(3); q2 ndcg@2 = (1/log2(3)) / (1 + 1/log2(3))
    q1 = 1 / math.log2(3)
    q2 = q1 / (1 + q1)
    assert report.means["ndcg@2"] == pytest.approx((q1 + q2) / 2)
    assert report.means["ndcg@1"] == 0.0


def test_evaluate_metric_names_and_default_ks():
    report = evaluate({"q": ["a"]}, {"q": {"a"}})
    assert list(report.means) == [
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
    ]
    assert all(value == 1.0 for value in report.means.values())


def test_evaluate_missing_run_scores_zero():
    report = evaluate({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, ks=(1,))
    assert report.evaluated == 2
    assert report.means == {"recall@1": 0.5, "mrr": 0.5, "ndcg@1": 0.5}


def test_evaluate_counts_skipped_and_no_relevant():
    runs = {"q1": ["a"], "extra": ["a"], "empty": ["a"]}
    qrels = {"q1": {"a"}, "empty": set(), "zero": {"a": 0.0}}
    report = evaluate(runs, qrels, ks=(1,))
    assert report.evaluated == 1
    assert report.skipped == 1
    assert report.no_relevant == 2
    assert report.means["mrr"] == 1.0


def test_evaluate_graded_qrels():
    report = evaluate({"q": ["a", "b", "c", "d"]}, {"q": {"a": 1.0, "c": 3.0, "e": 2.0}}, ks=(3,))
    assert report.means["ndcg@3"] == pytest.approx(0.5250049893849101)
    assert report.means["recall@3"] == pytest.approx(2 / 3)


def test_evaluate_without_judged_queries_raises():
    with pytest.raises(ValueError, match="no query"):
        evaluate({"q": ["a"]}, {"q": set()})


@pytest.mark.parametrize("ks", [(), (0,), (5, 5)])
def test_evaluate_bad_ks_raise(ks):
    with pytest.raises(ValueError):
        evaluate({"q": ["a"]}, {"q": {"a"}}, ks=ks)


def test_evaluate_non_string_query_ids_raise():
    with pytest.raises(TypeError, match="query"):
        evaluate({1: ["a"]}, {"q": {"a"}})
    with pytest.raises(TypeError, match="query"):
        evaluate({"q": ["a"]}, {1: {"a"}})


def test_evaluate_validates_rankings_of_unjudged_queries():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate({"q": ["a"], "extra": ["b", "b"]}, {"q": {"a"}})


def test_report_is_immutable():
    report = evaluate({"q": ["a"]}, {"q": {"a"}}, ks=(1,))
    with pytest.raises(AttributeError):
        report.evaluated = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.means["mrr"] = 0.0  # type: ignore[index]
