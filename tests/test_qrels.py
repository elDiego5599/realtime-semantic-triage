from datetime import UTC, datetime

import pytest

from semtriage.models import Event, Source
from semtriage.qrels import build_qrels


def make_event(
    source: Source,
    event_id: str,
    day: int,
    *,
    cve: str | None = None,
    cwe: tuple[str, ...] = (),
    products: tuple[str, ...] = (),
) -> Event:
    when = datetime(2024, 1, day, tzinfo=UTC)
    return Event(
        event_id=f"{source.value}:{event_id}",
        source=source,
        source_id=event_id,
        published=when,
        modified=when,
        text="text",
        cve_ids=(cve,) if cve else (),
        cwe_ids=cwe,
        ecosystems=(),
        packages=(),
        products=products,
        cna=None,
        severity=None,
    )


def test_cve_qrels_other_source_only() -> None:
    query = make_event(Source.NVD, "q", 2, cve="CVE-1")
    twin = make_event(Source.GHSA, "twin", 1, cve="CVE-1")
    same_source = make_event(Source.NVD, "same", 1, cve="CVE-1")
    cve_qrels, _ = build_qrels([query, twin, same_source], [query.event_id])
    assert cve_qrels == {query.event_id: {twin.event_id}}


def test_future_events_excluded() -> None:
    query = make_event(Source.NVD, "q", 2, cve="CVE-1", cwe=("CWE-1",), products=("p",))
    past = make_event(Source.GHSA, "past", 1, cve="CVE-1")
    future = make_event(Source.GHSA, "future", 3, cve="CVE-1", cwe=("CWE-1",), products=("p",))
    cve_qrels, graded_qrels = build_qrels([query, past, future], [query.event_id])
    assert cve_qrels[query.event_id] == {past.event_id}
    assert graded_qrels[query.event_id] == {past.event_id: 2.0}


def test_same_time_candidate_eligible() -> None:
    query = make_event(Source.NVD, "q", 2, cve="CVE-1")
    twin = make_event(Source.GHSA, "twin", 2, cve="CVE-1")
    cve_qrels, _ = build_qrels([query, twin], [query.event_id])
    assert cve_qrels[query.event_id] == {twin.event_id}


def test_graded_needs_cwe_and_product() -> None:
    query = make_event(Source.NVD, "q", 5, cwe=("CWE-1",), products=("p",))
    cwe_only = make_event(Source.GHSA, "cwe-only", 1, cwe=("CWE-1",))
    product_only = make_event(Source.GHSA, "product-only", 1, products=("p",))
    both = make_event(Source.GHSA, "both", 1, cwe=("CWE-1",), products=("p",))
    events = [query, cwe_only, product_only, both]
    cve_qrels, graded_qrels = build_qrels(events, [query.event_id])
    assert cve_qrels[query.event_id] == set()
    assert graded_qrels[query.event_id] == {both.event_id: 1.0}


def test_cve_match_gain_two_wins_over_graded() -> None:
    query = make_event(Source.NVD, "q", 5, cve="CVE-1", cwe=("CWE-1",), products=("p",))
    twin = make_event(Source.GHSA, "twin", 1, cve="CVE-1", cwe=("CWE-1",), products=("p",))
    own_source = make_event(Source.NVD, "own", 1, cwe=("CWE-1",), products=("p",))
    _, graded_qrels = build_qrels([query, twin, own_source], [query.event_id])
    assert graded_qrels[query.event_id] == {twin.event_id: 2.0, own_source.event_id: 1.0}


def test_query_without_positives_present_empty() -> None:
    query = make_event(Source.NVD, "q", 1)
    cve_qrels, graded_qrels = build_qrels([query], [query.event_id])
    assert cve_qrels == {query.event_id: set()}
    assert graded_qrels == {query.event_id: {}}


def test_unknown_query_raises_keyerror() -> None:
    query = make_event(Source.NVD, "q", 1)
    with pytest.raises(KeyError):
        build_qrels([query], ["nvd:missing"])
