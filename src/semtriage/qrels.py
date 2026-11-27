"""Build relevance judgments for advisory retrieval.

A candidate is eligible for a query when it is not the query itself and was
published at or before the query. The CVE qrels hold the eligible events from
the other source that share at least one CVE id with the query. The graded
qrels give gain 2.0 to those CVE matches and gain 1.0 to every other eligible
event, from any source, that shares at least one CWE id and at least one
product key with the query.
"""

from collections.abc import Sequence

from semtriage.models import Event


def build_qrels(
    events: Sequence[Event], query_ids: Sequence[str]
) -> tuple[dict[str, set[str]], dict[str, dict[str, float]]]:
    """Return ``(cve_qrels, graded_qrels)`` keyed by query event id.

    Queries without a positive still appear, with empty values. Raises KeyError
    for a query id that is not an event id.
    """
    index_by_id = {event.event_id: index for index, event in enumerate(events)}
    by_cve: dict[str, list[int]] = {}
    by_cwe_product: dict[tuple[str, str], list[int]] = {}
    for index, event in enumerate(events):
        for cve in event.cve_ids:
            by_cve.setdefault(cve, []).append(index)
        for cwe in event.cwe_ids:
            for product in event.products:
                by_cwe_product.setdefault((cwe, product), []).append(index)

    cve_qrels: dict[str, set[str]] = {}
    graded_qrels: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        query = events[index_by_id[query_id]]
        cve_matches: set[str] = set()
        for cve in query.cve_ids:
            for index in by_cve.get(cve, ()):
                candidate = events[index]
                if candidate.source != query.source and candidate.published <= query.published:
                    cve_matches.add(candidate.event_id)
        graded: dict[str, float] = dict.fromkeys(cve_matches, 2.0)
        for cwe in query.cwe_ids:
            for product in query.products:
                for index in by_cwe_product.get((cwe, product), ()):
                    candidate = events[index]
                    if candidate.event_id != query_id and candidate.published <= query.published:
                        graded.setdefault(candidate.event_id, 1.0)
        cve_qrels[query_id] = cve_matches
        graded_qrels[query_id] = graded
    return cve_qrels, graded_qrels
