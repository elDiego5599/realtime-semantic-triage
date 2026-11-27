from datetime import UTC, datetime

from semtriage.models import Event


def test_event_without_products_loads():
    event = Event.model_validate(
        {
            "event_id": "nvd:CVE-2026-00001",
            "source": "nvd",
            "source_id": "CVE-2026-00001",
            "published": datetime(2026, 1, 1, tzinfo=UTC),
            "modified": datetime(2026, 1, 1, tzinfo=UTC),
            "text": "A vulnerability.",
            "cve_ids": ["CVE-2026-00001"],
            "cwe_ids": [],
            "ecosystems": [],
            "packages": [],
            "cna": None,
            "severity": None,
        }
    )
    assert event.products == ()
