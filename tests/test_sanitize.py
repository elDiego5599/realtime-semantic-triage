import pytest

from semtriage.sanitize import find_identifiers, sanitize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Fixed in CVE-2026-64670.", "Fixed in ."),
        ("see cve-2024-1234 now", "see now"),
        ("Tracked as GHSA-crj3-q72p-ggqm upstream", "Tracked as upstream"),
        ("ghsa-crj3-q72p-ggqm", ""),
        ("CVE-2026-129230 and CVE-2026-1", "and CVE-2026-1"),
    ],
)
def test_strips_identifiers(text, expected):
    assert sanitize(text) == expected


def test_strips_urls():
    text = (
        "Details at https://github.com/advisories/GHSA-crj3-q72p-ggqm and http://example.com/x?y=1."
    )
    assert sanitize(text) == "Details at and ."


def test_removes_brackets_left_empty():
    text = (
        "Stored XSS (CVE-2026-64670) in client [GHSA-crj3-q72p-ggqm] and ( https://nvd.nist.gov/x )"
    )
    assert sanitize(text) == "Stored XSS in client and"


def test_collapses_whitespace_but_keeps_paragraphs():
    text = "### Summary\n\n`@waline/client`   renders\tthe preview\n\n\n\nImpact:  high  "
    assert sanitize(text) == "### Summary\n\n`@waline/client` renders the preview\n\nImpact: high"


def test_leaves_clean_text_untouched():
    text = "Memory leak in Juniper JUNOS allows remote attackers to cause a denial of service."
    assert sanitize(text) == text


def test_does_not_touch_version_numbers_or_similar_tokens():
    text = "Affects 2026-11-25 builds, versions 3.15.2 and prior, and CVSS 9.8."
    assert sanitize(text) == text


def test_sanitized_text_has_no_identifiers():
    text = "CVE-2026-64670 / GHSA-crj3-q72p-ggqm, see https://osv.dev/vulnerability/CVE-2026-64670"
    assert find_identifiers(sanitize(text)) == set()


def test_find_identifiers_normalizes_case_and_deduplicates():
    text = "cve-2026-64670, CVE-2026-64670 and ghsa-CRJ3-q72p-ggqm"
    assert find_identifiers(text) == {"CVE-2026-64670", "GHSA-CRJ3-Q72P-GGQM"}


def test_find_identifiers_inside_urls():
    text = "https://github.com/advisories/GHSA-crj3-q72p-ggqm"
    assert find_identifiers(text) == {"GHSA-CRJ3-Q72P-GGQM"}


def test_empty_input():
    assert sanitize("") == ""
    assert find_identifiers("") == set()
