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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see test_CVE-2026-12345.py and fix_GHSA-crj3-q72p-ggqm", "see test_.py and fix_"),
        ("CVE-2026-12345abc", "abc"),
    ],
)
def test_strips_identifiers_next_to_underscore_or_letters(text, expected):
    assert sanitize(text) == expected


def test_find_identifiers_next_to_underscore_or_letters():
    text = "see test_CVE-2026-12345.py and fix_GHSA-crj3-q72p-ggqm, CVE-2026-54321abc"
    assert find_identifiers(text) == {
        "CVE-2026-12345",
        "CVE-2026-54321",
        "GHSA-CRJ3-Q72P-GGQM",
    }


@pytest.mark.parametrize(
    "text",
    ["XCVE-2026-12345", "1CVE-2026-12345", "aGHSA-crj3-q72p-ggqm", "GHSA-crj3-q72p-ggqmx"],
)
def test_identifiers_glued_to_alphanumerics_are_not_identifiers(text):
    assert find_identifiers(text) == set()


@pytest.mark.parametrize("dash", [chr(c) for c in range(0x2010, 0x2016)])
def test_unicode_dashes_inside_identifiers(dash):
    text = f"x CVE{dash}2026{dash}12345 y GHSA{dash}crj3{dash}q72p-ggqm z"
    assert find_identifiers(text) == {"CVE-2026-12345", "GHSA-CRJ3-Q72P-GGQM"}
    assert sanitize(text) == "x y z"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[advisory](https://x.test/a)and more", "advisoryand more"),
        ("[advisory](https://x.test/a) and more", "advisory and more"),
        ("<https://x.test/a>", ""),
        ("See <https://x.test/a> now", "See now"),
        ("Read **https://x.test/a** now", "Read now"),
        ("Wiki https://x.test/Foo_(bar) page", "Wiki page"),
        ("(https://x.test/Foo_(bar))", ""),
        ('href="https://x.test/a">link', 'href="">link'),
        ("[CVE-2026-12345](https://nvd.nist.gov/vuln/detail/CVE-2026-12345) fixed", "fixed"),
    ],
)
def test_url_boundaries_and_markup_artifacts(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("([CVE-2026-12345])", ""),
        ("Bug (CVE-2026-12345, GHSA-crj3-q72p-ggqm) fixed", "Bug fixed"),
        ("(CVE-2026-12345 ; CVE-2026-12346 :)", ""),
        ("Call foo() with CVE-2026-12345 here", "Call foo() with here"),
    ],
)
def test_removes_nested_and_punctuation_only_brackets(text, expected):
    assert sanitize(text) == expected


def test_keeps_empty_brackets_that_were_already_in_clean_text():
    text = "Call f() and g[] and h(, ) as before."
    assert sanitize(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "CVE-2026-(CVE-2025-11111)12345",
        "GHSA-crj3-<GHSA-aaaa-bbbb-cccc>q72p-ggqm",
        "CVE-2026-[https://x.test/a]12345",
    ],
)
def test_removal_cannot_join_text_into_a_new_identifier(text):
    assert find_identifiers(sanitize(text)) == set()


def test_bold_cleanup_does_not_span_separate_bold_runs():
    text = "**Note:** CVE-2026-12345 **Severity:** high"
    assert sanitize(text) == "**Note:** **Severity:** high"


def test_url_does_not_include_trailing_brace():
    assert sanitize("{https://x.test/a}") == "{}"


def test_find_identifiers_only_matches_ascii_digits():
    assert find_identifiers("CVE-٢٠٢٦-١٢٣٤٥") == set()
