import time

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
    assert sanitize(text) == "Details at ."


def test_removes_brackets_left_empty():
    text = (
        "Stored XSS (CVE-2026-64670) in client [GHSA-crj3-q72p-ggqm] and ( https://nvd.nist.gov/x )"
    )
    assert sanitize(text) == "Stored XSS in client"


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


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Title\n\n**CVE:** This vulnerability corresponds to CVE-2026-12345.\n\n### Summary",
            "Title\n\n### Summary",
        ),
        (
            "**CVE:** This vulnerability corresponds to "
            "[CVE-2026-12345](https://nvd.nist.gov/vuln/detail/CVE-2026-12345).",
            "",
        ),
        ("This issue corresponds to GHSA-crj3-q72p-ggqm. More text.", "More text."),
    ],
)
def test_removes_sentence_that_only_names_the_identifier(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Intro\nCVE: CVE-2026-12345\nBody", "Intro\n\nBody"),
        ("Intro\nCVE ID: CVE-2026-12345.\nBody", "Intro\n\nBody"),
        ("Intro\n**GHSA:** GHSA-crj3-q72p-ggqm\nBody", "Intro\n\nBody"),
        ("Intro\n- Fix commit: https://x.test/c/1\nBody", "Intro\n\nBody"),
        ("Intro\nPatch: https://x.test/p\nBody", "Intro\n\nBody"),
        ("Intro\n**Fix reference**: https://x.test/r\nBody", "Intro\n\nBody"),
    ],
)
def test_removes_label_lines_left_without_a_value(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("## References\n\n- https://x.test/a\n- https://x.test/b\n\nEnd", "## References\n\nEnd"),
        ("List\n* GHSA-crj3-q72p-ggqm\nEnd", "List\n\nEnd"),
        ("List\n- `https://x.test/a`: https://x.test/b\nEnd", "List\n\nEnd"),
        ("List\n- `CVE-2026-12345` / `CVE-2026-12346`\nEnd", "List\n\nEnd"),
    ],
)
def test_removes_list_items_left_without_content(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Use `https://x.test/a` here", "Use here"),
        ("Tracked as `CVE-2026-12345` upstream", "Tracked as upstream"),
        ("Run `curl https://x.test/a` now", "Run `curl ` now"),
    ],
)
def test_removes_inline_code_left_empty(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Incomplete fixes for CVE-2026-1111, CVE-2026-2222 and CVE-2026-3333.",
            "Incomplete fixes for .",
        ),
        ("Bypass of CVE-2026-1111 / GHSA-crj3-q72p-ggqm in parser", "Bypass of in parser"),
        ("See CVE-2026-1111 or CVE-2026-2222, both fixed", "See , both fixed"),
    ],
)
def test_collapses_separators_between_removed_identifiers(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "This field corresponds to the user id.",
        "Each entry corresponds to one request, and a, b and c are parsed.",
        "Fix: upgrade to 2.4.1.",
        "CVE: pending assignment",
        "- item one\n- item two",
        "Set `--flag` and `` to disable",
        "x = 1 / 2, or y | z",
        "## Patches\n\n-\n\nNone.",
        "Call f() and g[] and h(, ) as before.",
        "Use ``` code ``` fences and `a`, `b`.",
    ],
)
def test_residue_rules_leave_clean_text_untouched(text):
    assert sanitize(text) == text


def test_keeps_label_with_other_content_after_removal():
    text = "Fix: upgrade, see https://x.test/a"
    assert sanitize(text) == "Fix: upgrade, see"


def test_keeps_list_item_with_words_after_removal():
    text = "- https://x.test/a (upstream patch)"
    assert sanitize(text) == "- (upstream patch)"


def test_keeps_sentence_that_also_says_more():
    text = "The vulnerabilities tracked by CVE-2026-1111 are related to parsing."
    assert sanitize(text) == "The vulnerabilities tracked by are related to parsing."


def test_url_does_not_swallow_closing_backtick():
    assert sanitize("`https://x.test/a` text") == "text"


def test_residue_rules_cannot_join_text_into_a_new_identifier():
    text = "CVE-2026-`CVE-2025-11111`12345 and GHSA-crj3-CVE-2025-1 / CVE-2025-2-q72p-ggqm"
    assert find_identifiers(sanitize(text)) == set()


@pytest.mark.parametrize(
    "text",
    [
        "| https://x.test/a " + "| " * 40 + "| ok |",
        "https://a.test" + ", " * 40 + "x",
        "CVE-2026-1111 " + " and " * 40 + "x",
    ],
)
def test_separator_rule_runs_in_linear_time(text):
    start = time.perf_counter()
    sanitize(text * 50)
    assert time.perf_counter() - start < 1.0


def test_template_sentence_rule_keeps_prose_after_the_identifier():
    text = "This issue corresponds to CVE-2026-1111 in the parser and allows RCE."
    assert sanitize(text) == "This issue corresponds to in the parser and allows RCE."


def test_empty_inline_code_rule_does_not_merge_adjacent_code_spans():
    assert sanitize("`a`https://x.test/b`b`") == "`a``b`"


def test_label_rule_allows_at_most_one_word_before_the_label():
    assert sanitize("Apply the fix: https://x.test/a") == "Apply the fix:"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Intro\r\nCVE: CVE-2026-12345\r\nBody", "Intro\n\nBody"),
        ("Intro\r\n- https://x.test/a\r\nBody", "Intro\n\nBody"),
    ],
)
def test_residue_rules_handle_crlf_line_endings(text, expected):
    assert sanitize(text) == expected


def test_punctuation_line_rule_runs_in_linear_time():
    text = " " * 100_000 + "CVE-2026-12345" + " " * 100_000 + "x"
    start = time.perf_counter()
    sanitize(text)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Unrelated issue: CVE-2026-12345", "Unrelated issue:"),
        ("Disclosure report: https://x.test/a", "Disclosure report:"),
        ("Bad commit: https://x.test/c", "Bad commit:"),
    ],
)
def test_label_rule_keeps_lines_whose_label_has_a_free_word(text, expected):
    assert sanitize(text) == expected
