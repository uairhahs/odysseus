"""Regression guards for the Documents editor scrolling UI.

Issues #1501 and #1496 both come from the same surface: the document editor
hid its real textarea scrollbar, and the line-number gutter tried to scroll an
overflow-hidden element. Long wrapped lines add another wrinkle: the textarea
can have more visual rows than logical newline rows, so the gutter rows must
match the textarea's measured row heights. Keep these as static checks because
document.js is browser-coupled and not importable in pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    """Normalize whitespace and quote style so cosmetic differences don't matter."""
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


DOC_JS = _norm((ROOT / "static/js/document.js").read_text(encoding="utf-8"))
STYLE_CSS = _norm((ROOT / "static/style.css").read_text(encoding="utf-8"))


def test_document_textarea_scrollbar_is_visible():
    textarea_rule_start = STYLE_CSS.index(".doc-editor-textarea { position: absolute;")
    textarea_rule_end = STYLE_CSS.index(
        ".doc-editor-textarea::-webkit-scrollbar {", textarea_rule_start
    )
    textarea_css = STYLE_CSS[textarea_rule_start:textarea_rule_end]

    assert "overflow-y: scroll;" in textarea_css
    assert "scrollbar-width: thin;" in textarea_css
    assert ".doc-editor-textarea::-webkit-scrollbar { width: 8px; }" in STYLE_CSS
    assert ".doc-editor-textarea::-webkit-scrollbar { display: none; }" not in STYLE_CSS


def test_line_number_gutter_translates_inner_content():
    assertions = [
        ("function _lineNumberContentEl(gutter)", True, DOC_JS),
        ('inner.className = "doc-line-number-content";', True, DOC_JS),
        ("`translateY(${-textarea.scrollTop}px)`;", True, DOC_JS),
        ("gutter.scrollTop = textarea.scrollTop;", False, DOC_JS),
        (".doc-line-number-content", True, STYLE_CSS),
    ]
    for snippet, should_exist, haystack in assertions:
        snippet = _norm(snippet)
        haystack = _norm(haystack)
        if should_exist:
            assert snippet in haystack
        else:
            assert snippet not in haystack


def test_line_number_gutter_accounts_for_wrapped_rows():
    assertions = [
        (
            "function _measureLineNumberHeights(textarea, lines, textWidth, style)",
            True,
            DOC_JS,
        ),
        ("probe = document.createElement('textarea');", True, DOC_JS),
        ("probe.wrap = 'soft';", True, DOC_JS),
        ("probe.value = line || ' ';", True, DOC_JS),
        ("Math.round(probe.scrollHeight / lineHeight)", True, DOC_JS),
        ("row.style.height = `${heights[i]}px`;", True, DOC_JS),
        ('label.className = "doc-line-number-label";', True, DOC_JS),
        ("inner.textContent=lines;", False, DOC_JS),
        (".doc-line-number-row", True, STYLE_CSS),
        (".doc-line-number-label", True, STYLE_CSS),
        (".doc-line-number-measure", True, STYLE_CSS),
    ]
    for snippet, should_exist, haystack in assertions:
        snippet = _norm(snippet)
        haystack = _norm(haystack)
        if should_exist:
            assert snippet in haystack
        else:
            assert snippet not in haystack
