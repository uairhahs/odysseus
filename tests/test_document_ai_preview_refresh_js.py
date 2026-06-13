"""Regression guards for AI document updates while Markdown Preview is visible (#2182)."""

import re
from pathlib import Path

from tests.helpers.linter_compat import _norm

SRC = Path(__file__).resolve().parent.parent / "static/js/document.js"


def _function_body(name: str) -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        rf"\n\s*(?:export\s+)?(?:async\s+)?function\s+{name}\([^)]*\)\s*\{{", text
    )
    assert match, f"{name} not found"

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    assert depth == 0, f"{name} body did not close"
    return text[start : i - 1]


def test_markdown_preview_refresh_rerenders_visible_preview():
    body = _function_body("_refreshMarkdownPreviewIfVisible")
    haystack = _norm(body)
    assert "_isMarkdownPreviewVisible()" in haystack
    assert "lang !== 'markdown'" in haystack
    assert "textarea.value = content;" in haystack
    assert "syncHighlighting();" in haystack
    assert "_setMarkdownPreviewActive(true, { remember: false });" in haystack


def test_doc_update_refreshes_preview_instead_of_hidden_editor_animation():
    body = _norm(_function_body("handleDocUpdate"))

    visible = "const markdownPreviewWasVisible = _isMarkdownPreviewVisible();"
    exit_preview = "if (markdownPreviewWasVisible) _setMarkdownPreviewActive(false, { remember: false });"
    diff = "enterDiffMode(oldContent, newContent);"
    refresh = "markdownPreviewWasVisible && _refreshMarkdownPreviewIfVisible(docId, newContent)"
    animate = "_animateDocEdit(textarea, newContent);"

    assert visible in body
    assert exit_preview in body
    assert diff in body
    assert body.index(exit_preview) < body.index(diff)
    assert refresh in body
    assert body.index(refresh) < body.index(animate)
    assert "_refreshMarkdownPreviewIfVisible(docId, newContent);" in body
