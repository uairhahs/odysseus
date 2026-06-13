"""Regression guards for markdown raw-HTML sanitizer helpers."""

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _norm(s: str) -> str:
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s)
    return s


def test_markdown_raw_html_sanitizer_checks_url_attr_edge_cases():
    src = (_REPO / "static" / "js" / "markdown.js").read_text(encoding="utf-8")
    haystack = _norm(src)
    needles = [
        "function _compactUrlSchemeValue(value)",
        "function _isDangerousUrl(value)",
        "function _isDangerousSrcset(value)",
        "'srcset'",
        "(candidate) => _isDangerousUrl(candidate)",
        "name === 'srcset' ? _isDangerousSrcset(attr.value) : _isDangerousUrl(attr.value)",
    ]
    for needle in needles:
        needle = _norm(needle)
        assert needle in haystack


def test_markdown_raw_html_sanitizer_strips_scriptable_css():
    src = (_REPO / "static" / "js" / "markdown.js").read_text(encoding="utf-8")
    haystack = _norm(src)
    needles = [
        "if (name === 'style')",
        r"javascript:|vbscript:|data:|expression\(",
        "el.removeAttribute(attr.name);",
    ]
    for needle in needles:
        needle = _norm(needle)
        assert needle in haystack
