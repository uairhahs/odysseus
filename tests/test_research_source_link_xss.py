"""Regression guards for API-provided research source hrefs."""

import re
from pathlib import Path


def _norm(s: str) -> str:
    """Normalize whitespace and quote style so cosmetic differences don't matter."""
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


_REPO = Path(__file__).resolve().parent.parent


def test_document_library_research_preview_whitelists_source_hrefs():
    src = (_REPO / "static" / "js" / "documentLibrary.js").read_text(encoding="utf-8")
    haystack = _norm(src)

    needles = [
        ("function _safeResearchHref(raw)", True),
        ("parsed.protocol === 'http:' || parsed.protocol === 'https:'", True),
        ("const url = _safeResearchHref(src.url);", True),
        ('href="${_esc(url)}"', False),
        ("Failed to load: ${_esc(e.message)}", True),
        ("Failed to load: ${e.message}", False),
    ]
    for needle, should_exist in needles:
        needle = _norm(needle)
        if should_exist:
            assert needle in haystack
        else:
            assert needle not in haystack


def test_research_panel_whitelists_source_hrefs():
    src = (_REPO / "static" / "js" / "research" / "panel.js").read_text(
        encoding="utf-8"
    )
    haystack = _norm(src)

    assert "function _safeSourceHref(raw)" in haystack
    assert "parsed.protocol === 'http:' || parsed.protocol === 'https:'" in haystack
    assert "const url = _safeSourceHref(s.url);" in haystack
    assert "const url = _esc(s.url || '');" not in haystack
