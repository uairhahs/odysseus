import re
from pathlib import Path

STYLE_CSS = Path(__file__).resolve().parents[1] / "static" / "style.css"
css = STYLE_CSS.read_text(encoding="utf-8")


def _norm(s: str) -> str:
    """Normalize whitespace and quote style so cosmetic differences don't matter."""
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def test_native_select_options_use_theme_tokens():
    needles = [
        "--select-option-bg:",
        "--select-option-fg:",
        "--select-option-active-bg:",
        "select option,\n    select optgroup",
        "background-color: var(--select-option-bg);",
        "color: var(--select-option-fg);",
        "select option:checked",
        "background-color: var(--select-option-active-bg);",
    ]
    haystack = _norm(css)
    for needle in needles:
        assert _norm(needle) in haystack, f"Expected to find '{needle}' in style.css"


def test_light_theme_keeps_native_selects_light():

    light_theme_start = css.index(":root.light {")
    light_theme_end = css.index("}", light_theme_start)
    light_theme_block = css[light_theme_start:light_theme_end]
    needles = [
        ("--select-bg: #eaeaea;", light_theme_block),
        ("--select-option-bg: var(--panel);", light_theme_block),
        (":root.light select { color-scheme: light; }", css),
    ]
    for needle, haystack in needles:
        assert _norm(needle) in _norm(
            haystack
        ), f"Expected to find '{needle}' in style.css"
