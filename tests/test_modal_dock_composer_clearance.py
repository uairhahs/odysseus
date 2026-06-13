import re
from pathlib import Path

CSS = Path("static/style.css").read_text(encoding="utf-8")
INIT_JS = Path("static/js/init.js").read_text(encoding="utf-8")


def _get_css_block(selector: str, css: str) -> str:
    """Extracts the CSS block for a given selector."""
    # Matches selector { ... }
    pattern = rf"{re.escape(selector)}\s*\{{([^}}]+)\}}"
    match = re.search(pattern, css)
    if not match:
        raise ValueError(f"Selector {selector} not found in CSS")
    return match.group(1)


def test_both_minimized_window_docks_clear_the_composer():
    # Verify the blocks exist
    min_dock = _get_css_block("#minimized-dock", CSS)
    modal_dock = _get_css_block("#modal-dock", CSS)

    # Check for the specific properties inside those blocks
    # Using regex to ignore spacing differences
    assert re.search(r"bottom\s*:\s*var\(--composer-clearance,\s*12px\)", min_dock)
    assert re.search(
        r"bottom\s*:\s*var\(--composer-clearance,\s*0(?:px)?\)", modal_dock
    )


def test_composer_clearance_tracks_input_and_attachment_height():
    # Regex search for the lines, allowing for variable spacing
    assert re.search(
        r"const\s+chatBar\s*=\s*document\.querySelector\('\.chat-input-bar'\)", INIT_JS
    )
    assert re.search(
        r"const\s+attachStrip\s*=\s*document\.getElementById\('attach-strip'\)", INIT_JS
    )
    assert re.search(
        r"root\.style\.setProperty\('--composer-clearance',\s*clearance\s*\+\s*'px'\)",
        INIT_JS,
    )
