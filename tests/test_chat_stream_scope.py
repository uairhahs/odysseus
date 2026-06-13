from pathlib import Path

from tests.helpers.linter_compat import _norm


def test_stream_render_helpers_are_visible_to_catch_block():
    # Read AND normalize the entire source file
    source = _norm(Path("static/js/chat.js").read_text(encoding="utf-8"))

    # Normalize your markers so strict spaces and newlines are ignored
    marker_try = _norm("try { // Re-enable auto-scroll")
    marker_catch = _norm("} catch (err) {")

    # Slice using the normalized string
    try_start = source.index(marker_try)
    catch_start = source.index(marker_catch, try_start)

    outer_scope = source[:try_start]
    try_body = source[try_start:catch_start]

    outer_needles = [
        "let _renderStream = () => {};",
        "let _cancelThinkingTimer = () => {};",
        "let _removeThinkingSpinner = () => {};",
    ]
    for needle in outer_needles:
        assert (
            _norm(needle) in outer_scope
        ), f"Missing declaration {needle!r} in outer scope"

    try_needles = [
        ("_renderStream = () => {", True),
        ("_cancelThinkingTimer = () => {", True),
        ("_removeThinkingSpinner = () => {", True),
        (  # Ensure we aren't using function declarations
            "function _renderStream()",
            False,
        ),
    ]
    for needle, should_exist in try_needles:
        normed = _norm(needle)
        if should_exist:
            assert normed in try_body, f"Missing assignment {needle!r} in try block"
        else:
            assert normed not in try_body, f"Found forbidden {needle!r} in try block"
