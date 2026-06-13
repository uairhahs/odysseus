"""Regression guards for agent-tool screenshot DOM sinks."""

import re
from pathlib import Path

from tests.helpers.linter_compat import _norm

_REPO = Path(__file__).resolve().parent.parent


def test_live_tool_screenshot_does_not_template_raw_sse_value():
    chat = (_REPO / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    chat = _norm(chat)

    needles = [
        ("safeToolScreenshotSrc(json.screenshot)", True),
        ("img.src = screenshotSrc", True),
        (
            'details.innerHTML = `<summary>Screenshot</summary><img src="${json.screenshot}"',
            False,
        ),
    ]
    for needle, should_exist in needles:
        needle = _norm(needle)
        if should_exist:
            assert needle in chat
        else:
            assert needle not in chat


def test_restored_tool_screenshot_uses_raster_data_url_whitelist():
    renderer = (_REPO / "static" / "js" / "chatRenderer.js").read_text(encoding="utf-8")

    assert "export function safeToolScreenshotSrc(raw)" in renderer
    assert "(?:png|jpe?g|gif|webp)" in renderer
    assert "safeToolScreenshotSrc(ev.screenshot)" in renderer
    assert 'src="${esc(ev.screenshot)}"' not in renderer


def test_streaming_tool_labels_are_escaped_before_inner_html():
    chat = (_REPO / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    compare = (_REPO / "static" / "js" / "compare" / "stream.js").read_text(
        encoding="utf-8"
    )

    assert '<span class="agent-thread-tool">${esc(toolLabel)}</span>' in chat
    assert '<span class="agent-thread-tool">${toolLabel}</span>' not in chat
    assert '<span class="agent-thread-tool">${escapeHtml(toolLabel)}</span>' in compare
    assert '<span class="agent-thread-tool">${toolLabel}</span>' not in compare


def test_generated_image_urls_are_vetted_before_assignment_or_open():
    renderer = (_REPO / "static" / "js" / "chatRenderer.js").read_text(encoding="utf-8")
    compare = (_REPO / "static" / "js" / "compare" / "stream.js").read_text(
        encoding="utf-8"
    )
    group = (_REPO / "static" / "js" / "group.js").read_text(encoding="utf-8")

    assert "export function safeDisplayImageSrc(raw)" in renderer
    assert "safeDisplayImageSrc(imageUrl)" in renderer
    assert "img.src = safeImageUrl" in renderer
    # less brittle assertion
    assert re.search(
        r"window\.open\(safeImageUrl,\s*(['\"])_blank\1,\s*(['\"])noopener,noreferrer\2\)",
        renderer,
    )
    assert "safeDisplayImageSrc," in renderer
    assert "safeDisplayImageSrc(json.image_url)" in compare
    assert "img.src = json.image_url" not in compare
    assert "chatRenderer.safeDisplayImageSrc(json.url)" in group
    assert "img.src = json.url" not in group


def test_group_chat_role_labels_are_escaped_before_inner_html():
    group = (_REPO / "static" / "js" / "group.js").read_text(encoding="utf-8")

    assert '<div class="role">${uiModule.esc(roleLabel)}' in group
    assert '<div class="role">${roleLabel}' not in group


def test_main_chat_role_labels_are_escaped_before_inner_html():
    chat = _norm((_REPO / "static" / "js" / "chat.js").read_text(encoding="utf-8"))

    needles = [
        # Must exist: The safely escaped template literals
        ('<div class="role">${uiModule.esc(roleLabel)}', True),
        ('<div class="role">${uiModule.esc(agentModelLabel)}', True),
        # Must NOT exist: The dangerous, unescaped raw variables
        ('<div class="role">${roleLabel}', False),
        ('<div class="role">${agentModelLabel}', False),
        ("'<div class=\"role\">' + roleLabel", False),
    ]

    # Note: I removed the positive check for the old string concatenation:
    # ("'<div class=\"role\">' + uiModule.esc(roleLabel)", True)
    # If the test passes without it, it means the devs fully migrated to template literals.

    for needle, should_exist in needles:
        normed = _norm(needle)
        if should_exist:
            assert normed in chat, f"Expected safe escaped label {needle!r} in chat.js"
        else:
            assert (
                normed not in chat
            ), f"SECURITY FAIL: Found unescaped label {needle!r} in chat.js"


def test_compare_search_result_links_are_http_only():
    compare = (_REPO / "static" / "js" / "compare" / "stream.js").read_text(
        encoding="utf-8"
    )

    assert "function _safeHttpHref(raw)" in compare
    assert "const safeUrl = _safeHttpHref(r.url);" in compare
    assert "titleLink.href = safeUrl;" in compare
    assert "titleLink.href = r.url || '#';" not in compare


def test_compare_probe_provider_labels_are_escaped():
    selector = (_REPO / "static" / "js" / "compare" / "selector.js").read_text(
        encoding="utf-8"
    )

    assert "${escapeHtml(p.label || p.id)}" in selector
    assert "${p.label || p.id}" not in selector
