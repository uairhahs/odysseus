from pathlib import Path

from tests.helpers.linter_compat import _norm

APP_JS = Path("static/app.js")


def _slice(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_new_chat_prefers_pending_and_current_model_before_default():
    source = APP_JS.read_text(encoding="utf-8")
    helper = _slice(
        source,
        "async function _createDirectChatFromPreferredModel()",
        "// ============================================",
    )

    default_pos = helper.index("const dc = await _refreshDefaultChat();")
    assert helper.index("sessionModule.getPendingChat") < default_pos
    assert helper.index("current.endpoint_url") < default_pos
    assert default_pos < helper.index("const withModel = sessions.filter")


def test_desktop_new_chat_actions_use_shared_preference_helper():
    source = APP_JS.read_text(encoding="utf-8")

    rail_handler = _slice(
        source, "// New session button on icon rail", "// Mobile new chat button"
    )
    brand_handler = _slice(
        source,
        "// Logo click → new chat",
        'const docBtn2 = el("overflow-doc-btn");',
    )

    assert _norm("if (await _createDirectChatFromPreferredModel()) return;") in _norm(
        rail_handler
    )
    assert _norm("if (await _createDirectChatFromPreferredModel()) return;") in _norm(
        brand_handler
    )
    assert _norm("const dc = await _refreshDefaultChat();") not in _norm(rail_handler)
    assert _norm("const dc = await _refreshDefaultChat();") not in _norm(brand_handler)
