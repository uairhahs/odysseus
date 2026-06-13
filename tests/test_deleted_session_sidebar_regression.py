from pathlib import Path

from tests.helpers.linter_compat import _norm

APP_JS = Path("static/app.js")
SESSIONS_JS = Path("static/js/sessions.js")


def test_rail_delete_uses_hard_delete_endpoint():
    source = _norm(APP_JS.read_text())
    rail_block = source[source.index("const railDelete = el('rail-delete-session');") :]
    haystack = rail_block[: rail_block.index("// Textarea auto-resize")]

    needles = [
        # Check the endpoint URL specifically
        ("fetch(`${API_BASE}/api/session/${currentId}`", True),
        # Check that the method is DELETE
        ("method: 'DELETE'", True),
        # Ensure we aren't using the old archive endpoint
        ("api/session/${currentId}/archive", False),
    ]

    for needle, should_exist in needles:
        needle = _norm(needle)
        if should_exist:
            assert (
                needle in haystack
            ), f"expected to find {needle!r} in rail delete block"
        else:
            assert (
                needle not in haystack
            ), f"unexpectedly found {needle!r} in rail delete block"


def test_deleted_sessions_are_pruned_from_local_sidebar_state():
    source = _norm(SESSIONS_JS.read_text())

    assert "function _removeSessionFromLocalState(sid)" in source
    assert "sessions = sessions.filter(s => String(s.id) !== id);" in source
    assert (
        "Storage.set('session-order', JSON.stringify(orderIds.filter(x => String(x) !== id)))"
        in source
    )
    assert "_removeSessionFromLocalState(s.id);" in source


def test_session_fetch_normalizes_duplicate_ids_before_render():
    source = _norm(SESSIONS_JS.read_text())

    assert "function _normalizeSessionsList(fetched)" in source
    assert "if (seen.has(id)) continue;" in source
    assert "sessions = _normalizeSessionsList(fetched);" in source
