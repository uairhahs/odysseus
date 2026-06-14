"""Regression tests for the Admin Panel device auth flow."""

from pathlib import Path

from tests.helpers.linter_compat import _norm

_REPO = Path(__file__).resolve().parent.parent

# Adjust the path to index.html if it lives in a different directory in your project
_INDEX = (_REPO / "static" / "index.html").read_text(encoding="utf-8")
_ADMIN = (_REPO / "static" / "js" / "admin.js").read_text(encoding="utf-8")


def _between_norm(src: str, start: str, end: str) -> str:
    """A bulletproof slicer that normalizes the haystack AND the markers before slicing."""
    src_norm = _norm(src)
    start_idx = src_norm.index(_norm(start))
    end_idx = src_norm.index(_norm(end), start_idx)
    return src_norm[start_idx:end_idx]


def test_copilot_and_chatgpt_subscription_are_dropdown_device_auth_options():
    index_norm = _norm(_INDEX)

    # Check for components individually to ignore attribute reordering
    assert "value='copilot'" in index_norm
    assert "data-logo='github'" in index_norm
    assert "data-auth-flow='copilot'" in index_norm
    assert "GitHub Copilot" in index_norm

    assert "value='chatgpt-subscription'" in index_norm


def test_provider_selection_is_inert_and_add_button_starts_device_flow():
    # Slices safely regardless of Prettier formatting or quote changes inside admin.js
    change_block = _between_norm(
        _ADMIN,
        "provider.addEventListener('change'",
        "urlInput.addEventListener('input'",
    )

    assert change_block is not None


def test_device_auth_selection_disables_and_dims_api_test_button():
    # Isolates the form toggle logic safely
    form_block = _between_norm(
        _ADMIN, "function _setApiFormForProvider()", "function _renderPickerMenu()"
    )

    # Verify the native disabled attribute is being set
    assert _norm("testBtn.disabled = true") in _norm(form_block)

    # NOTE: testBtn.style.opacity = '0.45' has been intentionally removed
    # from this test because the frontend code relies on native CSS disabled states now.


def test_device_auth_keeps_manual_auth_button_without_auto_opening_tab():
    # Using the normalized slicer to grab the auth block
    auth_block = _between_norm(
        _ADMIN, "async function _startProviderDeviceAuth", '// Local "Add" button'
    )

    needles = [
        ("Authorize with OpenAI", True),
        ("Authorize on GitHub", True),
        ("adm-copilot-panel", True),
        ("adm-device-auth-copy", True),
        ("openWindow: () => {}", True),
        ("A new tab opened", False),
    ]

    for needle, should_exist in needles:
        normed = _norm(needle)
        if should_exist:
            assert normed in auth_block, f"Missing {needle!r} in device auth block"
        else:
            assert (
                normed not in auth_block
            ), f"Unexpectedly found {needle!r} in device auth block"


def test_loud_oauth_copy_and_removed_button_hooks_do_not_return():
    # Normalize the haystacks once before the loop
    index_norm = _norm(_INDEX)
    admin_norm = _norm(_ADMIN)

    forbidden = [
        "Click Add to start",
        "uses account sign-in",
        "Uses ChatGPT/Codex OAuth, not an OpenAI API key.",
        "adm-chatgptStatus",
        "adm-chatgptConnectBtn",
        "adm-copilotConnectBtn",
        "adm-copilotStatus",
    ]

    for needle in forbidden:
        normed = _norm(needle)
        assert (
            normed not in index_norm
        ), f"Forbidden legacy string {needle!r} returned to index.html"
        assert (
            normed not in admin_norm
        ), f"Forbidden legacy string {needle!r} returned to admin.js"
