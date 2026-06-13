"""Static regressions for `/setup` account sign-in providers."""

from pathlib import Path

from tests.helpers.linter_compat import _norm

_REPO = Path(__file__).resolve().parent.parent
_SLASH = (_REPO / "static" / "js" / "slashCommands.js").read_text(encoding="utf-8")


def _between(src: str, start: str, end: str) -> str:
    start_idx = src.index(start)
    end_idx = src.index(end, start_idx)
    return src[start_idx:end_idx]


def test_setup_guide_lists_account_sign_in_providers():
    # unused ...
    # guide_block = _between(_SLASH, "function _showSetupEndpointChoices", "async function _hasConfiguredModels")
    haystack = _norm(_SLASH)
    needles = [
        'data-setup-provider="',
        "provider.key",
        "'copilot'",
        "'chatgpt-subscription'",
        "/setup copilot",
        "/setup chatgpt-subscription",
    ]
    for needle in needles:
        needle = _norm(needle)
        assert needle in haystack


def test_clicking_account_sign_in_provider_prefills_setup_command_not_api_key():
    click_block = _between(
        _SLASH,
        'const providerEl = e.target.closest(".setup-clickable-provider")',
        "// 3. Check",
    )
    haystack = _norm(click_block)
    needles = [
        "providerEl.dataset.setupProvider",
        "providerEl.dataset.setupKind === 'device-auth'",
        "'/setup ' + providerKey",
    ]
    for needle in needles:
        needle = _norm(needle)
        assert needle in haystack


def test_setup_chatgpt_subscription_prints_auth_url_without_auto_opening_tab():
    flow_block = _between(
        _SLASH, "async function _setupProviderDeviceFlow", "async function _cmdSetup"
    )
    flow_block = _norm(flow_block)

    needles = [
        "providerKey === 'chatgpt-subscription'",
        "Open this URL",
        "authUrl",
        "href=\"' + uiModule.esc(authUrl || '') + '\"",
        "if (providerKey === 'chatgpt-subscription') return;",
    ]
    for needle in needles:
        needle = _norm(needle)
        assert needle in flow_block
