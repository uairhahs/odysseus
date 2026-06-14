"""Regression guards for same-host Cookbook SSH server profiles (#3337)."""

from pathlib import Path

from tests.helpers.linter_compat import _norm

ROOT = Path(__file__).resolve().parent.parent


# Read and normalize the sources at module initialization
COOKBOOK = _norm((ROOT / "static/js/cookbook.js").read_text(encoding="utf-8"))
HWFIT = _norm((ROOT / "static/js/cookbook-hwfit.js").read_text(encoding="utf-8"))
DOWNLOAD = _norm((ROOT / "static/js/cookbookDownload.js").read_text(encoding="utf-8"))
SERVE = _norm((ROOT / "static/js/cookbookServe.js").read_text(encoding="utf-8"))
RUNNING = _norm((ROOT / "static/js/cookbookRunning.js").read_text(encoding="utf-8"))


def test_server_dropdown_options_use_profile_keys_not_hosts():
    needles = [
        ("remoteServerKey", True),
        ("export function _serverKey(s)", True),
        ("s?.name || ''", True),
        ("s?.host || ''", True),
        ("s?.port || ''", True),
        ("s?.envPath || ''", True),
        ("const value = _serverKey(s);", True),
        ("option value='${esc(s.host)}'", False),
    ]
    for needle, should_exist in needles:
        normed = _norm(needle)
        if should_exist:
            assert normed in COOKBOOK, f"Expected to find {needle!r} in cookbook.js"
        else:
            assert (
                normed not in COOKBOOK
            ), f"Unexpectedly found {needle!r} in cookbook.js"


def test_selected_server_helpers_prefer_profile_key_before_host_fallback():
    needles = [
        "_envState.remoteServerKey = _serverKey(s);",
        "const selected = hostOrTask === _envState.remoteHost ? _selectedServer() : null;",
        "const srv = selected || _serverByVal(hostOrTask);",
        "const _want = _currentServerValue();",
    ]
    for needle in needles:
        normed = _norm(needle)
        assert normed in COOKBOOK, f"Expected to find {needle!r} in cookbook.js"


def test_cookbook_submodules_resolve_visible_profile_selection():
    # Grouping needles by the specific file they belong to
    file_checks = [
        (
            DOWNLOAD,
            "cookbookDownload.js",
            [
                "_serverByVal?.(_ssv)",
                "_serverByVal?.(_envState.remoteServerKey || host)",
                "_serverByVal?.(_envState.remoteServerKey || _zh)",
            ],
        ),
        (
            HWFIT,
            "cookbook-hwfit.js",
            [
                "_serverByVal(_envState.remoteServerKey || remoteHost)",
                "hk: _currentServerValue()",
                "sel.value = _currentServerValue();",
            ],
        ),
        (
            SERVE,
            "cookbookServe.js",
            [
                "_serverByVal?.(_ssEl.value)",
                "_serverByVal?.(val)",
                "_serverByVal?.(_es.remoteServerKey || _es.remoteHost || '')",
                "_serverByVal?.(_envState.remoteServerKey || _probeHost)",
            ],
        ),
    ]

    for haystack, filename, needles in file_checks:
        for needle in needles:
            normed = _norm(needle)
            assert normed in haystack, f"Expected to find {needle!r} in {filename}"


def test_running_tab_resolves_profile_key_not_first_host():
    needles = [
        "_serverByVal(_envState.remoteServerKey || _tHost)",
        "_serverByVal(_envState.remoteServerKey || _host)",
        "_serverByVal(_envState.remoteServerKey || host)",
        "_serverByVal = shared._serverByVal;",
        "_selectedServer = shared._selectedServer;",
    ]
    for needle in needles:
        normed = _norm(needle)
        assert normed in RUNNING, f"Expected to find {needle!r} in cookbookRunning.js"


def test_no_same_host_selector_paths_resolve_by_first_matching_host():
    forbidden_needles = [
        "servers.find(s => s.host === select.value)",
        "servers.find(s => s.host === _ssEl.value)",
        "servers.find(x => x.host === val)",
        "servers.find(s => s.host === _ssv)",
    ]
    combined_haystack = "\n".join([DOWNLOAD, HWFIT, SERVE])

    for needle in forbidden_needles:
        normed = _norm(needle)
        assert (
            normed not in combined_haystack
        ), f"Unexpectedly found forbidden fallback {needle!r}"
