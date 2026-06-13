import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _normalize(source: str) -> str:
    """Collapse whitespace and normalize quote characters so assertions
    survive reformatting (line-wrapping, single vs double quotes, etc.)
    without changing the meaning of the code being checked."""
    collapsed = " ".join(source.split())
    return collapsed.replace('"', "'")


def test_backend_status_treats_download_exit_zero_as_completed():
    source = _normalize(_read("routes/cookbook_helpers.py"))

    # Exit-sentinel regex must look for the runner's
    # "=== process exited with code N ===" marker (case-insensitive).
    assert re.search(
        r"re\.search\(\s*r'=== process exited with code\\s\+\(-\?\\d\+\)'",
        source,
    ), "expected an exit-code sentinel regex"

    # Download tasks must resolve exit code 0 -> 'completed', non-zero -> 'error'.
    assert "task_type == 'download'" in source
    assert "'completed' if exit_code == 0 else 'error'" in source


def test_background_status_poll_reconciles_into_local_tasks():
    source = _normalize(_read("static/js/cookbookRunning.js"))

    # Build a lookup of backend-reported statuses keyed by session id.
    assert re.search(
        r"new Map\(tasks\.map\(\(?t\)?\s*=>\s*\[t\.session_id,\s*t\]\)\)",
        source,
    ), "expected a statusById map keyed by session_id"

    # 'completed' -> done, 'error' -> error in the reconciled status.
    assert "live.status === 'completed'" in source
    assert "'done'" in source
    assert "live.status === 'error'" in source
    assert "'error'" in source

    # Changed tasks get persisted and dependency installs get refreshed.
    assert "_saveTasks(localTasks)" in source
    assert re.search(
        r"completedDeps\.forEach\(\(?t\)?\s*=>\s*_refreshDepsAfterInstall\(t\)\)",
        source,
    ), "expected completedDeps to be refreshed after reconciliation"


def test_local_windows_session_commands_use_local_powershell_log_dir():
    source = _normalize(_read("static/js/cookbookRunning.js"))

    assert "const host = task.remoteHost;" in source

    # Local (non-remote) Windows sessions use $env:TEMP\odysseus-tmux;
    # remote sessions use $env:TEMP\odysseus-sessions.
    assert re.search(
        r"host\s*\?\s*'\$env:TEMP\\\\odysseus-sessions'\s*:\s*'\$env:TEMP\\\\odysseus-tmux'",
        source,
    ), "expected a host-dependent session dir ternary"

    # Remote commands go over ssh; local commands invoke powershell directly.
    assert re.search(r"host\s*\?\s*`ssh \$\{pf\}\$\{host\}", source)
    assert re.search(r":\s*`powershell -Command \"\$\{ps\}\"`", source) or re.search(
        r":\s*`powershell -Command '\$\{ps\}'`", source
    )


def test_dep_install_success_recognized_from_exit_sentinel():
    """A pip dependency install reports success via the runner's exit-0
    sentinel / pip's "Successfully installed" line, not the HuggingFace
    download markers. The shared helper must key off those, so an install
    whose tmux pane is gone isn't misread as crashed."""
    source = _normalize(_read("static/js/cookbookRunning.js"))

    assert "function _depInstallSucceeded(output) {" in source
    assert "=== Process exited with code" in source
    assert "Successfully installed" in source


def test_session_gone_heuristic_honors_dep_install_success():
    """The reconnect loop's session-gone branch (download tasks need an HF
    marker to look successful) must also accept a finished dependency install,
    otherwise a clean pip install with no HF markers is marked crashed."""
    source = _normalize(_read("static/js/cookbookRunning.js"))

    assert re.search(
        r"depInstallSucceeded\s*=\s*!!task\.payload\?\._dep\s*&&\s*_depInstallSucceeded\(lastOutput\)",
        source,
    ), "expected depInstallSucceeded derived from lastOutput"

    # A finished dependency install short-circuits looksSuccessful ahead of
    # the download/serve branch.
    assert re.search(
        r"looksSuccessful\s*=\s*depInstallSucceeded\s*\|\|\s*\(task\.type === 'download'",
        source,
    ), "expected depInstallSucceeded to gate looksSuccessful"


def test_background_poll_recovers_done_for_stopped_dependency_install():
    """When the backend reports a finished dependency install as "stopped"
    (its pip package is never in the HF cache the dead-session check inspects),
    the reconciler must recover "done" from the retained output instead of
    downgrading the card to crashed."""
    source = _normalize(_read("static/js/cookbookRunning.js"))

    assert re.search(
        r"depDone\s*=\s*!!task\.payload\?\._dep\s*&&\s*_depInstallSucceeded\(task\.output\)",
        source,
    ), "expected depDone derived from task.output"

    # depDone -> done; otherwise downloads crash and other tasks stop.
    assert re.search(r"depDone\s*\?\s*'done'", source)
    assert re.search(
        r"task\.type === 'download'\s*\?\s*'crashed'\s*:\s*'stopped'", source
    )


def test_dependency_install_payload_keeps_env_path_for_refresh():
    source = _normalize(_read("static/js/cookbook.js"))

    assert re.search(r"env_path:\s*_envState\.envPath\s*\|\|\s*''", source)


def test_local_dependency_probe_refreshes_user_site_visibility():
    source = _read("routes/shell_routes.py")

    assert "importlib.invalidate_caches()" in source
    assert "user_site = site.getusersitepackages()" in source
    assert (
        "if user_site and os.path.isdir(user_site) and user_site not in sys.path:"
        in source
    )
