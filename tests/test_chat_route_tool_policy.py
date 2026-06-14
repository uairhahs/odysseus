import ast
import re

"""Issue #3229 — allow_bash / allow_web_search must work for JSON API callers
and admin users must get bash enabled by default.

Bug: allow_bash and allow_web_search were only read from form_data, so JSON
API callers (Content-Type: application/json) always had bash disabled.

Fix: (1) Read from JSON body as fallback.
     (2) Only add bash/web_search to disabled_tools when explicitly set to a
         falsy value; when unset (None), defer to per-user privilege checks.
"""

from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


# ── Source-level guards ─────────────────────────────────────────


def test_allow_bash_reads_from_body_as_fallback():
    """chat_stream must read allow_bash from the JSON body, not just form_data."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the chat_stream function
    chat_stream_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_stream":
            chat_stream_func = node
            break
    assert chat_stream_func is not None, "chat_stream function not found"

    # Look for an assignment to allow_bash that references 'body'
    found_body_fallback = False
    for node in ast.walk(chat_stream_func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "allow_bash":
                    # Check if 'body' appears in the value
                    src_segment = ast.get_source_segment(source, node)
                    if src_segment and "body" in src_segment:
                        found_body_fallback = True
    assert (
        found_body_fallback
    ), "allow_bash assignment in chat_stream must fall back to JSON body"


def test_allow_web_search_reads_from_body_as_fallback():
    """chat_stream must read allow_web_search from the JSON body, not just form_data."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    chat_stream_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_stream":
            chat_stream_func = node
            break
    assert chat_stream_func is not None

    found_body_fallback = False
    for node in ast.walk(chat_stream_func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "allow_web_search":
                    src_segment = ast.get_source_segment(source, node)
                    if src_segment and "body" in src_segment:
                        found_body_fallback = True
    assert (
        found_body_fallback
    ), "allow_web_search assignment in chat_stream must fall back to JSON body"


def test_disabled_tools_does_not_bash_when_allow_bash_is_none():
    """When allow_bash is not set (None), bash must NOT be unconditionally
    added to disabled_tools.  The per-user privilege check handles it.
    """
    source = _CHAT_ROUTES.read_text(encoding="utf-8")

    # The fix changes:
    #   if str(allow_bash).lower() != "true":
    # to:
    #   if allow_bash is not None and str(allow_bash).lower() != "true":
    assert (
        "allow_bash is not None" in source
    ), "disabled_tools check must guard against allow_bash being None"
    assert (
        "allow_web_search is not None" in source
    ), "disabled_tools check must guard against allow_web_search being None"


# ── Functional tests of the disabled-tools logic ───────────────


def _build_disabled_tools(
    allow_bash=None,
    allow_web_search=None,
    can_use_bash=True,
    can_use_browser=True,
):
    """Replicate the disabled-tools logic from chat_stream for unit testing.

    Returns the set of tool names that would be disabled.
    """
    disabled_tools = set()

    # Issue #3229 fix: only disable when explicitly set to a falsy value.
    if allow_bash is not None and str(allow_bash).lower() != "true":
        disabled_tools.add("bash")
    if allow_web_search is not None and str(allow_web_search).lower() != "true":
        disabled_tools.add("web_search")
        disabled_tools.add("web_fetch")

    # Enforce per-user privileges
    if not can_use_bash:
        disabled_tools.update({"bash", "python", "read_file", "write_file"})
    if not can_use_browser:
        disabled_tools.add("builtin_browser")

    return disabled_tools


def test_json_body_allow_bash_true_enables_bash():
    """API caller sending {"allow_bash": true} gets bash enabled."""
    disabled = _build_disabled_tools(allow_bash="true")
    assert "bash" not in disabled


def test_json_body_allow_bash_false_disables_bash():
    """API caller sending {"allow_bash": false} gets bash disabled."""
    disabled = _build_disabled_tools(allow_bash="false")
    assert "bash" in disabled


def test_json_body_allow_web_search_true_enables_web():
    """API caller sending {"allow_web_search": true} gets web tools enabled."""
    disabled = _build_disabled_tools(allow_web_search="true")
    assert "web_search" not in disabled
    assert "web_fetch" not in disabled


def test_json_body_allow_web_search_false_disables_web():
    """API caller sending {"allow_web_search": false} gets web tools disabled."""
    disabled = _build_disabled_tools(allow_web_search="false")
    assert "web_search" in disabled
    assert "web_fetch" in disabled


def test_admin_user_gets_bash_enabled_by_default():
    """When allow_bash is not set and user has can_use_bash privilege,
    bash must NOT be disabled.
    """
    disabled = _build_disabled_tools(allow_bash=None, can_use_bash=True)
    assert "bash" not in disabled


def test_admin_user_gets_web_search_enabled_by_default():
    """When allow_web_search is not set and user has normal privileges,
    web_search must NOT be disabled.
    """
    disabled = _build_disabled_tools(allow_web_search=None)
    assert "web_search" not in disabled
    assert "web_fetch" not in disabled


def test_non_privileged_user_without_explicit_flag_still_disabled():
    """A user without can_use_bash privilege who doesn't send allow_bash
    should still have bash disabled via the privilege check.
    """
    disabled = _build_disabled_tools(allow_bash=None, can_use_bash=False)
    assert "bash" in disabled


def test_non_privileged_user_explicit_true_overridden_by_privilege():
    """Even if allow_bash=true is sent, a user without can_use_bash
    privilege still gets bash disabled by the privilege gate.
    """
    disabled = _build_disabled_tools(allow_bash="true", can_use_bash=False)
    assert "bash" in disabled


def test_form_data_none_body_true_works():
    """Simulates: form_data has no allow_bash, body has allow_bash=true.
    After the fallback (`form_data.get(...) or body.get(...)`), allow_bash
    should be "true".
    """
    # Simulate the fallback logic
    form_data_val = None  # not in form_data
    body_val = "true"  # from JSON body
    allow_bash = form_data_val or body_val
    assert str(allow_bash).lower() == "true"

    disabled = _build_disabled_tools(allow_bash=allow_bash)
    assert "bash" not in disabled


def test_explicit_false_disables_even_for_admin():
    """An admin who explicitly sends allow_bash=false should have bash disabled."""
    disabled = _build_disabled_tools(
        allow_bash="false",
        can_use_bash=True,
    )
    assert "bash" in disabled


# ── Frontend source-level guards ──────────────────────────────

_CHAT_JS = Path(__file__).resolve().parent.parent / "static" / "js" / "chat.js"


def test_frontend_always_sends_explicit_allow_bash():
    """chat.js must always send allow_bash (both true and false), not only on toggle ON."""
    source = _CHAT_JS.read_text(encoding="utf-8")
    # Must not only append 'true' — must also handle the false case
    assert (
        "allow_bash', el('bash-toggle').checked ? 'true' : 'false'" in source
        or "allow_bash', 'false'" in source
    ), "Frontend must send explicit allow_bash=false when toggle is off"


def test_frontend_sends_explicit_allow_web_search_false_in_agent_mode():
    """chat.js must send allow_web_search=false when web toggle is off in agent mode."""
    source = _CHAT_JS.read_text(encoding="utf-8")
    assert (
        "allow_web_search', 'false'" in source
    ), "Frontend must send explicit allow_web_search=false in agent mode when toggle is off"


def _source() -> str:
    return CHAT_ROUTES.read_text(encoding="utf-8")


def _normalise(src: str) -> str:
    src = re.sub(r"#[^\n]*", "", src)
    src = re.sub(r"'([^']*)'", r'"\1"', src)
    src = re.sub(r"\s+", " ", src)
    src = re.sub(r"\(\s+", "(", src)  # remove space after (
    src = re.sub(r"\s+\)", ")", src)  # remove space before )
    return src


def _fn_body(src: str, fn_name: str) -> str:
    """Extract the source of a top-level async def by name."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == fn_name
        ):
            lines = src.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Function {fn_name!r} not found")


def _assert_order(src: str, *patterns: str) -> None:
    """Assert that all patterns appear in src in the given order."""
    norm = _normalise(src)
    search_from = 0
    prev_pat = None
    for p in patterns:
        np = _normalise(p)
        idx = norm.find(np, search_from)
        assert idx != -1, (
            f"Pattern not found in source (searching from offset {search_from}):\n  {p!r}"
            + (f"\n  (after: {prev_pat!r})" if prev_pat else "")
        )
        search_from = idx + len(np)
        prev_pat = p


def _assert_present(src: str, *patterns: str) -> None:
    norm = _normalise(src)
    for p in patterns:
        np = _normalise(p)
        assert np in norm, f"Pattern not found in source:\n  {p!r}"


# ── tests ──────────────────────────────────────────────────────────────────────


def test_research_fast_path_respects_tool_policy():
    src = _source()
    _assert_present(src, '"is_research": effective_do_research')
    _assert_order(
        src,
        "pre_context_tool_policy = build_effective_tool_policy(",
        "allow_tool_preprocessing = not pre_context_tool_policy.block_all_tool_calls",
        "allow_tool_preprocessing=allow_tool_preprocessing",
        "research_blocked_by_policy = bool(",
        'tool_policy.blocks("trigger_research")',
        'tool_policy.blocks("manage_research")',
        "effective_do_research = bool(",
        "_effective_mode = 'research' if effective_do_research else",
        "if effective_do_research:",
        "do_research=effective_do_research",
    )


def test_non_streaming_chat_path_uses_tool_policy_before_context_and_research():
    src = _fn_body(_source(), "chat_endpoint")
    _assert_present(
        src,
        'tool_policy.blocks("trigger_research")',
        'tool_policy.blocks("manage_research")',
    )
    _assert_order(
        src,
        "tool_policy = build_effective_tool_policy(last_user_message=message)",
        "allow_tool_preprocessing = not tool_policy.block_all_tool_calls",
        'if not tool_policy.blocks("manage_memory"):',
        "allow_tool_preprocessing=allow_tool_preprocessing",
        "research_blocked_by_policy = tool_policy.blocks(",
        "if use_research and not research_blocked_by_policy:",
        "allow_background_extraction=not tool_policy.block_all_tool_calls",
    )


def test_image_generation_fast_path_checks_policy_before_tool_start():
    src = _source()
    _assert_order(
        src,
        'if tool_policy.blocks("generate_image"):',
        '"type": "tool_start", "tool": "generate_image"',
        "do_generate_image(",
    )


def test_streaming_chat_paths_disable_background_extraction_under_policy():
    src = _source()
    norm = _normalise(src)
    pattern = _normalise(
        "allow_background_extraction=not tool_policy.block_all_tool_calls"
    )
    count = norm.count(pattern)
    assert count >= 3, f"Expected >= 3 occurrences, found {count}"
