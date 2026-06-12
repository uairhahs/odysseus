import ast
import re
from pathlib import Path

CHAT_ROUTES = Path(__file__).resolve().parents[1] / "routes" / "chat_routes.py"


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
