"""Model-assisted route helpers must resolve endpoints with owner scope."""

import ast
import re
from pathlib import Path


def _norm(s: str) -> str:
    """Normalize whitespace and quote style so cosmetic differences don't matter."""
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def _function_source(path: str, name: str) -> str:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def test_document_ai_tidy_resolves_with_owner_scope():
    body = _function_source("routes/document_routes.py", "ai_tidy_documents")
    needles = [
        "resolve_task_endpoint(owner=user or None)",
        'resolve_endpoint("default", owner=user or None)',
    ]
    for needle in needles:
        assert _norm(needle) in _norm(body)


def test_calendar_quick_parse_resolves_with_owner_scope():
    body = _function_source("routes/calendar_routes.py", "quick_parse")
    assert "owner = _require_user(request)" in body
    needles = [
        'resolve_endpoint("utility", owner=owner or None)',
        'resolve_endpoint("default", owner=owner or None)',
    ]
    for needle in needles:
        assert _norm(needle) in _norm(body)


def test_task_parse_resolves_with_owner_scope():
    body = _function_source("routes/task_routes.py", "parse_task")
    needles = [
        "user = _owner(request)",
        'resolve_endpoint("utility", owner=user or None)',
        'resolve_endpoint("default", owner=user or None)',
    ]
    for needle in needles:
        assert _norm(needle) in _norm(body)


def test_history_compact_resolves_with_owner_scope():
    body = _function_source("routes/history_routes.py", "compact_session")
    needles = [
        "owner = _require_user(request)",
        'resolve_endpoint("utility", owner=owner or None)',
        'resolve_endpoint("default", owner=owner or None)',
    ]
    for needle in needles:
        assert _norm(needle) in _norm(body)


def test_note_reminder_synthesis_resolves_with_owner_scope():
    body = _function_source("routes/note_routes.py", "dispatch_reminder")
    needles = [
        'resolve_endpoint("utility", owner=owner or None)',
        'resolve_endpoint("default", owner=owner or None)',
    ]
    for needle in needles:
        assert _norm(needle) in _norm(body)
