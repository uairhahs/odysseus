# src/active_document.py

# Shared by tool_implementations.py and document_tools.py so they can both import it without circular imports.

from typing import Optional

_active_document_id: Optional[str] = None


def set_active_document(doc_id: Optional[str]):
    global _active_document_id
    _active_document_id = doc_id


def get_active_document() -> Optional[str]:
    return _active_document_id


def clear_active_document(doc_id: Optional[str] = None) -> bool:
    """Clear the in-memory active-document pointer.

    With ``doc_id`` given, only clears when it matches the current pointer, so a
    different active document is left untouched. Returns True if it was cleared.

    Called when a document is detached from its session or deleted (its tab is
    closed): without this, the stale pointer makes the last-resort doc-injection
    path re-surface a closed document in a later, unrelated chat — even one whose
    session no longer matches — because an unlinked doc has session_id NULL (#1160).
    """
    global _active_document_id
    if doc_id is None or _active_document_id == doc_id:
        _active_document_id = None
        return True
    return False
