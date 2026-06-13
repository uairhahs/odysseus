"""Regression guard for issue #1428 — the "x" on a chat output did nothing when
no model/API was selected.

deleteMessage() bailed at `if (!sessionId) return;`. An output shown before a
model is picked has no session and no persisted rows, so the early-out meant the
"x" never even removed the bubble from the DOM. The delete now falls through to
DOM removal when there's no session / no DB ids.

chat.js pulls in browser globals so it can't run under node; guard at the source.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "static/js/chat.js"


def _delete_message_body() -> str:
    text = SRC.read_text(encoding="utf-8")
    start = text.index("export async function deleteMessage(")
    rest = text[start:]

    try:
        # Find the very next top-level export (skipping the first character
        # so we don't match the start of deleteMessage itself).
        end = rest.index("\nexport ", 1)
        return rest[:end]
    except ValueError:
        # Fallback only if this is literally the last thing in the file
        return rest


def test_delete_does_not_early_return_on_missing_session():
    body = _delete_message_body()
    # The bug was an unconditional early-out when no session existed.
    assert not re.search(
        r"if\s*\(\s*!sessionId\s*\)\s*return\s*;", body
    ), "deleteMessage must not early-return on a missing session (#1428)"
    # The DOM-removal fallback must also fire when there's no session.
    assert re.search(
        r"!msgIds\.length\s*\|\|\s*!sessionId", body
    ), "DOM-removal fallback should cover the no-session case"
