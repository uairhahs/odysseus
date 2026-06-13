import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_RUNNING = ROOT / "static" / "js" / "cookbookRunning.js"


def _source() -> str:
    return COOKBOOK_RUNNING.read_text(encoding="utf-8")


def _strip_quotes_and_space(s: str) -> str:
    """Normalize whitespace and quote style so cosmetic differences don't matter."""
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    return s.strip()


def test_cookbook_marks_local_endpoint_registration_as_container_local():
    src = _source()
    assert "function _appendCookbookEndpointScope" in src

    norm = _strip_quotes_and_space(src)
    assert "fd.append('container_local', 'true')" in norm

    assert src.count("_appendCookbookEndpointScope(fd,") >= 3


def test_cookbook_does_not_use_local_as_endpoint_hostname():
    src = _source()
    assert "function _connectHostFromRemote" in src

    norm = _strip_quotes_and_space(src)
    # Must short-circuit to the fallback when host is empty or the literal "local"
    assert "if (!host || host === 'local') return fallback;" in norm

    # Must not blindly default an unset remoteHost to "localhost" for connection purposes
    assert "rawHost = task.remoteHost || 'localhost';" not in norm


def test_cookbook_advertised_bind_urls_keep_connectable_host():
    src = _source()
    assert "function _endpointFromAdvertisedUrl" in src

    norm = _strip_quotes_and_space(src)
    # Any-interface bind hosts (0.0.0.0, ::, etc.) must be rewritten to the
    # currently-connected host, not used verbatim.
    assert "_isAnyBindHost(u.hostname) ? currentHost" in norm

    # Must not naively take the advertised hostname as-is.
    assert "host = u.hostname || host;" not in norm
