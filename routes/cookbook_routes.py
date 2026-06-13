# Cookbook routes

#  model download, serve, cache scanning, and cookbook state sync

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.atomic_io import atomic_write_json
from core.constants import DATA_DIR
from core.database import utcnow_naive
from core.middleware import require_admin
from core.platform_compat import (
    IS_WINDOWS,
    NVIDIA_PATH_CANDIDATES,
    SSH_PATH_OVERRIDE,
    get_wsl_windows_user_profile,
    kill_process_tree,
    pid_alive,
    safe_chmod,
    translate_path,
    which_tool,
)
from routes.cookbook_bash_builders import (
    build_bash_download_lines,
    build_bash_serve_lines,
)
from routes.cookbook_helpers import (
    _SSH_PORT_RE,
    ModelDownloadRequest,
    ServeRequest,
    _cached_model_scan_script,
    _resolve_gone_session_status,
    _shell_path,
    _validate_gpus,
    _validate_include,
    _validate_local_dir,
    _validate_remote_host,
    _validate_repo_id,
    _validate_serve_cmd,
    _validate_serve_model_id,
    _validate_ssh_port,
    _validate_token,
    run_ssh_command_async,
)
from routes.cookbook_ps1_builders import build_ps1_download_lines, build_ps1_serve_lines
from routes.cookbook_utils import (
    binary_available,
    encrypt_secret,
    get_cookbook_known_hosts_path,
    get_cookbook_ssh_dir,
    get_cookbook_ssh_key_path,
    load_stored_hf_token,
    missing_binary_message,
    read_cookbook_public_key,
)
from routes.shell_routes import TMUX_LOG_DIR
from src.auth_helpers import require_user

# other prior imports have been moved to relevant helper modules to avoid circular dependencies

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Un-nested Router Definition
cookbook_router = APIRouter(tags=["cookbook"])
COOKBOOK_STATE_PATH = Path(DATA_DIR) / "cookbook_state.json"


@cookbook_router.get("/api/cookbook/")
async def get_cookbook_status():
    return {"status": "cookbook routes active"}


@cookbook_router.get("/api/cookbook/ssh-key")
async def get_cookbook_ssh_key(request: Request):
    require_admin(request)
    public_key = read_cookbook_public_key()
    return {
        "configured": bool(public_key),
        "public_key": public_key,
    }


@cookbook_router.post("/api/cookbook/ssh-key")
async def generate_cookbook_ssh_key(request: Request):
    require_admin(request)
    ssh_dir = get_cookbook_ssh_dir()
    key_path = get_cookbook_ssh_key_path()
    ssh_dir.mkdir(parents=True, exist_ok=True)
    safe_chmod(ssh_dir, 0o700)
    if not key_path.exists():
        ssh_keygen = which_tool("ssh-keygen") or "ssh-keygen"
        proc = await asyncio.create_subprocess_exec(
            ssh_keygen,
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "odysseus-cookbook",
            "-f",
            str(key_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()[-500:]
            return {"ok": False, "error": detail or "Failed to generate SSH key"}
    safe_chmod(key_path, 0o600)
    safe_chmod(key_path.with_suffix(".pub"), 0o644)
    return {"ok": True, "public_key": read_cookbook_public_key()}


@cookbook_router.post("/api/model/download")
async def model_download(request: Request, req: ModelDownloadRequest):
    require_admin(request)
    # Input validation
    _validate_remote_host(req.remote_host)
    req.ssh_port = _validate_ssh_port(req.ssh_port)
    req.hf_token = req.hf_token or load_stored_hf_token(COOKBOOK_STATE_PATH)
    _validate_token(req.hf_token)
    # Restored Download Validations
    _validate_repo_id(req.repo_id)
    _validate_include(req.include)
    req.local_dir = _validate_local_dir(req.local_dir)
    TMUX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_id = f"cookbook-{uuid.uuid4()}"
    remote = req.remote_host
    is_windows = (req.platform == "windows") if remote else IS_WINDOWS
    local_windows = IS_WINDOWS and not remote
    # All remote SSH calls use StrictHostKeyChecking=yes + the Cookbook-specific
    # known_hosts file. The UI "Test SSH" step (StrictHostKeyChecking=accept-new)
    # pre-populates that file as a prerequisite, so first-connect is handled
    # there and subsequent operations enforce strict key verification.
    # Check dependencies
    if (
        not is_windows
        and not local_windows
        and not await binary_available("tmux", remote, req.ssh_port)
    ):
        return {
            "ok": False,
            "error": missing_binary_message("tmux", remote or "local server"),
            "session_id": session_id,
        }
    try:
        _dl_short = req.repo_id.split("/")[-1] if "/" in req.repo_id else req.repo_id
        _dl_base = (
            (req.local_dir.rstrip("/") + "/" + _dl_short) if req.local_dir else None
        )
        _dl_shell = _shell_path(_dl_base) if _dl_base else None
        hf_cmd = f"hf download {req.repo_id}"
        if req.include:
            hf_cmd += f" --include '{req.include}'"
        if _dl_shell:
            hf_cmd += f" --local-dir {_dl_shell}"
        dl_pyarg = (
            (", local_dir=os.path.expanduser(" + repr(_dl_base) + ")")
            if _dl_base
            else ""
        )
        if is_windows:
            script_lines = build_ps1_download_lines(req, session_id, hf_cmd, dl_pyarg)
            script_content = "\n".join(script_lines)
            script_path = TMUX_LOG_DIR / f"{session_id}_run.ps1"
            script_path.write_text(script_content, encoding="utf-8")
            if remote:
                port_arg = (
                    ["-p", req.ssh_port]
                    if req.ssh_port and req.ssh_port != "22"
                    else []
                )
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "UserKnownHostsFile=" + str(get_cookbook_known_hosts_path()),
                    *port_arg,
                    remote,
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                # script_content is safe to embed verbatim: all user-supplied
                # values are validated or escaped before reaching the ps1 builder.
                #   repo_id   — _validate_repo_id: ^[A-Za-z0-9._-/]+$ (no PS metacharacters)
                #   hf_token  — wrapped with _ps_squote()
                #   dl_pyarg  — built with Python repr() on the local path
                #   include   — _validate_include: ^[A-Za-z0-9._\-*?/\[\]]+$
                #   env_prefix — routed through _safe_env_prefix()
                detached = (
                    "Start-Job -ScriptBlock {\n" + script_content + "\n} | Out-Null"
                )
                await proc.communicate(input=detached.encode("utf-8"))
                if proc.returncode != 0:
                    return {
                        "ok": False,
                        "error": f"Failed to launch remote download (exit {proc.returncode}).",
                        "session_id": session_id,
                    }
            else:
                proc = await asyncio.create_subprocess_exec(
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"Start-Process powershell -WindowStyle Hidden -ArgumentList "
                    f"'-NoProfile','-ExecutionPolicy','Bypass','-File','{script_path}'",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate()
        else:
            script_path = TMUX_LOG_DIR / f".{session_id}_run.sh"
            script_lines = build_bash_download_lines(
                req, session_id, hf_cmd, dl_pyarg, wrapper_script_path=str(script_path)
            )
            script_content = "\n".join(script_lines)
            script_path.write_text(script_content, encoding="utf-8")
            safe_chmod(script_path, 0o755)
            if remote:
                port_arg = (
                    ["-p", req.ssh_port]
                    if req.ssh_port and req.ssh_port != "22"
                    else []
                )
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "UserKnownHostsFile=" + str(get_cookbook_known_hosts_path()),
                    *port_arg,
                    remote,
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_id,
                    "bash",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate(input=script_content.encode("utf-8"))
            else:
                tmux_cmd = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_id,
                    "bash",
                    str(script_path),
                ]
                proc = await asyncio.create_subprocess_exec(
                    *tmux_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate()
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": f"Failed to launch download process (exit {proc.returncode}).",
                "session_id": session_id,
            }
    except Exception as e:
        logger.error(f"Launch exception: {e}")
        return {"ok": False, "error": str(e), "session_id": session_id}
    _persist_task(session_id, "download", remote, req.ssh_port, req.platform)
    return {
        "ok": True,
        "session_id": session_id,
        "remote": remote or "local",
    }


@cookbook_router.post("/api/model/serve")
async def model_serve(request: Request, req: ServeRequest):
    require_admin(request)
    # Validation logic here
    _validate_remote_host(req.remote_host)
    req.ssh_port = _validate_ssh_port(req.ssh_port)
    req.hf_token = req.hf_token or load_stored_hf_token(COOKBOOK_STATE_PATH)
    # Restored Serve Validations
    req.cmd = _validate_serve_cmd(req.cmd)
    if not req.cmd:
        raise HTTPException(400, "cmd is required")
    _validate_serve_model_id(req.repo_id)
    _validate_gpus(req.gpus)
    session_id = f"serve-{uuid.uuid4()}"
    TMUX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    remote = req.remote_host
    is_windows = (req.platform == "windows") if remote else IS_WINDOWS
    local_windows = IS_WINDOWS and not remote
    is_pip_install = (
        "pip install" in req.cmd or "uv pip" in req.cmd or "uv add" in req.cmd
    )
    try:
        if is_windows:
            script_lines = build_ps1_serve_lines(
                req, session_id, is_pip_install=is_pip_install
            )
            script_content = "\n".join(script_lines)
            script_path = TMUX_LOG_DIR / f"{session_id}_serve.ps1"
            script_path.write_text(script_content, encoding="utf-8")
            if remote:
                port_arg = (
                    ["-p", req.ssh_port]
                    if req.ssh_port and req.ssh_port != "22"
                    else []
                )
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "UserKnownHostsFile=" + str(get_cookbook_known_hosts_path()),
                    *port_arg,
                    remote,
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                detached = (
                    "Start-Job -ScriptBlock {\n" + script_content + "\n} | Out-Null"
                )
                await proc.communicate(input=detached.encode("utf-8"))
                if proc.returncode != 0:
                    return {
                        "ok": False,
                        "error": f"Failed to launch remote serve (exit {proc.returncode}).",
                        "session_id": session_id,
                    }
            else:
                proc = await asyncio.create_subprocess_exec(
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"Start-Process powershell -WindowStyle Hidden -ArgumentList "
                    f"'-NoProfile','-ExecutionPolicy','Bypass','-File','{script_path}'",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate()
        else:
            script_path = TMUX_LOG_DIR / f".{session_id}_serve.sh"
            script_lines = build_bash_serve_lines(
                req,
                session_id,
                is_pip_install=is_pip_install,
                local_windows=local_windows,
                remote=remote,
            )
            script_content = "\n".join(script_lines)
            script_path.write_text(script_content, encoding="utf-8")
            safe_chmod(script_path, 0o755)
            if remote:
                port_arg = (
                    ["-p", req.ssh_port]
                    if req.ssh_port and req.ssh_port != "22"
                    else []
                )
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "UserKnownHostsFile=" + str(get_cookbook_known_hosts_path()),
                    *port_arg,
                    remote,
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_id,
                    "bash",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate(input=script_content.encode("utf-8"))
            else:
                tmux_cmd = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_id,
                    "bash",
                    str(script_path),
                ]
                proc = await asyncio.create_subprocess_exec(
                    *tmux_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate()
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": f"Failed to launch serve process (exit {proc.returncode}).",
                "session_id": session_id,
            }
    except Exception as e:
        logger.error(f"Serve launch exception: {e}")
        return {"ok": False, "error": str(e), "session_id": session_id}
    _persist_task(session_id, "serve", remote, req.ssh_port, req.platform)
    return {
        "ok": True,
        "session_id": session_id,
        "remote": remote or "local",
    }


_SCRUB_ENV_KEYS = {"hfToken"}


def _scrub_state_for_client(state: dict) -> dict:
    """Strip server-side secrets before sending state to the browser."""
    import copy

    state = copy.deepcopy(state)
    env = state.get("env") if isinstance(state, dict) else None
    if isinstance(env, dict):
        token = env.pop("hfToken", None)
        env["hfTokenConfigured"] = bool(token)
        for key in _SCRUB_ENV_KEYS - {"hfToken"}:
            env.pop(key, None)
    return state


def _read_cookbook_state() -> dict:
    """Read the cookbook state from disk."""
    if not COOKBOOK_STATE_PATH.exists():
        return {}
    try:
        return json.loads(COOKBOOK_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cookbook_state(state: dict) -> bool:
    """Write the cookbook state to disk atomically."""
    try:
        atomic_write_json(COOKBOOK_STATE_PATH, state)
        return True
    except Exception:
        return False


def _persist_task(
    session_id: str,
    task_type: str,
    remote: str | None,
    ssh_port: str | None,
    platform: str | None,
) -> None:
    """Append a new running task entry to cookbook state."""
    state = _read_cookbook_state()
    tasks = list(state.get("tasks") or [])
    tasks.append(
        {
            "id": session_id,
            "sessionId": session_id,
            "type": task_type,
            "status": "running",
            "remoteHost": remote or None,
            "sshPort": ssh_port or None,
            "platform": platform or None,
            "created_at": utcnow_naive().isoformat() + "Z",
        }
    )
    state["tasks"] = tasks
    _write_cookbook_state(state)


@cookbook_router.post("/api/cookbook/dep-installed")
async def mark_dep_installed(request: Request):
    """Record a successfully completed dependency install in persistent state.

    Called by the client reconnect loop when a _dep task finishes with exit 0.
    Stores {pip_name -> {host, ts}} in cookbook_state so list_packages can
    return installed=true even if importlib.metadata hasn't reflected the
    install yet (extras venv, uv link-mode=copy, PATH not refreshed).
    """
    require_admin(request)
    import time

    body = await request.json()
    pip_name = str(body.get("pip") or "").strip()
    host = str(body.get("host") or "").strip()
    if not pip_name:
        return {"ok": False, "error": "pip required"}
    state = _read_cookbook_state()
    installed = state.setdefault("installed_deps", {})
    installed[pip_name] = {"host": host, "ts": int(time.time())}
    _write_cookbook_state(state)
    return {"ok": True}


@cookbook_router.get("/api/cookbook/state")
async def get_cookbook_state(request: Request):
    """Return the full cookbook state (tasks, metadata, etc.)."""
    require_admin(request)
    return _scrub_state_for_client(_read_cookbook_state())


@cookbook_router.get("/api/cookbook/tasks/status")
async def get_cookbook_tasks_status(request: Request):
    """Return task statuses for the Running tab."""
    require_admin(request)
    state = _read_cookbook_state()
    tasks = state.get("tasks") or []

    async def _resolve_status(t: dict) -> str:
        status = t.get("status")
        if status != "running":
            return status
        sid = t.get("sessionId")
        if not sid:
            return status
        remote_host = t.get("remoteHost")
        ssh_port = t.get("sshPort")
        remote_platform = t.get("platform", "")
        try:
            if remote_host:
                if remote_platform == "windows":
                    rc, _, _ = await run_ssh_command_async(
                        remote_host,
                        ssh_port,
                        f'powershell -NoProfile -Command "Test-Path $env:TEMP\\odysseus-sessions\\{sid}.exit"',
                        timeout=8,
                    )
                    return "stopped" if rc == 0 else "running"
                rc, _, _ = await run_ssh_command_async(
                    remote_host,
                    ssh_port,
                    f"tmux has-session -t {shlex.quote(sid)}",
                    timeout=8,
                )
                return "running" if rc == 0 else "stopped"
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "has-session",
                "-t",
                sid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if proc.returncode == 0:
                return "running"
            # Session is gone. Check the retained tmux pane history for the
            # runner's exit sentinel to distinguish a completed download from
            # a crashed/stopped task.
            cap_proc = await asyncio.create_subprocess_exec(
                "tmux",
                "capture-pane",
                "-p",
                "-S",
                "-200",
                "-t",
                sid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            cap_out, _ = await cap_proc.communicate()
            if cap_proc.returncode == 0:
                snapshot = cap_out.decode("utf-8", errors="replace")
                return _resolve_gone_session_status(snapshot, t.get("type", ""))
            return "stopped"
        except Exception:
            return status

    resolved = await asyncio.gather(
        *[_resolve_status(t) for t in tasks if isinstance(t, dict)]
    )
    return {
        "tasks": [
            {
                "id": t.get("id"),
                "sessionId": t.get("sessionId"),
                "session_id": t.get("sessionId"),
                "status": live_status,
                "type": t.get("type"),
                "created_at": t.get("created_at"),
                "remoteHost": t.get("remoteHost"),
            }
            for t, live_status in zip(
                [t for t in tasks if isinstance(t, dict)], resolved, strict=False
            )
        ]
    }


# Keys managed exclusively by the server; never overwritten by client payload.
_SERVER_MANAGED_KEYS = {"tasks"}


@cookbook_router.post("/api/cookbook/state")
async def post_cookbook_state(request: Request):
    """Update the cookbook state."""
    require_admin(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            return {"ok": False, "error": "Body must be a dict"}
        # Read once; use for both secret preservation and server-key protection.
        existing = _read_cookbook_state()
        existing_env = existing.get("env") or {}
        body_env = body.get("env")
        if isinstance(body_env, dict):
            for key in _SCRUB_ENV_KEYS:
                incoming = body_env.get(key)
                if incoming:
                    body_env[key] = encrypt_secret(incoming)
                elif existing_env.get(key):
                    body_env[key] = existing_env[key]
                else:
                    body_env.pop(key, None)
        # Merge: client payload wins for its own keys; server-managed keys are preserved.
        merged = {**body}
        for key in _SERVER_MANAGED_KEYS:
            if key in existing:
                merged[key] = existing[key]
        ok = _write_cookbook_state(merged)
        return {"ok": ok}
    except Exception as e:
        logger.warning(f"cookbook state update failed: {e}")
        return {"ok": False, "error": str(e)}


# ── /api/model/cached ──


@cookbook_router.get("/api/model/cached")
async def model_cached(
    request: Request,
    host: str | None = None,
    model_dir: str | None = None,
    ssh_port: str | None = None,
    platform: str | None = None,
):
    """List cached models. Scans HF cache + optional model directory."""
    require_admin(request)
    host = _validate_remote_host(host)
    if ssh_port is not None and ssh_port != "" and not _SSH_PORT_RE.fullmatch(ssh_port):
        raise HTTPException(400, "Invalid ssh_port")
    TMUX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_dirs = []
    if model_dir:
        for d in model_dir.split(","):
            d = d.strip()
            if d:
                model_dirs.append(translate_path(d) if not host else d)
    win_hf_hub = None
    if not host:
        win_profile = get_wsl_windows_user_profile()
        win_hf_hub = (
            os.path.join(win_profile, ".cache", "huggingface", "hub")
            if win_profile
            else None
        )
    paths_code = _cached_model_scan_script(model_dirs, win_hf_hub)
    scan_py = TMUX_LOG_DIR / "scan_cache.py"
    scan_py.write_text(paths_code, encoding="utf-8")
    scan_payload = scan_py.read_bytes()
    stdout_b = b""
    stderr_b = b""
    if host:
        if platform == "windows":
            remote_cmd = "powershell -NoProfile -Command -"
        else:
            remote_cmd = (
                "if command -v python3 >/dev/null 2>&1; then python3 -; "
                "elif command -v python >/dev/null 2>&1; then python -; "
                'else echo "python3/python not found" >&2; exit 127; fi'
            )
        _rc, stdout_b, stderr_b = await run_ssh_command_async(
            host, ssh_port, remote_cmd, timeout=60, stdin_data=scan_payload
        )
    else:
        local_py = (
            sys.executable or which_tool("python3") or which_tool("python") or "python"
        )
        proc = await asyncio.create_subprocess_exec(
            local_py,
            str(scan_py),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=60)
    models = []
    try:
        raw = json.loads(stdout_b.decode(errors="replace").strip())
        for m in raw:
            size_gb = m["size_bytes"] / (1024**3)
            size_str = (
                f"{size_gb:.1f} GB"
                if size_gb >= 1
                else f"{m['size_bytes'] / (1024**2):.0f} MB"
            )
            entry = {
                "repo_id": m["repo_id"],
                "size": size_str,
                "nb_files": m["nb_files"],
                "has_incomplete": m["has_incomplete"],
                "status": "downloading" if m["has_incomplete"] else "ready",
                "path": m.get("path", ""),
                "is_diffusion": m.get("is_diffusion", False),
            }
            for key in (
                "is_local_dir",
                "is_gguf",
                "backend",
                "is_ollama",
                "gguf_files",
            ):
                if m.get(key) is not None:
                    entry[key] = m[key]
            models.append(entry)
    except Exception as e:
        logger.warning("Failed to parse cached models: %s", e)
        logger.warning("stderr: %s", stderr_b.decode(errors="replace")[:500])
    return {"models": models, "host": host or "local"}


# ── /api/cookbook/setup ──


class SetupRequest(BaseModel):
    host: str
    ssh_port: str | None = None


@cookbook_router.post("/api/cookbook/setup")
async def server_setup(request: Request, req: SetupRequest):
    """Install required dependencies on a remote server via SSH."""
    require_admin(request)
    host = _validate_remote_host(req.host)
    if not host:
        raise HTTPException(400, "host is required")
    port = req.ssh_port
    if port is not None and port != "" and not _SSH_PORT_RE.fullmatch(port):
        raise HTTPException(400, "Invalid ssh_port")
    pf = f"-p {port} " if port and port != "22" else ""
    known_hosts = str(get_cookbook_known_hosts_path())
    ssh_opts = f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={known_hosts}"
    detect_cmd = f'ssh {ssh_opts} {pf}{host} "echo %OS%"'
    platform = "linux"
    try:
        proc = await asyncio.create_subprocess_shell(
            detect_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        out = stdout.decode().strip()
        if "Windows_NT" in out:
            platform = "windows"
        else:
            detect_cmd2 = f"ssh {ssh_opts} {pf}{host} 'test -d /data/data/com.termux && echo termux || echo linux'"
            proc2 = await asyncio.create_subprocess_shell(
                detect_cmd2,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=10)
            platform = stdout2.decode().strip()
    except Exception:
        platform = "linux"
    if platform == "windows":
        setup_script = (
            'powershell -Command "'
            "New-Item -ItemType Directory -Force -Path $env:TEMP\\odysseus-sessions | Out-Null; "
            "try { python --version } catch { Write-Host 'ERROR: Python not found — install from python.org'; exit 1 }; "
            "python -m pip install -q huggingface-hub 2>$null; "
            "python -c \\\"from huggingface_hub import snapshot_download; print('OK')\\\""
            '"'
        )
        cmd = f"ssh {ssh_opts} {pf}{host} {setup_script}"
    elif platform == "termux":
        setup_script = (
            "pkg install -y python tmux 2>/dev/null; "
            "pip install --no-deps -q huggingface-hub 2>/dev/null; "
            "pip install -q filelock fsspec packaging pyyaml tqdm typer httpx requests 2>/dev/null; "
            "python3 -c 'from huggingface_hub import snapshot_download; print(\"OK\")'"
        )
        cmd = f"ssh {ssh_opts} {pf}{host} '{setup_script}'"
    else:
        setup_script = (
            "if ! command -v tmux >/dev/null 2>&1; then "
            "  if command -v apt-get >/dev/null 2>&1; then sudo -n apt-get install -y tmux 2>/dev/null; "
            "  elif command -v pacman >/dev/null 2>&1; then sudo -n pacman -S --noconfirm tmux 2>/dev/null; "
            "  elif command -v dnf >/dev/null 2>&1; then sudo -n dnf install -y tmux 2>/dev/null; "
            "  elif command -v apk >/dev/null 2>&1; then sudo -n apk add --no-interactive tmux 2>/dev/null; "
            "  elif command -v zypper >/dev/null 2>&1; then sudo -n zypper --non-interactive install tmux 2>/dev/null; "
            "  fi; "
            "fi; "
            "command -v tmux >/dev/null 2>&1 || echo 'WARNING: tmux missing and auto-install failed (need passwordless sudo). Install manually.'; "
            "pip install -q huggingface_hub hf_transfer 2>/dev/null || "
            "pip install --user --break-system-packages -q huggingface_hub hf_transfer 2>/dev/null || "
            "pip3 install --user --break-system-packages -q huggingface_hub hf_transfer 2>/dev/null; "
            "python3 -c 'from huggingface_hub import snapshot_download; print(\"OK\")'"
        )
        cmd = f"ssh {ssh_opts} {pf}{host} '{setup_script}'"
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode() + stderr.decode()
        return {"ok": "OK" in output, "output": output.strip(), "platform": platform}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "Setup timed out (120s)", "platform": platform}
    except Exception as e:
        return {"ok": False, "error": str(e), "platform": platform}


# ── /api/cookbook/gpus helpers ──


async def _run_nvidia_smi(
    query: str, host: str | None, ssh_port: str | None, timeout: int = 8
):
    if host:
        candidates = [query]
        stripped = query.strip()
        if stripped.startswith("nvidia-smi "):
            args = stripped[len("nvidia-smi ") :]
            candidates.append(
                "bash -lc " + shlex.quote(f"{SSH_PATH_OVERRIDE}nvidia-smi {args}")
            )
            for nvidia_path in NVIDIA_PATH_CANDIDATES:
                candidates.append(f"{nvidia_path} {args}")
        last_err = "nvidia-smi failed"
        for candidate in candidates:
            try:
                rc, stdout, stderr = await run_ssh_command_async(
                    host, ssh_port, candidate, connect_timeout=5, timeout=timeout
                )
            except asyncio.TimeoutError:
                return None, "nvidia-smi timed out"
            if rc == 0:
                return stdout.decode("utf-8", errors="replace"), None
            err = (stderr.decode("utf-8", errors="replace") or "").strip()[:200]
            if err:
                last_err = err
        return None, last_err
    proc = await asyncio.create_subprocess_exec(
        *shlex.split(query),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return None, "nvidia-smi timed out"
    if proc.returncode != 0:
        err = (stderr.decode("utf-8", errors="replace") or "").strip()[:200]
        return None, err or "nvidia-smi failed"
    return stdout.decode("utf-8", errors="replace"), None


async def _run_gpu_shell(
    cmd_text: str, host: str | None, ssh_port: str | None, timeout: int = 8
):
    known_hosts = str(get_cookbook_known_hosts_path())
    if host:
        pf = f"-p {ssh_port} " if ssh_port and ssh_port != "22" else ""
        quoted_cmd = shlex.quote(cmd_text)
        remote_cmd = (
            f"if command -v sh >/dev/null 2>&1; then sh -lc {quoted_cmd}; "
            f"elif command -v bash >/dev/null 2>&1; then bash -lc {quoted_cmd}; "
            f"elif command -v zsh >/dev/null 2>&1; then zsh -lc {quoted_cmd}; "
            "else echo 'No POSIX shell found for GPU probe' >&2; exit 127; fi"
        )
        cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=yes "
            f"-o UserKnownHostsFile={known_hosts} {pf}{host} {shlex.quote(remote_cmd)}"
        )
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    else:
        proc = await asyncio.create_subprocess_shell(
            cmd_text, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return None, "GPU probe timed out"
    if proc.returncode != 0:
        err = (stderr.decode("utf-8", errors="replace") or "").strip()[:200]
        return None, err or f"GPU probe failed ({proc.returncode})"
    return stdout.decode("utf-8", errors="replace"), None


async def _gpu_read_file(
    path: str, host: str | None, ssh_port: str | None
) -> str | None:
    out, err = await _run_gpu_shell(
        f"cat {shlex.quote(path)} 2>/dev/null", host, ssh_port, timeout=4
    )
    return out.strip() if err is None and out is not None else None


async def _probe_gpu_device_processes(host: str | None, ssh_port: str | None) -> list:
    pid_cmd = (
        "{ command -v lsof >/dev/null 2>&1 && "
        "lsof -w -t /dev/kfd /dev/dri/renderD* 2>/dev/null || true; "
        "command -v fuser >/dev/null 2>&1 && "
        "fuser /dev/kfd /dev/dri/renderD* 2>/dev/null || true; } "
        "| tr ' ' '\\n' | sed '/^[0-9][0-9]*$/!d' | sort -n -u"
    )
    out, err = await _run_gpu_shell(pid_cmd, host, ssh_port, timeout=5)
    if err is not None or not out:
        return []
    processes = []
    seen: set = set()
    for raw in out.splitlines():
        try:
            pid = int(raw.strip())
        except ValueError:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        name_out, _ = await _run_gpu_shell(
            f"ps -p {pid} -o comm= 2>/dev/null", host, ssh_port, timeout=3
        )
        name = (
            (name_out or "").strip().splitlines()[0]
            if (name_out or "").strip()
            else "process"
        )
        processes.append({"pid": pid, "name": name[:80], "used_mb": 0})
    return processes


async def _probe_amd_sysfs(host: str | None, ssh_port: str | None) -> list:
    out, err = await _run_gpu_shell(
        "ls -1 /sys/class/drm 2>/dev/null", host, ssh_port, timeout=4
    )
    if err is not None or not out:
        return []
    gpus = []
    for entry in out.split():
        if not entry.startswith("card") or "-" in entry:
            continue
        base = f"/sys/class/drm/{entry}/device"
        vendor = await _gpu_read_file(f"{base}/vendor", host, ssh_port)
        if vendor != "0x1002":
            continue
        vram_raw = await _gpu_read_file(f"{base}/mem_info_vram_total", host, ssh_port)
        vis_raw = await _gpu_read_file(
            f"{base}/mem_info_vis_vram_total", host, ssh_port
        )
        gtt_raw = await _gpu_read_file(f"{base}/mem_info_gtt_total", host, ssh_port)
        vram_bytes = int(vram_raw) if vram_raw and vram_raw.isdigit() else 0
        vis_bytes = int(vis_raw) if vis_raw and vis_raw.isdigit() else 0
        gtt_bytes = int(gtt_raw) if gtt_raw and gtt_raw.isdigit() else 0
        total_bytes = max(vram_bytes, vis_bytes)
        used_attr = (
            "mem_info_vis_vram_used"
            if vis_bytes and vis_bytes >= vram_bytes
            else "mem_info_vram_used"
        )
        unified = bool(vis_bytes and vis_bytes >= vram_bytes)
        if total_bytes <= 0:
            total_bytes = gtt_bytes
            used_attr = "mem_info_gtt_used"
            unified = True
        if total_bytes <= 0:
            continue
        used_raw = await _gpu_read_file(f"{base}/{used_attr}", host, ssh_port)
        used_bytes = int(used_raw) if used_raw and used_raw.isdigit() else 0
        name = await _gpu_read_file(f"{base}/product_name", host, ssh_port)
        if not name:
            device = await _gpu_read_file(f"{base}/device", host, ssh_port)
            name = f"AMD GPU {device or entry}"
        total_mb = max(0, int(total_bytes / (1024 * 1024)))
        used_mb = max(0, min(total_mb, int(used_bytes / (1024 * 1024))))
        free_mb = max(0, total_mb - used_mb)
        gtt_used_raw = await _gpu_read_file(f"{base}/mem_info_gtt_used", host, ssh_port)
        gtt_used_mb = (
            max(0, int(int(gtt_used_raw) / (1024 * 1024)))
            if (gtt_used_raw and gtt_used_raw.isdigit())
            else 0
        )
        gpus.append(
            {
                "index": len(gpus),
                "name": name,
                "uuid": entry,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "gtt_used_mb": gtt_used_mb,
                "util_pct": 0,
                "busy": bool(total_mb and (free_mb / total_mb) < 0.85),
                "processes": [],
                "backend": "rocm",
                "source": "amd-sysfs",
                "unified_memory": unified,
            }
        )
    if gpus:
        processes = await _probe_gpu_device_processes(host, ssh_port)
        if processes:
            gpus[0]["processes"] = processes
            gpus[0]["busy"] = True
    return gpus


# ── /api/cookbook/gpus ──


@cookbook_router.get("/api/cookbook/gpus")
async def list_gpus(
    request: Request, host: str | None = None, ssh_port: str | None = None
):
    """Probe GPU memory/process state locally or via SSH."""
    require_admin(request)
    host = _validate_remote_host(host)
    if ssh_port is not None and ssh_port != "" and not _SSH_PORT_RE.fullmatch(ssh_port):
        raise HTTPException(400, "Invalid ssh_port")
    gpu_query = "nvidia-smi --query-gpu=index,name,memory.free,memory.total,memory.used,utilization.gpu,uuid --format=csv,noheader,nounits"
    nvidia_error = None
    try:
        gpu_out, err = await _run_nvidia_smi(gpu_query, host, ssh_port)
        if err is not None:
            nvidia_error = err
            gpu_out = ""
    except FileNotFoundError:
        nvidia_error = "nvidia-smi not found"
        gpu_out = ""
    except Exception as e:
        nvidia_error = str(e)[:200]
        gpu_out = ""
    gpus = []
    uuid_to_idx: dict = {}
    for line in (gpu_out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(float(parts[2]))
            total_mb = int(float(parts[3]))
            used_mb = int(float(parts[4]))
            util_pct = int(float(parts[5]))
            gpu_uuid = parts[6]
        except (ValueError, IndexError):
            continue
        uuid_to_idx[gpu_uuid] = idx
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "uuid": gpu_uuid,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "util_pct": util_pct,
                "busy": total_mb > 0 and (free_mb / total_mb) < 0.5,
                "processes": [],
            }
        )
    proc_query = "nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits"
    try:
        proc_out, proc_err = await _run_nvidia_smi(
            proc_query, host, ssh_port, timeout=5
        )
        if proc_err is None and proc_out:
            gpus_by_idx = {g["index"]: g for g in gpus}
            for line in proc_out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    pid = int(parts[0])
                    pmem = int(float(parts[3]))
                except (ValueError, IndexError):
                    continue
                idx = uuid_to_idx.get(parts[1])
                if idx is None or idx not in gpus_by_idx:
                    continue
                gpus_by_idx[idx]["processes"].append(
                    {"pid": pid, "name": parts[2], "used_mb": pmem}
                )
    except Exception as e:
        logger.warning("Error occurred while probing GPU processes: %s", e)
    if gpus:
        return {"ok": True, "gpus": gpus, "backend": "cuda", "source": "nvidia-smi"}
    if not host and sys.platform == "darwin":
        try:
            from services.hwfit.hardware import detect_system

            info = detect_system(fresh=True)
            backend = str(info.get("backend") or "").lower()
            if backend in {"metal", "mps", "apple"} and info.get("gpu_count", 0) > 0:
                total_mb = int(
                    float(info.get("gpu_vram_gb") or info.get("total_ram_gb") or 0)
                    * 1024
                )
                free_mb = int(float(info.get("available_ram_gb") or 0) * 1024)
                if total_mb and (free_mb <= 0 or free_mb > total_mb):
                    free_mb = total_mb
                used_mb = max(0, total_mb - max(0, free_mb))
                return {
                    "ok": True,
                    "gpus": [
                        {
                            "index": 0,
                            "name": info.get("gpu_name")
                            or info.get("cpu_name")
                            or "Apple Silicon GPU",
                            "uuid": "apple-metal-0",
                            "free_mb": max(0, free_mb),
                            "total_mb": max(0, total_mb),
                            "used_mb": used_mb,
                            "util_pct": 0,
                            "busy": bool(total_mb and (free_mb / total_mb) < 0.5),
                            "processes": [],
                            "backend": "metal",
                            "source": "apple-metal",
                            "unified_memory": True,
                        }
                    ],
                    "backend": "metal",
                    "source": "apple-metal",
                    "fallback_from": "nvidia-smi",
                    "nvidia_error": nvidia_error,
                }
        except Exception as e:
            logger.warning("Apple Metal GPU fallback failed: %s", e)
    amd_gpus = await _probe_amd_sysfs(host, ssh_port)
    if amd_gpus:
        return {
            "ok": True,
            "gpus": amd_gpus,
            "backend": "rocm",
            "source": "amd-sysfs",
            "fallback_from": "nvidia-smi",
            "nvidia_error": nvidia_error,
        }
    processes = await _probe_gpu_device_processes(host, ssh_port)
    if processes:
        return {
            "ok": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "GPU device holders",
                    "uuid": "dev-dri",
                    "free_mb": 0,
                    "total_mb": 0,
                    "used_mb": 0,
                    "util_pct": 0,
                    "busy": True,
                    "processes": processes,
                    "backend": "generic",
                    "source": "gpu-devices",
                }
            ],
            "backend": "generic",
            "source": "gpu-devices",
            "fallback_from": "nvidia-smi",
            "nvidia_error": nvidia_error,
        }
    return {
        "ok": False,
        "error": nvidia_error or "No GPU memory probe available",
        "gpus": [],
    }


# ── /api/cookbook/kill-pid ──


class KillPidRequest(BaseModel):
    pid: int
    host: str | None = None
    ssh_port: str | None = None
    signal: str = "TERM"


@cookbook_router.post("/api/cookbook/kill-pid")
async def kill_pid(request: Request, req: KillPidRequest):
    """Kill a PID that's holding GPU memory. Admin-gated."""
    require_admin(request)
    if req.pid < 100:
        raise HTTPException(
            400, f"Refusing to signal PID {req.pid} (<100, likely system process)"
        )
    sig = (req.signal or "TERM").upper()
    if sig not in ("TERM", "KILL", "INT"):
        raise HTTPException(400, "signal must be TERM, KILL, or INT")
    host = _validate_remote_host(req.host)
    if req.ssh_port and not _SSH_PORT_RE.fullmatch(req.ssh_port):
        raise HTTPException(400, "Invalid ssh_port")
    kill_cmd = f"kill -{sig} {req.pid}"
    try:
        if host:
            pf = f"-p {req.ssh_port} " if req.ssh_port and req.ssh_port != "22" else ""
            known_hosts = str(get_cookbook_known_hosts_path())
            cmd = (
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=yes "
                f"-o UserKnownHostsFile={known_hosts} {pf}{host} '{kill_cmd}'"
            )
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        elif IS_WINDOWS:
            if not pid_alive(req.pid):
                return {"ok": False, "error": f"PID {req.pid} is not running"}
            await asyncio.to_thread(kill_process_tree, req.pid)
            return {"ok": True, "pid": req.pid, "signal": sig}
        else:
            proc = await asyncio.create_subprocess_exec(
                "kill",
                f"-{sig}",
                str(req.pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode != 0:
            err = (stderr.decode("utf-8", errors="replace") or "").strip()[:200]
            return {"ok": False, "error": err or f"kill returned {proc.returncode}"}
        return {"ok": True, "pid": req.pid, "signal": sig}
    except asyncio.TimeoutError:
        return {"ok": False, "error": "kill command timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ── /api/cookbook/hf-latest ──


@cookbook_router.get("/api/cookbook/hf-latest")
async def hf_latest(
    vram_gb: float = 0,
    limit: int = 10,
    pipeline: str = "text-generation",
    owner: str = Depends(require_user),
):
    """Fetch trending HuggingFace models filtered by available VRAM."""
    pool_size = max(limit * 15, 100)
    url = (
        "https://huggingface.co/api/models"
        f"?sort=trendingScore&direction=-1&limit={pool_size}&filter={pipeline}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {"models": [], "error": f"HF API HTTP {resp.status_code}"}
            raw = resp.json()
    except Exception as e:
        return {"models": [], "error": str(e)}

    def _est_vram_fp16(repo_id: str) -> float | None:
        m = re.search(r"[-_/](\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])", repo_id)
        return float(m.group(1)) * 2.0 if m else None

    def _quant_factor(repo_id: str, tags: list) -> float:
        text = (repo_id + " " + " ".join(tags or [])).lower()
        if any(k in text for k in ("fp4", "nf4", "int4", "4bit", "q4", "awq", "gptq")):
            return 0.25
        if any(k in text for k in ("int8", "8bit", "q8", "fp8")):
            return 0.5
        return 1.0

    EXCLUDE_TAGS = (
        "lora",
        "adapter",
        "peft",
        "qlora",
        "dataset",
        "embeddings",
        "merge",
        "control-lora",
        "diffusion-lora",
        "text-classification",
        "token-classification",
        "feature-extraction",
        "sentence-similarity",
    )
    EXCLUDE_NAMES = (
        "lora",
        "adapter",
        "peft",
        "qlora",
        "embedding",
        "embed-",
        "dataset",
    )

    def _is_excluded(repo_id: str, tags: list) -> bool:
        text = repo_id.lower()
        if any(s in text for s in EXCLUDE_NAMES):
            return True
        tag_text = " ".join(t.lower() for t in (tags or []))
        return any(s in tag_text for s in EXCLUDE_TAGS)

    out = []
    for entry in raw:
        repo_id = entry.get("modelId") or entry.get("id") or ""
        if not repo_id:
            continue
        tags = entry.get("tags") or []
        pipeline_tag = entry.get("pipeline_tag") or ""
        if pipeline and pipeline_tag and pipeline_tag != pipeline:
            continue
        if _is_excluded(repo_id, tags):
            continue
        est_fp16 = _est_vram_fp16(repo_id)
        quant_mult = _quant_factor(repo_id, tags)
        est_vram = (est_fp16 * quant_mult) if est_fp16 else None
        needed_vram = (est_vram * 1.3) if est_vram else None
        if vram_gb > 0 and needed_vram is not None and needed_vram > vram_gb:
            continue
        out.append(
            {
                "repo_id": repo_id,
                "downloads": entry.get("downloads", 0),
                "likes": entry.get("likes", 0),
                "createdAt": entry.get("createdAt", ""),
                "tags": tags[:5],
                "pipeline_tag": pipeline_tag,
                "est_vram_gb": round(est_vram, 1) if est_vram else None,
                "needed_vram_gb": round(needed_vram, 1) if needed_vram else None,
            }
        )
        if len(out) >= limit:
            break
    return {"models": out}
