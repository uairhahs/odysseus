import asyncio
import json
import logging

# import os
import re
import shlex
import shutil

# import subprocess
from pathlib import Path

from core.platform_compat import (  # ,NVIDIA_PATH_CANDIDATES,; SSH_PATH_OVERRIDE,; get_wsl_windows_user_profile,; kill_process_tree,; pid_alive,; safe_chmod,; translate_path,; which_tool,
    IS_WINDOWS,
)

# from routes.cookbook_helpers import run_ssh_command_async

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# --- SSH & Path Helpers ---


def get_cookbook_ssh_dir() -> Path:
    if not IS_WINDOWS:
        app_ssh = Path("/app/.ssh")
        if Path("/app").exists():
            return app_ssh
    return Path.home() / ".ssh"


def get_cookbook_ssh_key_path() -> Path:
    return get_cookbook_ssh_dir() / "id_ed25519"


def get_cookbook_known_hosts_path() -> Path:
    return get_cookbook_ssh_dir() / "known_hosts"


def read_cookbook_public_key() -> str:
    pub = get_cookbook_ssh_key_path().with_suffix(".pub")
    if not pub.exists():
        return ""
    return pub.read_text(encoding="utf-8", errors="replace").strip()


# --- Binary & Environment Checks ---


def needs_binary(cmd: str, binary: str) -> bool:
    return bool(re.search(rf"(^|[\s;&|()]){re.escape(binary)}($|[\s;&|()])", cmd or ""))


def missing_binary_message(binary: str, target: str) -> str:
    if binary == "tmux":
        return f"tmux is required for Cookbook background downloads/serves on {target}."
    if binary == "docker":
        return f"Docker is required by this Cookbook launch command on {target}."
    return f"{binary} is required on {target}, but it was not found."


async def remote_binary_available(
    remote: str, ssh_port: str | None, binary: str, windows: bool = False
) -> bool:
    port_arg = ssh_port or ""
    pf = ["-p", port_arg] if port_arg and port_arg != "22" else []
    if windows:
        check = f'powershell -NoProfile -Command "if (Get-Command {binary} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 127 }}"'
    else:
        check = f"command -v {shlex.quote(binary)} >/dev/null 2>&1"

    try:
        known_hosts = str(get_cookbook_known_hosts_path())
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o",
            "ConnectTimeout=6",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            *pf,
            remote,
            check,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


async def binary_available(
    binary: str, remote: str | None, ssh_port: str | None, windows: bool = False
) -> bool:
    if remote:
        return await remote_binary_available(remote, ssh_port, binary, windows=windows)
    return shutil.which(binary) is not None


# --- Secret Management ---


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "stored"
    return f"{value[:4]}...{value[-4:]}"


def decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    from src.secret_storage import decrypt

    return decrypt(value)


def encrypt_secret(value: str) -> str:
    from src.secret_storage import encrypt

    return encrypt(value)


def load_stored_hf_token(state_path: Path) -> str:
    if not state_path.exists():
        return ""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        env = state.get("env") if isinstance(state, dict) else {}
        return decrypt_secret(env.get("hfToken") if isinstance(env, dict) else "")
    except Exception:
        return ""
