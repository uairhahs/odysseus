import asyncio

# import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from core.platform_compat import (  # ,NVIDIA_PATH_CANDIDATES,; SSH_PATH_OVERRIDE,; get_wsl_windows_user_profile,; kill_process_tree,; pid_alive,; safe_chmod,; translate_path,; which_tool,
    IS_WINDOWS,
    detached_popen_kwargs,
    find_bash,
)
from routes.cookbook_helpers import (
    _validate_token,
)
from routes.shell_routes import TMUX_LOG_DIR

# from routes.cookbook_helpers import run_ssh_command_async
from src.constants import COOKBOOK_STATE_FILE

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


_cookbook_state_path = Path(COOKBOOK_STATE_FILE)


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "stored"
    return f"{value[:4]}...{value[-4:]}"


def _decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    from src.secret_storage import decrypt

    return decrypt(value)


def _encrypt_secret(value: str) -> str:
    from src.secret_storage import encrypt

    return encrypt(value)


def _strip_task_secrets(state):
    tasks = state.get("tasks") if isinstance(state, dict) else None
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, dict) and isinstance(task.get("payload"), dict):
                task["payload"].pop("hf_token", None)
    return state


def _diagnose_serve_output(text: str) -> dict | None:
    """Server-side mirror of the Cookbook UI's common serve diagnoses.
    The browser uses cookbook-diagnosis.js for clickable fixes. This gives
    the agent/tool path the same structured signal so it can retry with an
    adjusted command instead of guessing from raw tmux output.
    """
    if not text:
        return None
    tail = text[-6000:]
    patterns = [
        (
            r"No available memory for the cache blocks|Available KV cache memory:.*-",
            "No GPU memory left for KV cache after loading model.",
            [
                {
                    "label": "retry with GPU memory utilization 0.95",
                    "op": "replace",
                    "flag": "--gpu-memory-utilization",
                    "value": "0.95",
                },
                {
                    "label": "retry with context 2048",
                    "op": "replace",
                    "flag": "--max-model-len",
                    "value": "2048",
                },
            ],
        ),
        (
            r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUDA error: out of memory|warming up sampler|max_num_seqs.*gpu_memory_utilization",
            "GPU ran out of memory during startup or warmup.",
            [
                {
                    "label": "retry with context 4096",
                    "op": "replace",
                    "flag": "--max-model-len",
                    "value": "4096",
                },
                {
                    "label": "retry with GPU memory utilization 0.80",
                    "op": "replace",
                    "flag": "--gpu-memory-utilization",
                    "value": "0.80",
                },
                {
                    "label": "retry with --enforce-eager",
                    "op": "append",
                    "arg": "--enforce-eager",
                },
            ],
        ),
        (
            r"not divisib|must be divisible|attention heads.*divisible",
            "Tensor parallel size is incompatible with the model.",
            [
                {
                    "label": "retry with tensor parallel size 1",
                    "op": "replace",
                    "flag": "--tensor-parallel-size",
                    "value": "1",
                },
                {
                    "label": "retry with tensor parallel size 2",
                    "op": "replace",
                    "flag": "--tensor-parallel-size",
                    "value": "2",
                },
            ],
        ),
        (
            r"KV cache.*too (small|large)|max_model_len.*exceeds|maximum.*context",
            "Context length is too large for available GPU memory.",
            [
                {
                    "label": "retry with context 8192",
                    "op": "replace",
                    "flag": "--max-model-len",
                    "value": "8192",
                },
                {
                    "label": "retry with context 4096",
                    "op": "replace",
                    "flag": "--max-model-len",
                    "value": "4096",
                },
            ],
        ),
        (
            r"enable-auto-tool-choice requires --tool-call-parser",
            "Auto tool choice requires an explicit tool call parser.",
            [
                {
                    "label": "retry with Hermes tool parser",
                    "op": "append",
                    "arg": "--tool-call-parser hermes",
                }
            ],
        ),
        (
            r"Please pass.*trust.remote.code=True|contains custom code which must be executed to correctly load|does not recognize this architecture|model type.*but Transformers does not",
            "Model requires custom code or newer model support.",
            [
                {
                    "label": "retry with --trust-remote-code",
                    "op": "append",
                    "arg": "--trust-remote-code",
                }
            ],
        ),
        (
            r"Either a revision or a version must be specified|transformers\.integrations\.hub_kernels|kernels/layer",
            "vLLM/Transformers kernel package mismatch.",
            [
                {
                    "label": "update vLLM, Transformers, and kernels on this server",
                    "op": "dependency",
                    "package": "vllm transformers kernels",
                }
            ],
        ),
        (
            r"Address already in use|bind.*address.*in use",
            "Port is already in use.",
            [
                {
                    "label": "retry on port 8001",
                    "op": "replace",
                    "flag": "--port",
                    "value": "8001",
                }
            ],
        ),
        (
            r"No CUDA GPUs are available|no GPU.*found|CUDA_VISIBLE_DEVICES.*invalid",
            "No GPUs are visible to the serve process.",
            [
                {
                    "label": "clear Cookbook GPU selection or choose available GPUs",
                    "op": "settings",
                    "field": "gpus",
                    "value": "",
                }
            ],
        ),
        (
            r"Failed to infer device type|NVML Shared Library Not Found|No module named 'amdsmi'|platform is not available",
            "vLLM could not find a supported GPU (CUDA or ROCm). "
            "This machine may have integrated or unsupported graphics only.",
            [
                {
                    "label": "switch to llama.cpp (CPU/Metal, works without a discrete GPU)",
                    "op": "manual",
                },
                {
                    "label": "switch to Ollama (CPU/Metal, works without a discrete GPU)",
                    "op": "manual",
                },
            ],
        ),
        (
            r"vllm.*command not found|No module named vllm|ERROR: vLLM is not installed",
            "vLLM is not installed or not in PATH on this server.",
            [
                {
                    "label": "install vLLM in Cookbook Dependencies",
                    "op": "dependency",
                    "package": "vllm",
                }
            ],
        ),
        (
            r"sglang.*command not found|No module named sglang|SGLang is not installed",
            "SGLang is not installed or not in PATH on this server.",
            [
                {
                    "label": "install SGLang in Cookbook Dependencies",
                    "op": "dependency",
                    "package": "sglang[all]",
                }
            ],
        ),
        (
            r"llama-server.*command not found|llama\.cpp.*not found|No module named.*llama_cpp|No module named 'starlette_context'|git: command not found|cmake: command not found",
            "llama.cpp / llama-cpp-python dependencies are missing.",
            [
                {
                    "label": "install llama.cpp dependencies or llama-cpp-python[server]",
                    "op": "dependency",
                    "package": "llama-cpp-python[server]",
                }
            ],
        ),
        (
            r"No GGUF found on this host|no \.gguf file|No GGUF file found",
            "No GGUF file found for this model on this host. The llama.cpp backend needs a .gguf file.",
            [
                {
                    "label": "download a GGUF build of this model (repo name usually ends in -GGUF, file like Q4_K_M.gguf)",
                    "op": "manual",
                }
            ],
        ),
        (
            r"No module named 'torch'|No module named torch|No module named 'diffusers'|No module named diffusers",
            "Diffusion serving requires PyTorch and diffusers.",
            [
                {
                    "label": "install diffusers[torch] in Cookbook Dependencies",
                    "op": "dependency",
                    "package": "diffusers[torch]",
                }
            ],
        ),
        (
            r"403 Forbidden|401 Unauthorized|Access to model.*is restricted|gated repo|not in the authorized list|awaiting a review",
            "Model access is gated or unauthorized.",
            [
                {
                    "label": "set HF token and request model access on HuggingFace",
                    "op": "manual",
                }
            ],
        ),
    ]
    for pattern, message, suggestions in patterns:
        if re.search(pattern, tail, re.I):
            return {"message": message, "suggestions": suggestions}
    if re.search(r"Traceback \(most recent call last\)", tail, re.I) and not re.search(
        r"Application startup complete|GET /v1/|Uvicorn running on", tail, re.I
    ):
        return {
            "message": "Python traceback detected during serve startup.",
            "suggestions": [
                {
                    "label": "inspect traceback and retry with adjusted backend/settings",
                    "op": "manual",
                }
            ],
        }
    return None


def _state_for_client(state):
    """Return cookbook state without raw secrets for browser clients."""
    _strip_task_secrets(state)
    env = state.get("env") if isinstance(state, dict) else None
    if isinstance(env, dict):
        token = _decrypt_secret(env.get("hfToken"))
        env.pop("hfToken", None)
        env["hfTokenConfigured"] = bool(token)
        env["hfTokenMasked"] = _mask_secret(token)
    return state


def _state_for_storage(state, on_disk=None):
    """Encrypt cookbook secrets before writing state to disk."""
    _strip_task_secrets(state)
    env = state.get("env") if isinstance(state, dict) else None
    disk_env = (
        on_disk.get("env")
        if isinstance(on_disk, dict) and isinstance(on_disk.get("env"), dict)
        else {}
    )
    if isinstance(env, dict):
        incoming = env.get("hfToken")
        if incoming:
            _validate_token(incoming)
            env["hfToken"] = _encrypt_secret(incoming)
        elif disk_env.get("hfToken"):
            env["hfToken"] = disk_env["hfToken"]
        else:
            env.pop("hfToken", None)
        env.pop("hfTokenMasked", None)
        env.pop("hfTokenConfigured", None)
    return state


def _launch_local_detached(session_id: str, bash_lines: list[str]) -> dict:
    """Windows-native stand-in for a LOCAL tmux session (tmux doesn't exist
    on Windows). Mirrors shell_routes._generate_win_detached / bg_jobs.launch:
    runs the wrapper detached so it survives a browser/SSE disconnect (the
    whole point of the tmux feature for long downloads/serves), writing a
    <session>.log the status poller tails and a <session>.pid for liveness.
    `bash_lines` is the same bash wrapper used on POSIX. Prefers Git Bash
    for full command-syntax parity; falls back to a cmd.exe wrapper that
    runs the script through whatever bash is reachable, else best-effort
    directly (simple commands only). Returns the launched job record."""
    log_path = TMUX_LOG_DIR / f"{session_id}.log"
    pid_path = TMUX_LOG_DIR / f"{session_id}.pid"
    bash = find_bash()
    if bash:
        # Run the existing bash wrapper verbatim through Git Bash, redirecting
        # all output to the log the poller reads. Paths handed to bash use
        # POSIX form + shell-quoting so drive paths / spaces survive.
        inner = TMUX_LOG_DIR / f"{session_id}_run.sh"
        inner.write_text("\n".join(bash_lines) + "\n", encoding="utf-8")
        lp = shlex.quote(log_path.as_posix())
        ip = shlex.quote(inner.as_posix())
        script_path = TMUX_LOG_DIR / f"{session_id}.sh"
        script_path.write_text(
            f"bash {ip} > {lp} 2>&1\n",
            encoding="utf-8",
        )
        argv = [bash, str(script_path)]
    else:
        # No bash on this Windows host: the bash wrapper can't run. Fall back
        # to a cmd.exe wrapper that just records a clear error to the log so
        # the UI surfaces "install Git Bash" instead of silently hanging.
        script_path = TMUX_LOG_DIR / f"{session_id}.cmd"
        script_path.write_text(
            "@echo off\r\n"
            f'echo Cookbook LOCAL execution on Windows needs Git Bash ^(bash.exe^) on PATH. > "{log_path}" 2>&1\r\n'
            f'echo Install Git for Windows, then retry. >> "{log_path}"\r\n',
            encoding="utf-8",
        )
        argv = [os.environ.get("ComSpec", "cmd.exe"), "/c", str(script_path)]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=env,
        **detached_popen_kwargs(),
    )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return {"pid": proc.pid, "log_path": str(log_path)}
