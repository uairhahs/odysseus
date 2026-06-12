import shutil
import sys

from core.platform_compat import IS_WINDOWS
from routes.cookbook_helpers import (
    _append_llama_cpp_linux_accel_build_lines,
    _append_serve_exit_code_lines,
    _append_serve_preflight_exit_lines,
    _append_vllm_linux_preflight_lines,
    _bash_squote,
    _local_tooling_path_export,
    _ollama_bind_from_cmd,
    _pip_install_fallback_chain,
    _safe_env_prefix,
    _user_shell_path_bootstrap,
)

HF_TOKEN_STATUS_SNIPPET = (
    'if [ -n "${HF_TOKEN}" ]; then '  # noqa: S105
    'echo "[odysseus] HF token: applied"; '
    "else "
    'echo "[odysseus] HF token: NOT SET — gated/private models will be denied. '
    'Add one in Odysseus Settings -> Cookbook -> HuggingFace Token."; '
    "fi"
)


def build_bash_download_lines(
    req, session_id: str, hf_cmd: str, dl_pyarg: str, wrapper_script_path: str = None
) -> list[str]:
    """Generates the bash script for downloading models."""
    remote = req.remote_host
    lines = ["#!/bin/bash"]
    lines.extend(_user_shell_path_bootstrap())

    if remote:
        lines.append('ODYSSEUS_TMPDIR="${TMPDIR:-/tmp}"')
        lines.append('ODYSSEUS_LOG_DIR="$ODYSSEUS_TMPDIR/odysseus-tmux"')
        lines.append(f'ODYSSEUS_EXIT_FILE="$ODYSSEUS_LOG_DIR/{session_id}.exit"')
        lines.append('mkdir -p "$ODYSSEUS_LOG_DIR" 2>/dev/null || true')
        lines.append('rm -f "$ODYSSEUS_EXIT_FILE" 2>/dev/null || true')
        lines.append("# Auto-detect environment")
        lines.append("deactivate 2>/dev/null; hash -r")
        if req.hf_token:
            lines.append(f"export HF_TOKEN='{_bash_squote(req.hf_token)}'")
        if req.env_prefix:
            lines.append(_safe_env_prefix(req.env_prefix))
        else:
            lines.append(
                "for p in ~/vllm-env ~/venv ~/.venv; do "
                'if [ -f "$p/bin/activate" ]; then source "$p/bin/activate"; break; fi; '
                "done"
            )
        lines.append('export PATH="$HOME/.local/bin:$PATH"')
        lines.append(
            f"command -v hf >/dev/null 2>&1 || {_pip_install_fallback_chain('huggingface_hub', python_cmd='pip', upgrade=True)}"
        )

        if req.disable_hf_transfer:
            lines.append("export HF_HUB_ENABLE_HF_TRANSFER=0")
            lines.append("export HF_HUB_DOWNLOAD_MAX_WORKERS=4")
        else:
            lines.append(
                f"python3 -c 'import hf_transfer' 2>/dev/null || {_pip_install_fallback_chain('hf_transfer', python_cmd='pip')}"
            )
            lines.append(
                "python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1"
            )
            lines.append("export HF_HUB_DOWNLOAD_MAX_WORKERS=8")

        _max_workers = 4 if req.disable_hf_transfer else 8
        _local_dir_py = (
            ", local_dir=os.environ['_ODY_LOCAL_DIR']" if req.local_dir else ""
        )
        _py_dl = (
            f"python3 -c 'import os; from huggingface_hub import snapshot_download; "
            f'snapshot_download(os.environ["_ODY_REPO_ID"]{_local_dir_py}, max_workers={_max_workers})\''
        )
        lines.append(f"export _ODY_REPO_ID={_bash_squote(req.repo_id)}")
        if req.local_dir:
            lines.append(f"export _ODY_LOCAL_DIR={_bash_squote(req.local_dir)}")
        lines.append(HF_TOKEN_STATUS_SNIPPET)
        lines.append("if command -v hf &>/dev/null; then")
        lines.append(f"  {hf_cmd} < /dev/null")
        lines.append('elif python3 -c "import huggingface_hub" 2>/dev/null; then')
        lines.append('  echo "hf CLI not found, using Python huggingface_hub..."')
        lines.append(f"  {_py_dl}")
        lines.append("else")
        lines.append('  echo "Installing huggingface-hub and dependencies..."')
        lines.append("  pip install --no-deps -q huggingface-hub 2>/dev/null")
        if req.disable_hf_transfer:
            lines.append(
                "  pip install -q filelock fsspec packaging pyyaml tqdm typer httpx requests 2>/dev/null"
            )
            lines.append("  export HF_HUB_ENABLE_HF_TRANSFER=0")
        else:
            lines.append(
                "  pip install -q filelock fsspec packaging pyyaml tqdm typer httpx requests hf_transfer 2>/dev/null"
            )
            lines.append(
                "  python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1"
            )
        lines.append(f"  {_py_dl}")
        lines.append("fi")
        lines.append(
            '_ec=$?; if [ $_ec -eq 0 ]; then echo ""; echo "DOWNLOAD_OK"; else echo ""; echo "DOWNLOAD_FAILED (exit $_ec)"; fi'
        )
        lines.append('printf "%s\\n" "$_ec" > "$ODYSSEUS_EXIT_FILE"')
        lines.append(f"rm -f .{session_id}_run.sh")
        lines.append('exec "${SHELL:-/bin/bash}"')

    else:
        if req.hf_token:
            lines.append(f"export HF_TOKEN='{_bash_squote(req.hf_token)}'")
        lines.append('export PATH="$HOME/.local/bin:$PATH"')
        lines.append(_local_tooling_path_export(sys.executable))
        lines.append(
            f"command -v hf >/dev/null 2>&1 || {_pip_install_fallback_chain('huggingface_hub', upgrade=True)}"
        )

        _in_uv_venv = sys.prefix != sys.base_prefix and shutil.which("uv") is not None
        if req.disable_hf_transfer or _in_uv_venv:
            lines.append("export HF_HUB_ENABLE_HF_TRANSFER=0")
            lines.append("export HF_HUB_DOWNLOAD_MAX_WORKERS=4")
        else:
            lines.append(
                f"python3 -c 'import hf_transfer' 2>/dev/null || {_pip_install_fallback_chain('hf_transfer')}"
            )
            lines.append(
                "python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1"
            )
            lines.append("export HF_HUB_DOWNLOAD_MAX_WORKERS=8")

        if req.env_prefix:
            lines.append(_safe_env_prefix(req.env_prefix))
        else:
            lines.append("deactivate 2>/dev/null; hash -r")

        lines.append('ODYSSEUS_TMPDIR="${TMPDIR:-/tmp}"')
        lines.append('ODYSSEUS_LOG_DIR="$ODYSSEUS_TMPDIR/odysseus-tmux"')
        lines.append(f'ODYSSEUS_EXIT_FILE="$ODYSSEUS_LOG_DIR/{session_id}.exit"')
        lines.append('mkdir -p "$ODYSSEUS_LOG_DIR" 2>/dev/null || true')
        lines.append('rm -f "$ODYSSEUS_EXIT_FILE" 2>/dev/null || true')
        lines.append(HF_TOKEN_STATUS_SNIPPET)

        if IS_WINDOWS:
            lines.append(hf_cmd)
            lines.append(
                '_ec=$?; if [ $_ec -eq 0 ]; then echo ""; echo "DOWNLOAD_OK"; else echo ""; echo "DOWNLOAD_FAILED (exit $_ec)"; fi'
            )
            lines.append('printf "%s\\n" "$_ec" > "$ODYSSEUS_EXIT_FILE"')
        else:
            lines.append(f"{hf_cmd} < /dev/null")
            lines.append(
                '_ec=$?; if [ $_ec -eq 0 ]; then echo ""; echo "DOWNLOAD_OK"; else echo ""; echo "DOWNLOAD_FAILED (exit $_ec)"; fi'
            )
            lines.append('printf "%s\\n" "$_ec" > "$ODYSSEUS_EXIT_FILE"')
            if wrapper_script_path:
                lines.append(f"rm -f '{wrapper_script_path}'")
            lines.append('exec "${SHELL:-/bin/bash}"')

    return lines


def build_bash_serve_lines(
    req, session_id: str, is_pip_install: bool, local_windows: bool, remote: str
) -> list[str]:
    """Generates the bash wrapper script for serving models."""
    runner_lines = ["#!/bin/bash"]
    runner_lines.append('ODYSSEUS_TMPDIR="${TMPDIR:-/tmp}"')
    runner_lines.append('ODYSSEUS_LOG_DIR="$ODYSSEUS_TMPDIR/odysseus-tmux"')
    runner_lines.append('mkdir -p "$ODYSSEUS_LOG_DIR" 2>/dev/null || true')
    runner_lines.append(f'ODYSSEUS_EXIT_FILE="$ODYSSEUS_LOG_DIR/{session_id}.exit"')
    runner_lines.append('rm -f "$ODYSSEUS_EXIT_FILE" 2>/dev/null || true')
    runner_lines.append("exec 3>&1 4>&2")
    runner_lines.append(f'exec > >(tee -a "$ODYSSEUS_LOG_DIR/{session_id}.log") 2>&1')
    runner_lines.extend(_user_shell_path_bootstrap())

    if not remote:
        runner_lines.append(_local_tooling_path_export(sys.executable))

    runner_lines.append("export FLASHINFER_DISABLE_VERSION_CHECK=1")
    if req.hf_token:
        runner_lines.append(f"export HF_TOKEN='{_bash_squote(req.hf_token)}'")
    if req.gpus:
        runner_lines.append(f"export CUDA_VISIBLE_DEVICES='{req.gpus}'")
    if req.env_prefix:
        runner_lines.append(_safe_env_prefix(req.env_prefix))
    else:
        runner_lines.append("deactivate 2>/dev/null; hash -r")

    runner_lines.append(HF_TOKEN_STATUS_SNIPPET)
    handled_ollama_serve = False

    if "llama_cpp" in req.cmd or "llama-server" in req.cmd:
        runner_lines.append("# Ensure a llama.cpp server (prefer native llama-server)")
        runner_lines.append(
            'export PATH="$HOME/.local/bin:$HOME/bin:$HOME/llama.cpp/build/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"'
        )

        if local_windows:
            runner_lines.append(
                'if ! command -v llama-server &>/dev/null && ! python -c "import llama_cpp" 2>/dev/null; then'
            )
            runner_lines.append(
                '  echo "llama-server not found — installing Python bindings..."'
            )
            runner_lines.append(
                f"  {_pip_install_fallback_chain('llama-cpp-python[server]', python_cmd='python')} || true"
            )
            runner_lines.append("fi")
            runner_lines.append(
                'if ! command -v llama-server &>/dev/null && ! python -c "import llama_cpp" 2>/dev/null; then'
            )
            runner_lines.append(
                '  echo "ERROR: llama.cpp serving is not available after install attempts."'
            )
            runner_lines.append("  ODYSSEUS_PREFLIGHT_EXIT=127")
            runner_lines.append("fi")
        else:
            runner_lines.append("if [ -d /data/data/com.termux ]; then")
            runner_lines.append(
                '  if ! python3 -c "import llama_cpp" 2>/dev/null; then'
            )
            runner_lines.append("    pkg install -y cmake 2>/dev/null")
            runner_lines.append("    pip install numpy diskcache jinja2 2>/dev/null")
            runner_lines.append(
                "    CMAKE_ARGS=\"-DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF\" pip install 'llama-cpp-python[server]' --no-build-isolation --no-cache-dir 2>&1 || true"
            )
            runner_lines.append("  fi")
            runner_lines.append("elif ! command -v llama-server &>/dev/null; then")
            runner_lines.append(
                '  echo "Native llama-server not found — building from source (one-time, may take a few minutes)..."'
            )
            runner_lines.append("  mkdir -p ~/bin")
            runner_lines.append(
                "  cd ~ && [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp"
            )
            runner_lines.append(
                '  NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"'
            )
            runner_lines.append('  if [ "$(uname -s)" = "Darwin" ]; then')
            runner_lines.append(
                '    command -v cmake >/dev/null 2>&1 || echo "WARNING: cmake not found — install it with: brew install cmake (or: brew install llama.cpp for a prebuilt llama-server)."'
            )
            runner_lines.append(
                "    cd ~/llama.cpp && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release \\"
            )
            runner_lines.append(
                '      && cmake --build build -j"$NPROC" --target llama-server \\'
            )
            runner_lines.append(
                "      && ln -sf ~/llama.cpp/build/bin/llama-server ~/bin/llama-server"
            )
            runner_lines.append("  else")
            _append_llama_cpp_linux_accel_build_lines(runner_lines)
            runner_lines.append("  fi")
            runner_lines.append(
                '  if ! command -v llama-server &>/dev/null && ! python3 -c "import llama_cpp" 2>/dev/null; then'
            )
            runner_lines.append(
                '    echo "llama-server build failed — installing Python bindings as fallback..."'
            )
            runner_lines.append(
                f"    {_pip_install_fallback_chain('llama-cpp-python[server]', python_cmd='pip')} || true"
            )
            runner_lines.append("  fi")
            runner_lines.append(
                '  if ! command -v llama-server &>/dev/null && ! python3 -c "import llama_cpp" 2>/dev/null; then'
            )
            runner_lines.append(
                '    echo "ERROR: llama.cpp serving is not available after install/build attempts."'
            )
            runner_lines.append("    ODYSSEUS_PREFLIGHT_EXIT=127")
            runner_lines.append("  fi")
            runner_lines.append("fi")

    elif "ollama" in req.cmd:
        handled_ollama_serve = True
        _ollama_default_host = "0.0.0.0" if remote else "127.0.0.1"  # noqa: S104
        _ollama_host, _ollama_port = _ollama_bind_from_cmd(
            req.cmd, default_host=_ollama_default_host
        )

        runner_lines.append(f"ODYSSEUS_OLLAMA_HOST={_bash_squote(_ollama_host)}")
        runner_lines.append(f'ODYSSEUS_OLLAMA_PORT="{_ollama_port}"')
        runner_lines.append("for _ody_off in 0 1 2 3 4 5 6 7 8 9; do")
        runner_lines.append("  _ody_try_port=$((ODYSSEUS_OLLAMA_PORT + _ody_off))")
        runner_lines.append(
            "  if ! (exec 3<>/dev/tcp/127.0.0.1/$_ody_try_port) 2>/dev/null; then"
        )
        runner_lines.append("    exec 3<&-; exec 3>&-")
        runner_lines.append('    ODYSSEUS_OLLAMA_PORT="$_ody_try_port"')
        runner_lines.append("    break")
        runner_lines.append("  fi")
        runner_lines.append("done")
        runner_lines.append(
            "if (exec 3<>/dev/tcp/127.0.0.1/$ODYSSEUS_OLLAMA_PORT) 2>/dev/null; then"
        )
        runner_lines.append("  exec 3<&-; exec 3>&-")
        runner_lines.append(
            '  echo "[odysseus] Ollama API ready on port ${ODYSSEUS_OLLAMA_PORT}: http://${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT}"'
        )
        runner_lines.append(
            '  echo "[odysseus] This task is monitoring an existing Ollama server; stopping it here will not stop an external Docker/system service."'
        )

        if local_windows:
            runner_lines.append("  while true; do sleep 60; done")
        else:
            runner_lines.append("  exec bash -i")

        runner_lines.append("fi")
        runner_lines.append("if ! command -v ollama &>/dev/null; then")
        runner_lines.append(
            '  echo "ERROR: Ollama not found on this server. Install it from https://ollama.com/download or `curl -fsSL https://ollama.com/install.sh | sh`."'
        )
        runner_lines.append("  echo")
        runner_lines.append('  echo "=== Process exited with code 127 ==="')

        if local_windows:
            runner_lines.append("  exit 127")
        else:
            runner_lines.append("  exec bash -i")

        runner_lines.append("fi")
        runner_lines.append(
            'ODYSSEUS_OLLAMA_URL="http://${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT}"'
        )

        if remote and _ollama_host in ("0.0.0.0", "::"):  # noqa: S104
            runner_lines.append(
                'echo "[odysseus] WARNING: remote Ollama will bind to ${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT} so Odysseus can reach it from this host."'
            )
            runner_lines.append(
                'echo "[odysseus] Ollama has no built-in authentication; expose this only on a trusted LAN/VPN or provide an explicit OLLAMA_HOST with your own access controls."'
            )

        runner_lines.append(
            'echo "Starting ollama server on ${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT}..."'
        )
        runner_lines.append(
            'OLLAMA_HOST="${ODYSSEUS_OLLAMA_HOST}:${ODYSSEUS_OLLAMA_PORT}" ollama serve'
        )

        if local_windows:
            _append_serve_exit_code_lines(runner_lines, keep_shell_open=False)
        else:
            _append_serve_exit_code_lines(runner_lines, keep_shell_open=True)

    elif "vllm serve" in req.cmd:
        runner_lines.append('if [ "$(uname -s)" = "Darwin" ]; then')
        runner_lines.append(
            '  echo "ERROR: vLLM does not run on macOS. Use Ollama or llama.cpp (Metal) instead."'
        )
        runner_lines.append("  ODYSSEUS_PREFLIGHT_EXIT=1")
        runner_lines.append("fi")
        _append_vllm_linux_preflight_lines(runner_lines)

    elif "sglang.launch_server" in req.cmd:
        runner_lines.append('export PATH="$HOME/.local/bin:$PATH"')
        runner_lines.append("if ! command -v sglang &>/dev/null; then")
        runner_lines.append('  echo "ERROR: SGLang is not installed."')
        runner_lines.append("  ODYSSEUS_PREFLIGHT_EXIT=127")
        runner_lines.append(
            'elif ! ODYSSEUS_SGLANG_IMPORT_ERROR="$(python3 -c "import sglang" 2>&1)"; then'
        )
        runner_lines.append('  echo "ERROR: SGLang is installed but failed to import."')
        runner_lines.append('  printf "%s\\n" "$ODYSSEUS_SGLANG_IMPORT_ERROR"')
        runner_lines.append("  ODYSSEUS_PREFLIGHT_EXIT=127")
        runner_lines.append("fi")

    elif "scripts/diffusion_server.py" in req.cmd or ".diffusion_server.py" in req.cmd:
        runner_lines.append('export PATH="$HOME/.local/bin:$PATH"')
        runner_lines.append(
            'if ! ODYSSEUS_DIFFUSION_IMPORT_ERROR="$(python3 -c "import torch, diffusers" 2>&1)"; then'
        )
        runner_lines.append(
            '  echo "ERROR: Diffusion serving requires PyTorch + diffusers."'
        )
        runner_lines.append('  printf "%s\\n" "$ODYSSEUS_DIFFUSION_IMPORT_ERROR"')
        runner_lines.append("  ODYSSEUS_PREFLIGHT_EXIT=127")
        runner_lines.append("fi")

    if not handled_ollama_serve:
        _append_serve_preflight_exit_lines(
            runner_lines, keep_shell_open=not local_windows
        )
        runner_lines.append(req.cmd)
        if local_windows:
            _append_serve_exit_code_lines(
                runner_lines, keep_shell_open=False, is_pip_install=is_pip_install
            )
        else:
            _append_serve_exit_code_lines(
                runner_lines, keep_shell_open=True, is_pip_install=is_pip_install
            )

    return runner_lines
