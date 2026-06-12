from routes.cookbook_helpers import _ps_squote, _safe_env_prefix


def build_ps1_download_lines(
    req, session_id: str, hf_cmd: str, dl_pyarg: str
) -> list[str]:
    """Generates the PowerShell script for downloading models."""
    ps_lines = []
    ps_lines.append('$sessionDir = "$env:TEMP\\odysseus-sessions"')
    ps_lines.append("New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null")
    ps_lines.append(f'$exitPath = "$sessionDir\\{session_id}.exit"')
    ps_lines.append("Remove-Item -Force $exitPath -ErrorAction SilentlyContinue")
    ps_lines.append('$env:PYTHONIOENCODING = "utf-8"')
    ps_lines.append('$env:PYTHONUTF8 = "1"')

    if req.hf_token:
        ps_lines.append(f"$env:HF_TOKEN = '{_ps_squote(req.hf_token)}'")
    if req.env_prefix:
        ps_lines.append(_safe_env_prefix(req.env_prefix))

    ps_lines.append("try {")
    ps_lines.append("  $hfPath = Get-Command hf -ErrorAction SilentlyContinue")
    ps_lines.append("  if ($hfPath) {")
    ps_lines.append(f"    $null | {hf_cmd}")
    ps_lines.append("  } else {")
    ps_lines.append('    python -c "import huggingface_hub" 2>$null')
    ps_lines.append("    if ($LASTEXITCODE -eq 0) {")
    ps_lines.append(
        '      Write-Host "hf CLI not found, using Python huggingface_hub..."'
    )
    ps_lines.append("      python -m pip install -q hf_transfer 2>$null")
    ps_lines.append('      $env:HF_HUB_ENABLE_HF_TRANSFER = "1"')
    ps_lines.append(
        f"      python -c \"import os; from huggingface_hub import snapshot_download; snapshot_download('{req.repo_id}'{dl_pyarg}, max_workers=8)\""
    )
    ps_lines.append("    } else {")
    ps_lines.append('      Write-Host "Installing huggingface-hub..."')
    ps_lines.append("      python -m pip install -q huggingface-hub hf_transfer")
    ps_lines.append('      $env:HF_HUB_ENABLE_HF_TRANSFER = "1"')
    ps_lines.append(
        f"      python -c \"import os; from huggingface_hub import snapshot_download; snapshot_download('{req.repo_id}'{dl_pyarg}, max_workers=8)\""
    )
    ps_lines.append("    }")
    ps_lines.append("  }")
    ps_lines.append(
        '  if ($LASTEXITCODE -eq 0) { Write-Host ""; Write-Host "DOWNLOAD_OK" }'
    )
    ps_lines.append(
        '  else { Write-Host ""; Write-Host "DOWNLOAD_FAILED (exit $LASTEXITCODE)" }'
    )
    ps_lines.append('  "$LASTEXITCODE" | Out-File -Encoding ascii $exitPath')
    ps_lines.append("} catch {")
    ps_lines.append('  Write-Host ""; Write-Host "DOWNLOAD_FAILED ($_)"')
    ps_lines.append('  "1" | Out-File -Encoding ascii $exitPath')
    ps_lines.append("}")

    return ps_lines


def build_ps1_serve_lines(req, session_id: str, is_pip_install: bool) -> list[str]:
    """Generates the PowerShell wrapper script for serving models."""
    ps_lines = []
    ps_lines.append('$sessionDir = "$env:TEMP\\odysseus-sessions"')
    ps_lines.append("New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null")
    ps_lines.append(f'$exitPath = "$sessionDir\\{session_id}.exit"')
    ps_lines.append("Remove-Item -Force $exitPath -ErrorAction SilentlyContinue")
    ps_lines.append('$env:PYTHONIOENCODING = "utf-8"')
    ps_lines.append('$env:PYTHONUTF8 = "1"')

    if req.hf_token:
        ps_lines.append(f"$env:HF_TOKEN = '{_ps_squote(req.hf_token)}'")
    if req.gpus:
        ps_lines.append(f"$env:CUDA_VISIBLE_DEVICES = '{req.gpus}'")
    if req.env_prefix:
        ps_lines.append(_safe_env_prefix(req.env_prefix))

    if "ollama" in req.cmd:
        ps_lines.append("# Check if ollama is available")
        ps_lines.append(
            "if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {"
        )
        ps_lines.append(
            '  Write-Host "Ollama not found. Please install from https://ollama.com/download/windows"'
        )
        ps_lines.append('  Write-Host ""')
        ps_lines.append('  Write-Host "=== Process exited with code 1 ==="')
        ps_lines.append('  "1" | Out-File -Encoding ascii $exitPath')
        ps_lines.append("  exit 1")
        ps_lines.append("}")
    elif "llama_cpp" in req.cmd or "llama-server" in req.cmd:
        ps_lines.append("# Auto-install llama-cpp-python if missing")
        ps_lines.append('try { python -c "import llama_cpp" 2>$null } catch {}')
        ps_lines.append("if ($LASTEXITCODE -ne 0) {")
        ps_lines.append('  Write-Host "Installing llama-cpp-python..."')
        ps_lines.append(
            "  python -m pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        )
        ps_lines.append("}")
    elif "vllm" in req.cmd:
        ps_lines.append(
            'Write-Host "ERROR: vLLM is not supported on Windows. Use Ollama or llama.cpp instead."'
        )
        ps_lines.append('Write-Host ""')
        ps_lines.append('Write-Host "=== Process exited with code 1 ==="')
        ps_lines.append('"1" | Out-File -Encoding ascii $exitPath')
        ps_lines.append("exit 1")

    if is_pip_install:
        ps_lines.append(req.cmd)
        ps_lines.append(
            'if ($LASTEXITCODE -eq 0) { Write-Host ""; Write-Host "DOWNLOAD_OK" }'
        )
        ps_lines.append('Write-Host ""')
        ps_lines.append('Write-Host "=== Process exited with code $LASTEXITCODE ==="')
        ps_lines.append('"$LASTEXITCODE" | Out-File -Encoding ascii $exitPath')
    else:
        # Serve: run the command inline; the script file itself is launched
        # detached by the caller (Start-Process -WindowStyle Hidden), so env
        # vars set above are inherited and the API returns immediately.
        ps_lines.append(req.cmd)
        ps_lines.append('Write-Host ""')
        ps_lines.append('Write-Host "=== Process exited with code $LASTEXITCODE ==="')
        ps_lines.append('"$LASTEXITCODE" | Out-File -Encoding ascii $exitPath')

    return ps_lines
