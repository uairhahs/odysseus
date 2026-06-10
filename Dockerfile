FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# System deps. tmux is required by Cookbook for background downloads/serves.
# openssh-client is required for Cookbook remote server tests, setup, probes,
# downloads, and serves from Docker installs.
# git/cmake are required when Cookbook builds llama.cpp on first llama.cpp
# launch inside Docker.
# nodejs/npm provide npx for the optional built-in Browser MCP server.
# gosu lets the entrypoint drop privileges cleanly so signals still reach
# uvicorn directly (no extra shell layer like `su`/`sudo` would add).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    nodejs \
    npm \
    tmux \
    openssh-client \
    gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from pyproject.toml. Optional extras (PyMuPDF AGPL, etc.)
# are opt-in so the default image stays MIT-core.
# Copy pyproject.toml first for layer caching.
ARG INSTALL_OPTIONAL=false
COPY pyproject.toml uv.lock ./
RUN if [ "$INSTALL_OPTIONAL" = "true" ]; then \
      uv sync --all-extras --frozen; \
    else \
      uv sync --frozen; \
    fi

# Copy lockfiles before the rest of the source code to cache the npm layer
COPY package.json package-lock.json ./

# Use 'npm ci' for deterministic, lockfile-bound installation
# --omit=dev ensures we don't install unnecessary build-time tools in production
# IMPORTANT: Install Playwright browsers and OS dependencies.
RUN npm ci --omit=dev &&\
    npx playwright install --with-deps chromium

# Copy app code
COPY . .

# Verify the tool is available locally (--no-install prevents dynamic downloads)
# and create necessary data directories
RUN npx --no-install @playwright/mcp --version \
    && mkdir -p data logs services/cache/search

# Entrypoint that drops to PUID/PGID (default 1000:1000) and repairs
# ownership on the bind-mounted /app/data and /app/logs. Without this,
# the container runs as root and writes root-owned files into host
# bind mounts — any later non-root run (or a host user trying to
# update them) silently fails on EPERM, breaking skill extraction,
# prefs persistence, mail attachments, etc.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7000/api/ready', timeout=5).read()"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000", "--loop", "asyncio"]