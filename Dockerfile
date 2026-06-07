FROM python:3.12-slim

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

# Install uv for fast, reliable Python dependency management.
# Symlink into /usr/local/bin so the path is stable regardless of which
# directory the installer chooses (~/.local/bin vs ~/.cargo/bin).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && (ln -sf /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || \
        ln -sf /root/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null)

ENV PATH="/usr/local/bin:/root/.local/bin:/root/.cargo/bin:$PATH"

# Install Python deps from pyproject.toml. Optional extras (PyMuPDF AGPL, etc.)
# are opt-in so the default image stays MIT-core. Use --no-cache-dir to minimize
# image size. --system is required inside Docker (no virtualenv active).
# Copy pyproject.toml first for layer caching.
ARG INSTALL_OPTIONAL=false
COPY pyproject.toml uv.lock ./
RUN uv pip install --no-cache-dir --system -r pyproject.toml \
    && if [ "$INSTALL_OPTIONAL" = "true" ]; then \
         uv pip install --no-cache-dir --system "pyproject.toml[optional]"; \
       fi

# Copy app code
COPY . .

# Create data directory (mount a volume here for persistence)
RUN mkdir -p data logs services/cache/search

# Entrypoint that drops to PUID/PGID (default 1000:1000) and repairs
# ownership on the bind-mounted /app/data and /app/logs. Without this,
# the container runs as root and writes root-owned files into host
# bind mounts — any later non-root run (or a host user trying to
# update them) silently fails on EPERM, breaking skill extraction,
# prefs persistence, mail attachments, etc.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000", "--loop", "asyncio"]