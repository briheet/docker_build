# syntax=docker/dockerfile:1.7
# SDXL-Turbo Worker: Python 3.11 + PyTorch + CUDA 12.6

ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS cozy_base

WORKDIR /app

ENV UV_CACHE_DIR=/var/cache/uv
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=300

# Git is required for direct VCS dependencies
RUN --mount=type=cache,id=cozy-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=cozy-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean

# Install stable shared runtime layers first for better cache reuse.
ARG UV_TORCH_BACKEND=cu126
ARG TORCH_SPEC="~=2.5"

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --torch-backend ${UV_TORCH_BACKEND} \
    "torch${TORCH_SPEC}"

# Optional: vendor local copy of gen-worker for development
COPY .cozy/vendor/gen-worker /opt/gen-worker

RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    if [ -f /opt/gen-worker/pyproject.toml ]; then \
      uv pip install --system --break-system-packages /opt/gen-worker; \
    else \
      uv pip install --system --break-system-packages "gen-worker>=0.2.1"; \
    fi

# Install shared dependencies
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages \
      "safetensors>=0.7.0" \
      "flashpack>=0.2.1" \
      "numpy>=2.0.0"

# Final stage: app-specific deps + code
FROM cozy_base

# Copy pyproject.toml and lock file (if exists)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies into global site-packages (no project venv).
# Exclude torch + gen-worker since they're already installed in base layer.
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv lock && \
    uv export --no-dev --no-hashes --no-sources --no-emit-project \
      --no-emit-package torch --no-emit-package gen-worker \
      -o /tmp/requirements.all.txt \
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.all.txt > /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-deps -r /tmp/requirements.txt

# Copy application code late so app edits only invalidate the final layers
COPY . /app

# Install the project as a package
RUN --mount=type=cache,id=cozy-uv-cache,target=/var/cache/uv,sharing=locked \
    uv pip install --system --break-system-packages --no-deps --no-sources /app

# Generate function manifest at build time using gen_worker.discover
RUN mkdir -p /app/.cozy \
    && python -m gen_worker.discover > /app/.cozy/manifest.json \
    && echo "Manifest:" && cat /app/.cozy/manifest.json

# Run as non-root at runtime
RUN groupadd --system --gid 10001 cozy \
    && useradd --system --uid 10001 --gid cozy --create-home --home-dir /home/cozy --shell /usr/sbin/nologin cozy \
    && chown -R cozy:cozy /app /home/cozy

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HOME=/home/cozy
ENV XDG_CACHE_HOME=/home/cozy/.cache
ENV HF_HOME=/home/cozy/.cache/huggingface

USER cozy:cozy

# Default command - runs gen-worker entrypoint
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
