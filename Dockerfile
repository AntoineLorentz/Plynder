ARG DEBIAN_DISTRIB=trixie
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_DISTRIB}

# ── CUDA ──────────────────────────────────────────────────────────────────────
COPY --from=nvcr.io/nvidia/cuda-dl-base:26.05-cuda13.2-devel-ubuntu24.04 /usr/local/cuda-13.2 /usr/local/cuda-13.2
RUN ln -s /usr/local/cuda-13.2 /usr/local/cuda-13 && ln -s /usr/local/cuda-13.2 /usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda-13.2"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ── System packages ───────────────────────────────────────────────────────────
RUN sed -i 's/Components: main/Components: main contrib non-free/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
      libaio-dev \
      curl \
      unzip \
      gnupg2 \
      procps \
      less \
      sudo \
      openssh-client \
      supervisor \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Rust toolchain ────────────────────────────────────────────────────────────
COPY --from=rust:1.89-bookworm /usr/local/cargo /usr/local/cargo
COPY --from=rust:1.89-bookworm /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}"

# ── uv (package manager) ──────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

# ── Non-root user ─────────────────────────────────────────────────────────────
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} vscode \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m vscode \
    && echo "vscode ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode

# ── Rust + uv environment ─────────────────────────────────────────────────────
ENV UV_PROJECT_ENVIRONMENT="/home/vscode/venv/.venv"
ENV UV_LINK_MODE="copy"
RUN rustup toolchain link local /usr/local/rustup/toolchains/1.89.0-x86_64-unknown-linux-gnu && \
    rustup default local
RUN uv generate-shell-completion bash >> /home/vscode/.bashrc

# ── Project build ─────────────────────────────────────────────────────────────
WORKDIR /workspaces/
COPY uv.lock pyproject.toml .python-version ./
COPY --chown=vscode:vscode rust_extension/ ./rust_extension/
RUN mkdir -p src/plynder && \
    touch src/plynder/__init__.py README.md && \
    uv sync --frozen && \
    rm -rf /home/vscode/.cache

CMD ["sleep", "infinity"]
