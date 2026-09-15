FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency metadata first so Docker can cache dependency installation.
COPY pyproject.toml ./

# Resolve and install dependencies. UV creates/updates uv.lock in the image.
RUN uv sync --no-install-project

# Copy application source.
COPY . .

# Install the local project.
RUN uv sync

CMD ["uv", "run", "python", "main.py"]
