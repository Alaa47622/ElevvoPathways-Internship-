FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml ./

RUN uv sync --no-dev

COPY app ./app
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p /app/storage/chroma

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "python", "scripts/start_api.py"]
