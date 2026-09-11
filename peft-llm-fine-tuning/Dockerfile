FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# Install the project and dependencies.
# Model weights are intentionally NOT baked into the image.
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "peft_summarizer.api:app", "--host", "0.0.0.0", "--port", "8000"]
