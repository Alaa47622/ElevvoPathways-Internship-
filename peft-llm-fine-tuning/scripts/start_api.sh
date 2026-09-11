#!/usr/bin/env bash
set -e
uv run uvicorn peft_summarizer.api:app --host 0.0.0.0 --port 8000
