#!/usr/bin/env bash
set -e
uv run python -m peft_summarizer.evaluate \
  --adapter-path artifacts/adapters/phi2-samsum \
  --max-samples 100 \
  --compare-base
