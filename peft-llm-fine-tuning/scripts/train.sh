#!/usr/bin/env bash
set -e
uv run python -m peft_summarizer.train --config configs/train.yaml
