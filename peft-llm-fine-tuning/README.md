# SAMSum QLoRA MLOps Project

 **Task 9: Efficient LLM Fine-Tuning (PEFT)**.

## What this project does

- Uses the **SAMSum** dialogue summarization dataset.
- Uses **Microsoft Phi-2 (2.7B)** as the baseline/fine-tuning model.
- Loads Phi-2 in **4-bit** and fine-tunes it with **LoRA / QLoRA**.
- Evaluates the base and fine-tuned model with **ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum**.
- Tracks experiments with **MLflow**.
- Saves a lightweight LoRA adapter.
- Optionally merges the LoRA adapter into the base model.
- Serves the fine-tuned model through **FastAPI**.
- Includes **Docker / Docker Compose**.
- Includes **GitHub Actions CI**.
- Uses **uv** for reproducible Python environments.




## Why Phi-2?

Phi-2 is a 2.7B-parameter causal language model. It is small enough to make a practical single-GPU PEFT demonstration while still being a real LLM.

The official model card documents Phi-2 as a 2.7B model and notes that it has a 2048-token context length.

## Hardware target

### Training
Recommended:
- Google Colab T4 (16 GB)
- Linux + NVIDIA GPU with CUDA
- 16 GB+ VRAM

The default configuration is deliberately conservative:
- 4-bit NF4
- batch size 1
- gradient accumulation
- gradient checkpointing
- max sequence length 512
- 2 epochs

If you get CUDA out-of-memory errors, lower `max_seq_length`, reduce LoRA rank, or increase gradient accumulation while keeping batch size at 1.

### CPU / Windows
The FastAPI application can be used for testing, but QLoRA training with `bitsandbytes` is most reliable in Linux/Colab. For your actual training experiment, use a Linux GPU environment such as Colab T4.

## Project structure

```text
samsum-qlora-mlops/
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── src/
│   └── peft_summarizer/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── prompts.py
│       ├── model_utils.py
│       ├── train.py
│       ├── baseline.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── merge_adapter.py
│       └── api.py
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── start_api.sh
├── tests/
│   ├── test_prompts.py
│   └── test_api.py
├── artifacts/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## 1. Install uv

Install uv from the official installer, then verify:

```bash
uv --version
```

## 2. Create the environment

From the project directory:

```bash
uv sync
```

Run commands through the project environment:

```bash
uv run python -m peft_summarizer.baseline
```

You do not need to manually activate `.venv` when using `uv run`.

## 3. Download / inspect the dataset

```bash
uv run python -m peft_summarizer.data --inspect
```

The project uses:

```text
knkarthick/samsum
```

The dataset contains `dialogue`, `summary`, and `id`, with train/validation/test splits.

## 4. Evaluate the base model

```bash
uv run python -m peft_summarizer.baseline
```

For a fast smoke test:

```bash
uv run python -m peft_summarizer.baseline --max-samples 20
```

This writes:

```text
artifacts/evaluation/base.json
artifacts/evaluation/base_predictions.jsonl
```

## 5. Train QLoRA

```bash
uv run python -m peft_summarizer.train --config configs/train.yaml
```

For a quick debugging run:

```bash
uv run python -m peft_summarizer.train \
  --config configs/train.yaml \
  --max-train-samples 100 \
  --max-eval-samples 20
```

The adapter is saved under:

```text
artifacts/adapters/phi2-samsum/
```

## 6. Evaluate the fine-tuned model

```bash
uv run python -m peft_summarizer.evaluate \
  --adapter-path artifacts/adapters/phi2-samsum \
  --max-samples 100
```

The result is written to:

```text
artifacts/evaluation/finetuned.json
artifacts/evaluation/finetuned_predictions.jsonl
```

## 7. Compare base vs fine-tuned

```bash
uv run python -m peft_summarizer.evaluate \
  --adapter-path artifacts/adapters/phi2-samsum \
  --max-samples 100 \
  --compare-base
```

The final report should contain real values like:

```text
Metric       Base       Fine-tuned
-----------------------------------
ROUGE-1      ...        ...
ROUGE-2      ...        ...
ROUGE-L      ...        ...
ROUGE-Lsum   ...        ...
```

Do not fabricate these numbers.

## 8. MLflow

Start MLflow:

```bash
uv run mlflow ui --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000
```

Training logs:
- hyperparameters
- training loss
- evaluation loss
- model information
- configuration
- output artifacts

## 9. Merge the adapter

After training:

```bash
uv run python -m peft_summarizer.merge_adapter \
  --adapter-path artifacts/adapters/phi2-samsum \
  --output-path artifacts/merged_model
```

This produces a standalone merged model.

> Merging is done with the non-quantized base model because a 4-bit training representation is not the same thing as a normal merged deployment checkpoint.

## 10. Run FastAPI

The API loads the adapter model by default:

```bash
uv run uvicorn peft_summarizer.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Summarization:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d "{\"dialogue\":\"Amanda: Are you coming tonight? Jerry: Yes, I will arrive at 8.\"}"
```

## 11. Docker

Build:

```bash
docker build -t samsum-qlora-api .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_SOURCE=microsoft/phi-2 \
  samsum-qlora-api
```

For GPU inference, use an NVIDIA-enabled Docker host and:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e MODEL_SOURCE=artifacts/adapters/phi2-samsum \
  samsum-qlora-api
```

## 12. Docker Compose

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- MLflow: `http://localhost:5000`

The Compose file is intended as a development/deployment template. Training should normally happen separately on a GPU machine.

## 13. Tests

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check src tests
```

## 14. Suggested experiment plan

Run at least three experiments:

```text
Experiment A:
LoRA r=8

Experiment B:
LoRA r=16

Experiment C:
LoRA r=32
```

Track:
- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- training loss
- evaluation loss
- training time
- GPU memory if available

Then select the best configuration based on validation performance and reasonable resource usage.

## 15. What to put in the final report

### Problem
Large general-purpose LLMs are expensive to fine-tune and deploy.

### Solution
Use parameter-efficient fine-tuning with QLoRA to specialize a smaller model for dialogue summarization.

### Resource optimization
Explain:
- 4-bit quantization
- LoRA adapters
- frozen base parameters
- gradient accumulation
- gradient checkpointing

### Evaluation
Compare:
- base model
- fine-tuned model


