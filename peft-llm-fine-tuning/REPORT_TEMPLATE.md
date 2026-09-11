# Task 9 Report Template

## 1. Objective
Fine-tune a small language model for dialogue summarization under a single-GPU resource constraint.

## 2. Dataset
SAMSum:
- dialogue -> input
- summary -> target

## 3. Base model
Phi-2 (2.7B).

## 4. Method
QLoRA:
- 4-bit NF4 quantization
- LoRA adapters
- frozen base model
- gradient accumulation
- gradient checkpointing

## 5. Experiments
Record each MLflow run.

| Run | LoRA r | LR | Epochs | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |

## 6. Base vs Fine-tuned

| Metric | Base | Fine-tuned | Improvement |
|---|---:|---:|---:|
| ROUGE-1 | | | |
| ROUGE-2 | | | |
| ROUGE-L | | | |
| ROUGE-Lsum | | | |

## 7. Resource Optimization
Discuss:
- GPU memory
- training time
- 4-bit quantization
- number of trainable parameters
- batch size
- gradient accumulation

## 8. MLOps
Explain:
- uv
- MLflow
- GitHub Actions
- Docker
- FastAPI
- optional DVC

## 9. Conclusion
Explain whether QLoRA improved task-specific summarization and whether the resource savings justified the approach.

## 10. Limitations
- Phi-2 is not a modern instruction-tuned model.
- ROUGE is only one measure of summarization quality.
- Results depend on generation settings and sample size.
- Full test-set evaluation is recommended for final reporting.
