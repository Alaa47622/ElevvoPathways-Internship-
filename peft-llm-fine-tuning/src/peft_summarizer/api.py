import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import load_for_inference
from .model_utils import generate_summary


app = FastAPI(
    title="SAMSum QLoRA Summarization API",
    version="0.1.0",
    description="Dialogue summarization using a PEFT/QLoRA fine-tuned language model.",
)

_model = None
_tokenizer = None
_generation = None


class SummarizeRequest(BaseModel):
    dialogue: str = Field(..., min_length=1, description="Dialogue to summarize")


class SummarizeResponse(BaseModel):
    summary: str
    generation_time_ms: float


@app.on_event("startup")
def startup():
    global _model, _tokenizer, _generation
    _model, _tokenizer, _generation = load_for_inference(
        os.getenv("INFERENCE_CONFIG", "configs/inference.yaml")
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model": os.getenv("MODEL_SOURCE", "microsoft/phi-2"),
        "adapter_path": os.getenv(
            "ADAPTER_PATH",
            "artifacts/adapters/phi2-samsum",
        ),
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    start = time.perf_counter()
    summary = generate_summary(
        _model,
        _tokenizer,
        request.dialogue,
        _generation,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return SummarizeResponse(
        summary=summary,
        generation_time_ms=round(elapsed_ms, 2),
    )
