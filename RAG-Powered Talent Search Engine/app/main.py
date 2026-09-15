from fastapi import FastAPI, HTTPException

from app.rag import search
from app.schemas import SearchRequest, SearchResponse
from app.services.bias import demographic_bias_report

app = FastAPI(
    title="Talent RAG Search Engine",
    version="0.1.0",
    description="Semantic resume search with LLM-based candidate explanations.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search_candidates(request: SearchRequest) -> SearchResponse:
    try:
        candidates, evaluations = search(request.query, request.top_k)
        return SearchResponse(
            query=request.query,
            candidates=candidates,
            evaluations=evaluations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/bias-check")
def bias_check() -> dict:
    """
    Small audit endpoint over the currently indexed candidates.

    For a real hiring system, run a statistically meaningful audit over a
    defined evaluation set instead of treating this endpoint as a compliance
    decision.
    """
    try:
        from app.rag import get_vector_store

        store = get_vector_store()
        data = store.get(include=["metadatas"])
        metadata = data.get("metadatas", [])
        return demographic_bias_report(metadata, field="gender")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
