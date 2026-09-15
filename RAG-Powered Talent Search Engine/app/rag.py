from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.schemas import Candidate, Evaluation
from app.services.evaluator import EvaluationService
from app.services.ingestion import load_resume_documents


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store() -> Chroma:
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(),
    )


def ingest(path: str | Path) -> int:
    docs = load_resume_documents(path)
    if not docs:
        return 0

    store = get_vector_store()
    ids = [doc.metadata["candidate_id"] for doc in docs]
    # Delete existing IDs first so repeated ingestion is safe.
    try:
        store.delete(ids=ids)
    except Exception:
        pass
    store.add_documents(documents=docs, ids=ids)
    return len(docs)


def search(query: str, top_k: int = 3) -> tuple[list[Candidate], list[Evaluation]]:
    store = get_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=top_k)

    candidates: list[Candidate] = []
    evaluations: list[Evaluation] = []
    evaluator = EvaluationService()

    for document, score in results:
        candidate_id = str(document.metadata.get("candidate_id", "unknown"))
        name = str(document.metadata.get("name", candidate_id))

        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                name=name,
                similarity=round(float(score), 4),
                resume=document.page_content,
                metadata=document.metadata,
            )
        )

        evaluation = evaluator.evaluate(query, document)
        evaluations.append(
            Evaluation(
                candidate_id=candidate_id,
                name=name,
                fit_score=evaluation.fit_score,
                explanation=evaluation.explanation,
            )
        )

    return candidates, evaluations
