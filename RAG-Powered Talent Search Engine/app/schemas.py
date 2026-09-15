from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


class Candidate(BaseModel):
    candidate_id: str
    name: str
    similarity: float
    resume: str
    metadata: dict[str, Any] = {}


class Evaluation(BaseModel):
    candidate_id: str
    name: str
    fit_score: int | None = None
    explanation: str


class SearchResponse(BaseModel):
    query: str
    candidates: list[Candidate]
    evaluations: list[Evaluation]


class BiasGroup(BaseModel):
    group: str
    count: int
    selection_rate: float


class BiasReport(BaseModel):
    demographic_field: str
    groups: list[BiasGroup]
    representation_ratio: float | None
    warning: str | None
