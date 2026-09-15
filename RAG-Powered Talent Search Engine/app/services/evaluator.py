from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings


class CandidateEvaluation(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    explanation: str


class EvaluationService:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0,
            )

    def evaluate(self, query: str, document: Document) -> CandidateEvaluation:
        if self.llm is None:
            return CandidateEvaluation(
                fit_score=None,
                explanation=(
                    "LLM evaluation is disabled because OPENAI_API_KEY is not configured. "
                    "The semantic retrieval result is still available."
                ),
            )

        prompt = f"""
You are an evidence-based recruiting assistant.

Recruiter request:
{query}

Candidate resume:
{document.page_content}

Evaluate only the evidence present in the resume. Do not infer protected or
demographic characteristics. Do not invent experience, education, tools, or
seniority.

Return:
- fit_score: 0-100
- explanation: concise explanation of why the candidate matches or does not
  match, explicitly mentioning evidence and important gaps.
""".strip()

        result = self.llm.with_structured_output(CandidateEvaluation).invoke(prompt)
        return result
