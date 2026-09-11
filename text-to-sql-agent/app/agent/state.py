from typing import Any, TypedDict

class AgentState(TypedDict, total=False):
    question: str
    schema: str
    sql: str
    rows: list[dict[str, Any]]
    columns: list[str]
    error: str | None
    answer: str
    retry_count: int
    chart: Any
    history: list[str]
