from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.agent.prompts import CORRECTION_PROMPT, SQL_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.config import OPENAI_MODEL, MAX_RETRIES
from app.database.schema import get_schema
from app.tools.chart_tool import make_chart, should_chart
from app.tools.sql_tool import execute_sql

def llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

def inspect_schema(state):
    state["schema"] = get_schema()
    state.setdefault("retry_count", 0)
    state.setdefault("history", [])
    return state

def generate_sql(state):
    prompt = SQL_SYSTEM_PROMPT.format(
        schema=state["schema"], current_date=date.today().isoformat()
    )
    response = llm().invoke([SystemMessage(content=prompt), HumanMessage(content=state["question"])])
    state["sql"] = str(response.content).strip().replace("```sql", "").replace("```", "").strip()
    state["history"].append("Generated SQL: " + state["sql"])
    return state

def execute_query(state):
    result = execute_sql(state["sql"])
    state["rows"] = result["rows"]
    state["columns"] = result["columns"]
    state["error"] = result["error"]
    if state["error"]:
        state["history"].append("SQL error: " + state["error"])
    return state

def route_after_execution(state):
    if not state.get("error"):
        return "analyze"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "fail"
    return "correct"

def self_correct(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    prompt = CORRECTION_PROMPT.format(
        question=state["question"], schema=state["schema"],
        sql=state["sql"], error=state["error"]
    )
    response = llm().invoke([HumanMessage(content=prompt)])
    state["sql"] = str(response.content).strip().replace("```sql", "").replace("```", "").strip()
    state["history"].append(f"Self-correction #{state['retry_count']}: {state['sql']}")
    return state

def analyze_result(state):
    if not state.get("rows"):
        state["answer"] = "The query executed successfully but returned no rows."
        return state
    response = llm().invoke([
        SystemMessage(content=(
            "You are a business analyst. Answer using ONLY the supplied SQL result. "
            "Be concise, factual, and do not invent information."
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\nSQL: {state['sql']}\n"
            f"Columns: {state['columns']}\nRows: {state['rows'][:30]}"
        )),
    ])
    state["answer"] = str(response.content).strip()
    if should_chart(state["question"], state["columns"], state["rows"]):
        state["chart"] = make_chart(state["rows"], state["columns"], state["question"])
    return state

def fail(state):
    state["answer"] = (
        f"I could not produce valid SQL after {MAX_RETRIES} correction attempts. "
        f"Last error: {state.get('error', 'unknown error')}"
    )
    return state
