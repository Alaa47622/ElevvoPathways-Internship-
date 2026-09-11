from langgraph.graph import END, START, StateGraph
from app.agent.nodes import (
    inspect_schema, generate_sql, execute_query, self_correct,
    analyze_result, fail, route_after_execution,
)
from app.agent.state import AgentState

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("inspect_schema", inspect_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_query", execute_query)
    graph.add_node("self_correct", self_correct)
    graph.add_node("analyze_result", analyze_result)
    graph.add_node("fail", fail)
    graph.add_edge(START, "inspect_schema")
    graph.add_edge("inspect_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_query")
    graph.add_conditional_edges(
        "execute_query", route_after_execution,
        {"analyze": "analyze_result", "correct": "self_correct", "fail": "fail"},
    )
    graph.add_edge("self_correct", "execute_query")
    graph.add_edge("analyze_result", END)
    graph.add_edge("fail", END)
    return graph.compile()
