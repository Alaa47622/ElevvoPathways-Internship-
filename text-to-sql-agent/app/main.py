import streamlit as st
from app.agent.graph import build_graph

st.set_page_config(page_title="Autonomous Text-to-SQL Agent", page_icon="🧠", layout="wide")
st.title("🧠 Autonomous Text-to-SQL Agent")
st.caption("Natural language → SQL → SQLite → Answer / Chart")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            with st.expander("Generated SQL"):
                st.code(message["sql"], language="sql")
        if message.get("rows"):
            st.dataframe(message["rows"], use_container_width=True)
        if message.get("chart"):
            st.pyplot(message["chart"])

question = st.chat_input("Ask a question about the database...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and querying the database..."):
            try:
                result = build_graph().invoke({
                    "question": question, "retry_count": 0, "history": []
                })
                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)
                if result.get("sql"):
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
                if result.get("rows"):
                    st.dataframe(result["rows"], use_container_width=True)
                if result.get("chart"):
                    st.pyplot(result["chart"])
                if result.get("history"):
                    with st.expander("Agent execution trace"):
                        for item in result["history"]:
                            st.write(item)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sql": result.get("sql"), "rows": result.get("rows"),
                    "chart": result.get("chart"),
                })
            except Exception as exc:
                st.error(f"Application error: {exc}")
