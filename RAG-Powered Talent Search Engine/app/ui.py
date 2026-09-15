import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Talent RAG Search", page_icon="🔎", layout="wide")

st.title("🔎 Talent RAG Search Engine")
st.caption("Ask a recruiter-style question and inspect the top semantic matches.")

query = st.text_input(
    "Recruiter query",
    value="Find me a Junior Data Analyst who knows SQL and Tableau.",
)

if st.button("Search", type="primary"):
    with st.spinner("Searching resumes and generating explanations..."):
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": 3},
            timeout=120,
        )

    if response.ok:
        data = response.json()

        st.subheader("Top candidates")
        for candidate, evaluation in zip(data["candidates"], data["evaluations"]):
            with st.container(border=True):
                st.markdown(f"### {candidate['name']}")
                st.write(f"**Semantic relevance:** {candidate['similarity']:.3f}")

                if evaluation["fit_score"] is not None:
                    st.write(f"**LLM fit score:** {evaluation['fit_score']}/100")

                st.write(evaluation["explanation"])

                with st.expander("Resume evidence"):
                    st.text(candidate["resume"])
    else:
        st.error(response.text)

st.divider()
st.subheader("Bias Check")
if st.button("Run demographic audit"):
    try:
        response = requests.get(f"{API_URL}/bias-check", timeout=30)
        if response.ok:
            report = response.json()
            st.json(report)
            if report.get("warning"):
                st.warning(report["warning"])
            else:
                st.success("No threshold warning was generated for the available audit data.")
        else:
            st.error(response.text)
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
