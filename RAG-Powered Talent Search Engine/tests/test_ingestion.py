from pathlib import Path

from app.services.ingestion import load_resume_documents


def test_ingestion_excludes_demographics_from_search_text():
    docs = load_resume_documents(Path("data/sample_resumes.csv"))

    assert len(docs) == 8
    assert "SQL" in docs[0].page_content
    assert "gender" not in docs[0].page_content.lower()
    assert docs[0].metadata["gender"] == "Female"
