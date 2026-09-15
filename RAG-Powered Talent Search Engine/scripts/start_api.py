from pathlib import Path

from app.rag import get_vector_store, ingest


def main() -> None:
    store = get_vector_store()
    if store._collection.count() == 0:
        sample = Path("data/sample_resumes.csv")
        if sample.exists():
            print("Vector database is empty; indexing sample resumes...")
            ingest(sample)
    else:
        print("Vector database already contains data; skipping bootstrap.")

    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
