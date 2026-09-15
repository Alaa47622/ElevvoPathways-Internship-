import argparse

from app.rag import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Index resume data into ChromaDB.")
    parser.add_argument(
        "--file",
        default="data/sample_resumes.csv",
        help="Path to a CSV or JSON resume file.",
    )
    args = parser.parse_args()

    count = ingest(args.file)
    print(f"Indexed {count} resumes.")


if __name__ == "__main__":
    main()
