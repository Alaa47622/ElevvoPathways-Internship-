import csv
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


SEARCH_EXCLUDED_FIELDS = {
    "gender",
    "sex",
    "age",
    "date_of_birth",
    "dob",
    "race",
    "ethnicity",
    "religion",
    "nationality",
    "marital_status",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("records", [])

    raise ValueError("Only CSV and JSON are supported.")


def _record_to_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in record.items():
        if key.lower() in SEARCH_EXCLUDED_FIELDS:
            continue
        if value is None or str(value).strip() == "":
            continue
        parts.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(parts)


def load_resume_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    records = _load_records(path)
    documents: list[Document] = []

    for index, record in enumerate(records, start=1):
        candidate_id = str(
            record.get("candidate_id")
            or record.get("id")
            or record.get("ID")
            or f"CAND-{index:04d}"
        )
        name = str(record.get("name") or record.get("Name") or candidate_id)

        metadata = {
            "candidate_id": candidate_id,
            "name": name,
        }

        # Keep demographic fields only as metadata for auditing; never put them
        # into the searchable document text.
        for key, value in record.items():
            if key.lower() in SEARCH_EXCLUDED_FIELDS and value not in (None, ""):
                metadata[key.lower()] = str(value)

        documents.append(
            Document(
                page_content=_record_to_text(record),
                metadata=metadata,
            )
        )

    return documents
