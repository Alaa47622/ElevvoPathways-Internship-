install:
	uv sync --dev

ingest:
	uv run python scripts/ingest.py

api:
	uv run uvicorn app.main:app --reload

ui:
	uv run streamlit run app/ui.py

test:
	uv run pytest

lint:
	uv run ruff check .

docker:
	docker compose up --build
