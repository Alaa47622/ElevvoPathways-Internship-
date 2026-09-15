# RAG-Powered Talent Search Engine

A production-style learning project for **Task 8: RAG-Powered Talent Search Engine**.

## What it does

A recruiter can ask natural-language questions such as:

> Find me a Junior Data Analyst who knows SQL and Tableau.

The system:

1. Loads resumes from CSV/JSON.
2. Normalizes each resume into a searchable document.
3. Creates embeddings with a Hugging Face sentence-transformer.
4. Stores vectors in ChromaDB.
5. Retrieves the top candidates using semantic similarity.
6. Sends the top 3 resumes to an LLM for an evidence-based fit explanation.
7. Exposes the pipeline through FastAPI.
8. Provides a Streamlit recruiter chat UI.
9. Includes a fairness/bias audit that keeps demographic fields out of retrieval.
10. Runs tests/linting in GitHub Actions and builds a Docker image.

## Architecture

```text
Resume CSV/JSON
      |
      v
  Ingestion
      |
      v
HuggingFace Embeddings
      |
      v
   ChromaDB  <---- persistent vector storage
      ^
      |
Recruiter query
      |
      v
Semantic Retriever (top 3)
      |
      v
LLM evaluator
      |
      +----> candidate fit explanations
      |
      +----> bias audit (optional)
      |
      v
FastAPI ----> Streamlit UI
```

## Project structure

```text
talent-rag-engine/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   ├── rag.py
│   ├── services/
│   │   ├── ingestion.py
│   │   ├── evaluator.py
│   │   └── bias.py
│   └── models/
├── data/
│   └── sample_resumes.csv
├── scripts/
│   └── ingest.py
├── tests/
│   ├── test_api.py
│   ├── test_bias.py
│   └── test_ingestion.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 1. Install with uv

Install uv first if you do not have it.

Then:

```bash
uv sync
```

Activate the environment if desired:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Configure the LLM

Copy:

```text
.env.example -> .env
```

Set:

```env
OPENAI_API_KEY=your_key_here
```

The embedding model is local, so an OpenAI key is only required for the LLM explanation step.

## 3. Ingest the sample resumes

```bash
uv run python scripts/ingest.py
```

This creates the persistent Chroma database under `storage/chroma`.

## 4. Run the API

```bash
uv run uvicorn app.main:app --reload
```

API docs:

```text
http://localhost:8000/docs
```

## 5. Run Streamlit

Open another terminal:

```bash
uv run streamlit run app/ui.py
```

Then open the address shown by Streamlit.

## 6. Run everything with Docker

```bash
# Optional: create .env and add OPENAI_API_KEY for LLM explanations.
docker compose up --build
```

The API container automatically indexes the bundled sample resumes on first startup.


Services:

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Streamlit: http://localhost:8501

The Chroma storage is mounted to `./storage`.

## API example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Junior Data Analyst with SQL and Tableau\",\"top_k\":3}"
```

## Bias Check

The search pipeline does **not** use demographic fields as retrieval text.

If the dataset contains a `gender` field, the audit can calculate selection rates by group for the retrieved candidates. This is an audit signal, not a hiring decision.

Example:

```bash
curl http://localhost:8000/bias-check
```

The audit reports:

- group counts
- selection rates
- representation ratio
- a warning when the simple ratio falls below the configured threshold

This is intentionally transparent and should not be treated as a legally sufficient employment fairness assessment.

## Dataset

The project includes a tiny synthetic CSV so it runs immediately.

For the recommended Kaggle Resume Entities for NER dataset, download the dataset separately and adapt the column mapping in `app/services/ingestion.py`.

Do not commit a large downloaded dataset or API credentials to Git.

## CI/CD

GitHub Actions in `.github/workflows/ci.yml`:

- installs uv
- syncs dependencies
- runs Ruff
- runs pytest
- builds the Docker image

To deploy to a cloud platform later, add a protected GitHub secret and a deployment job after the Docker build. The repository is deliberately configured so the build/test pipeline is independent of deployment credentials.

## Why uv?

`uv` manages:

- the virtual environment
- dependency installation
- lockfile resolution
- reproducible CI setup
- running project commands

Typical commands:

```bash
uv sync
uv add package-name
uv add --dev package-name
uv lock
uv run pytest
uv run ruff check .
```
