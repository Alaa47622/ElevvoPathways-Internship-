# 🧠 Autonomous "Text-to-SQL" Agent

An industry-style AI agent that converts natural-language business questions into read-only SQLite queries, executes them, automatically repairs failed SQL, analyzes results, and optionally creates charts.

## Architecture

```text
                         ┌───────────────────┐
                         │       User        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    Streamlit UI   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │     LangGraph     │
                         │   Agent Workflow  │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              Schema Agent    SQL Generator   Security
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                         ┌───────────────────┐
                         │  Read-only SQL    │
                         │     Executor      │
                         └─────────┬─────────┘
                                   │
                         SQL error?│
                         ┌─────────┴─────────┐
                         │                   │
                        YES                  NO
                         │                   │
                         ▼                   ▼
                ┌────────────────┐   ┌────────────────┐
                │ Self-Correction│   │ Result Analyzer│
                │  max 3 retries │   │  + Chart Tool  │
                └───────┬────────┘   └───────┬────────┘
                        │                    │
                        └──────► Execute ◄───┘
                                             │
                                             ▼
                                  Answer + SQL + Data
                                      + Optional Chart


                 ┌────────────────────────────────┐
                 │          MLOps Layer           │
                 ├────────────────────────────────┤
                 │ uv + uv.lock                   │
                 │ pytest + Ruff                   │
                 │ GitHub Actions CI               │
                 │ Docker                          │
                 │ GitHub Container Registry       │
                 │ CD / deployment hook            │
                 └────────────────────────────────┘
```

## Stack

- Python 3.11+
- `uv` + `uv.lock`
- LangGraph
- LangChain OpenAI integration
- OpenAI API
- SQLite
- Streamlit
- Pandas / Matplotlib
- pytest
- Ruff
- Docker / Docker Compose
- GitHub Actions
- GitHub Container Registry

## 1. Local setup with uv

```powershell
uv sync
copy .env.example .env
```

Add your OpenAI API key to `.env`.

Create/reset the demo database:

```powershell
uv run python scripts/init_db.py
```

Run tests:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
uv run ruff format --check .
```

Run the application:

```powershell
uv run streamlit run app/main.py
```

## 2. Database

The demo SQLite database is:

```text
data/ecommerce.db
```

It contains:

- `customers`
- `products`
- `orders`
- `order_items`

If you want to use Chinook instead, place the SQLite file in `data/` and update `DATABASE_PATH` in `.env`.

## 3. Docker

Build:

```powershell
docker build -t autonomous-text-to-sql-agent .
```

Run:

```powershell
docker run --rm -p 8501:8501 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  autonomous-text-to-sql-agent
```

Open:

```text
http://localhost:8501
```

### Docker Compose

Create `.env` first, then:

```powershell
docker compose up --build
```

Stop:

```powershell
docker compose down
```

The database is mounted read-only inside the container.

## 4. CI pipeline

`.github/workflows/ci.yml` runs on pushes and pull requests.

```text
GitHub Push / PR
       ↓
Checkout
       ↓
Install uv
       ↓
uv sync --locked
       ↓
Ruff
       ↓
pytest
       ↓
Docker build
       ↓
CI PASS
```

This means a change cannot silently pass without code-quality checks, tests, and a successful container build.

## 5. CD pipeline

`.github/workflows/cd.yml` runs for version tags such as:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Pipeline:

```text
Version Tag
    ↓
uv sync --locked
    ↓
pytest
    ↓
Docker build
    ↓
GitHub Container Registry
    ↓
ghcr.io/<owner>/<repo>:v1.0.0
    ↓
ghcr.io/<owner>/<repo>:latest
```

The deployment job is intentionally provider-neutral. Set `DEPLOY_ENABLED=true` as a GitHub Actions repository variable and add your hosting provider's deployment command/action.

## 6. Self-correction

If generated SQL fails, the agent captures the error and sends:

- user question
- database schema
- failed SQL
- SQLite error

back to the LLM.

The LLM generates a corrected query and the graph retries. Retries are limited by `MAX_RETRIES`.

## 7. Security

The SQL execution layer allows only a single read-only `SELECT` or `WITH` statement.

Blocked operations include:

- INSERT
- UPDATE
- DELETE
- DROP
- ALTER
- CREATE
- REPLACE
- ATTACH
- DETACH
- VACUUM
- PRAGMA
- multiple SQL statements

The SQLite runtime connection is opened in read-only mode.

## 8. Example questions

- How many customers are in the database?
- What is the total revenue?
- Which product generated the most revenue?
- Show revenue by country.
- Which customer spent the most money?
- Show the top products by revenue.

## 9. Evaluation

```powershell
uv run python evaluation/evaluate.py
```

The questions are stored in:

```text
evaluation/questions.json
```

## 10. Project structure

```text
autonomous-text-to-sql-agent/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── app/
│   ├── agent/
│   ├── database/
│   ├── tools/
│   ├── config.py
│   └── main.py
├── data/
│   └── ecommerce.db
├── evaluation/
├── scripts/
├── tests/
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

## MLOps checklist

- [x] Reproducible dependencies with uv
- [x] Locked dependency file
- [x] Environment configuration
- [x] Automated tests
- [x] Ruff quality gates
- [x] Evaluation dataset
- [x] Read-only database
- [x] SQL validation
- [x] Self-correction
- [x] Retry limits
- [x] Agent execution trace
- [x] Streamlit UI
- [x] Docker image
- [x] Docker Compose
- [x] GitHub Actions CI
- [x] GitHub Actions CD
- [x] Container Registry publishing

## Production upgrades

For a stronger production deployment, add:

1. LangSmith tracing
2. Docker image scanning
3. Dependabot
4. Secret management
5. Authentication / authorization
6. SQL AST validation
7. Query timeout and result-size limits
8. Prometheus/Grafana metrics
9. Larger golden evaluation set
10. Deployment to AWS / Azure / GCP / Render / Railway
