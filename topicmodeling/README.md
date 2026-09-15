# News Topic Modeling — BBC News

A complete NLP project for discovering hidden topics in BBC news articles using:

- **LDA (Latent Dirichlet Allocation)** — main task
- **NMF (Non-negative Matrix Factorization)** — bonus
- **Gensim** — topic coherence evaluation
- **pyLDAvis** — interactive LDA visualization
- **WordCloud** — topic-word visualization
- **UV** — Python project/environment/dependency management
- **Docker** — reproducible execution

## 1. Project structure

```text
news-topic-modeling/
├── data/
│   ├── raw/
│   │   └── BBC_News.csv
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   └── topic_modeling/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── vectorization.py
│       ├── lda_model.py
│       ├── nmf_model.py
│       ├── evaluation.py
│       └── visualization.py
├── main.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## 2. Dataset

Download the BBC News dataset and put the CSV here:

```text
data/raw/BBC_News.csv
```

The loader is intentionally flexible and can detect common column names such as:

- `text`
- `article`
- `content`
- `description`

If the dataset has a category column such as `category` or `label`, it is kept for optional analysis but **is not used to train the unsupervised topic models**.

> Do not put Kaggle credentials inside this project or Docker image.

## 3. Install UV

On Windows PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Restart the terminal if necessary, then verify:

```powershell
uv --version
```

## 4. Create the environment

From the project root:

```powershell
uv sync
```

This creates the environment and installs the dependencies defined by `pyproject.toml`.

Generate the lockfile, then run the project:

```powershell
uv lock
uv sync
uv run python main.py
```

## 5. Configuration

You can change the main experiment settings in:

```text
src/topic_modeling/config.py
```

Important settings:

```python
N_TOPICS = 5
N_TOP_WORDS = 10
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95
RANDOM_STATE = 42
```

BBC News naturally has five well-known categories, so `N_TOPICS=5` is a reasonable starting experiment. The model is still unsupervised; category labels are not supplied to LDA/NMF.

## 6. Run

```powershell
uv run python main.py
```

The program will:

1. Load the articles.
2. Clean and tokenize the text.
3. Remove stopwords.
4. Build a Bag-of-Words matrix.
5. Train LDA.
6. Print the most significant words per topic.
7. Build TF-IDF features.
8. Train NMF.
9. Compare LDA and NMF.
10. Calculate topic coherence.
11. Save charts and a comparison report.
12. Save the trained models.
13. Create a pyLDAvis HTML visualization when possible.

Outputs appear under:

```text
outputs/
├── figures/
├── models/
└── reports/
```

## 7. Docker

Build:

```powershell
docker build -t news-topic-modeling .
```

Run:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/outputs:/app/outputs" news-topic-modeling
```

The volume mappings mean your dataset and generated outputs stay on your Windows machine.

If Docker Desktop is not installed/running, use UV locally first. The application itself does not depend on Docker.

## 8. What the algorithms do

### LDA

LDA treats every document as a mixture of topics and every topic as a probability distribution over words.

Conceptually:

```text
Documents
   ↓
Bag of Words
   ↓
LDA
   ↓
Topic 1 → government, minister, election...
Topic 2 → football, match, player...
...
```

### NMF

NMF factorizes the non-negative TF-IDF document-term matrix:

```text
TF-IDF matrix ≈ document-topic matrix × topic-word matrix
```

The largest values in each topic-word vector become the topic's representative words.

## 9. LDA vs NMF

The project compares:

- Topic coherence
- Runtime
- Top words
- Qualitative interpretability

A higher coherence score can indicate more semantically consistent topics, but it should not be treated as the only measure of model quality.

## 10. Why UV?

`pyproject.toml` describes the project and its dependencies.

`uv.lock` records the resolved dependency graph so that the environment can be reproduced more consistently.

Typical commands:

```powershell
uv sync
uv run python main.py
uv add package-name
uv lock
```

## 11. Why Docker?

UV manages the Python environment/dependencies.

Docker packages the application and its runtime into a container.

They solve related but different problems:

```text
UV
 └── Python project + dependencies

Docker
 └── Application + runtime/container environment
```


