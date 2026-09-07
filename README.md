# ElevoPathways Internship

A collection of applied **NLP engineering** projects completed as part of the **ElevoPathways Internship**, with an emphasis on production-readiness through **MLOps** practices and **Docker**-based deployment. Each task simulates a real-world industry problem — from efficient LLM fine-tuning to retrieval and autonomous agents — with practical constraints like resource optimization, self-correction, containerization, and evaluation.

##  About

This repository documents hands-on work across core areas of modern NLP engineering and MLOps:

- **LLM Fine-Tuning & Optimization** (PEFT, QLoRA, quantization)
- **Retrieval-Augmented Generation (RAG)** and vector search
- **Autonomous AI Agents** (tool use, self-correction, function calling)
- **Natural Language Processing** (topic modeling, unsupervised learning)
- **MLOps & Deployment** (containerization with Docker, experiment tracking, reproducible pipelines)

Each task includes a real business context, a defined constraint that mirrors production challenges, and a bonus stretch goal.

##  Tasks

| # | Task | Focus | Key Tools |
|---|------|-------|-----------|
| 5 | Topic Modeling on News Articles | Unsupervised NLP, LDA | Gensim, pyLDAvis, NLTK/spaCy, Scikit-learn |
| 8 | RAG-Powered Talent Search Engine | Retrieval-Augmented Generation, Vector DBs | LangChain/LlamaIndex, ChromaDB/FAISS, HuggingFace, OpenAI/Gemini API |
| 9 | Efficient LLM Fine-Tuning (PEFT) | LoRA/QLoRA, Quantization, Model Evaluation | HuggingFace (PEFT/BitsAndBytes), PyTorch, WandB |
| 10 | Autonomous "Text-to-SQL" Agent | AI Agents, Function Calling, Text-to-SQL | LangChain/LangGraph, SQLite, OpenAI/Gemini API, Streamlit |

*(More NLP tasks will be added as the internship progresses.)*

Each task lives in its own folder and includes a `Dockerfile` for containerized setup:

```
ElevoPathways-Internship/
├── task-05-topic-modeling/
├── task-08-rag-talent-search/
├── task-09-peft-fine-tuning/
├── task-10-text-to-sql-agent/
└── README.md
```

##  Task Details

### Task 5 — Topic Modeling on News Articles
Discover hidden themes across a news corpus (BBC News dataset). Preprocesses text (tokenization, lowercasing, stopword removal), applies Latent Dirichlet Allocation (LDA) to extract dominant topics, and surfaces the top words per topic.
- **Bonus:** Compare LDA vs. NMF; visualize topics with pyLDAvis or word clouds.

### Task 8 — RAG-Powered Talent Search Engine
A semantic search engine for recruiters: embeds resumes into a vector database and answers natural-language queries like *"Find me a Junior Data Analyst who knows SQL and Tableau."*
- **Constraint:** LLM-based evaluation — the top 3 matches are passed to an LLM to generate a plain-language explanation of fit, rather than a raw similarity score.
- **Bonus:** A bias-check feature flagging demographic skew in results.

### Task 9 — Efficient LLM Fine-Tuning (PEFT)
Fine-tunes a small language model (Llama 3 8B, Mistral, or Phi-2) using QLoRA to outperform a larger general-purpose model on a specific task (dialogue summarization via the Samsum dataset).
- **Constraint:** Trains on a single GPU (e.g., Colab T4); success is measured by comparing ROUGE scores of the base vs. fine-tuned model.
- **Bonus:** Merge LoRA adapters into a standalone model and publish to the HuggingFace Hub.

### Task 10 — Autonomous "Text-to-SQL" Agent
An AI agent that translates natural-language business questions ("How much revenue did we make last Friday?") into executable SQL, runs them, and summarizes the results.
- **Constraint:** Self-correction logic — if the generated SQL errors out, the agent reads the error and rewrites the query automatically.
- **Bonus:** A chart-generation tool option; read-only DB permissions to prevent accidental data loss.

## 🛠️ Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers/PEFT/BitsAndBytes` · `LangChain/LangGraph/LlamaIndex` · `ChromaDB/FAISS` · `Gensim` · `Scikit-learn` · `Streamlit` · `WandB` · `OpenAI/Gemini API` · `Docker` · `MLOps tooling (CI/CD, experiment tracking, reproducible environments)`

##  Getting Started

Each task folder contains its own notebook/script, `requirements.txt`, and `Dockerfile`. General setup:

```bash
git clone https://github.com/Alaa47622/ElevoPathways-Internship.git
cd ElevoPathways-Internship/<task-folder>

# Option A: local environment
pip install -r requirements.txt

# Option B: run in Docker
docker build -t <task-name> .
docker run --rm -it <task-name>
```

