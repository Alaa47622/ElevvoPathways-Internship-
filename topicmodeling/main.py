from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.topic_modeling.config import (
    FIGURES_DIR,
    MODELS_DIR,
    NMF_MAX_ITER,
    N_TOPICS,
    N_TOP_WORDS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    LDA_MAX_ITER,
    create_directories,
)
from src.topic_modeling.data_loader import load_bbc_news
from src.topic_modeling.evaluation import ModelResult, calculate_coherence
from src.topic_modeling.lda_model import (
    get_top_words as get_lda_top_words,
    save_lda_model,
    train_lda,
)
from src.topic_modeling.nmf_model import (
    get_top_words as get_nmf_top_words,
    save_nmf_model,
    train_nmf,
)
from src.topic_modeling.preprocessing import TextPreprocessor
from src.topic_modeling.vectorization import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
)
from src.topic_modeling.visualization import (
    plot_model_comparison,
    plot_topic_words,
    plot_wordclouds,
    save_pyldavis,
)


def print_topics(model_name: str, topics: list[list[str]]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{model_name} TOPICS")
    print(f"{'=' * 60}")

    for index, words in enumerate(topics, start=1):
        print(f"Topic {index}: {', '.join(words)}")


def save_topic_json(results: dict, path: Path) -> None:
    serializable = {
        name: {
            "topics": result.topics,
            "runtime_seconds": result.runtime_seconds,
            "coherence_cv": result.coherence_cv,
        }
        for name, result in results.items()
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    create_directories()

    print("Loading BBC News dataset...")
    df = load_bbc_news(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} unique articles.")

    # -----------------------------
    # 1. Text preprocessing
    # -----------------------------
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()

    df["clean_text"] = preprocessor.transform(df["text"])

    # Remove documents that became empty after preprocessing.
    df = df[df["clean_text"].str.strip().ne("")].reset_index(drop=True)

    processed_path = (
        Path(RAW_DATA_PATH).parents[1]
        / "processed"
        / "bbc_news_clean.csv"
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    tokenized_documents = [
        text.split()
        for text in df["clean_text"]
    ]

    # -----------------------------
    # 2. LDA — Bag of Words
    # -----------------------------
    print("\nBuilding Bag-of-Words matrix...")
    count_vectorizer = build_count_vectorizer()
    count_matrix = count_vectorizer.fit_transform(df["clean_text"])
    count_features = count_vectorizer.get_feature_names_out()

    print(
        f"BoW shape: {count_matrix.shape[0]} documents × "
        f"{count_matrix.shape[1]} terms"
    )

    print("\nTraining LDA...")
    start = time.perf_counter()
    lda = train_lda(
        count_matrix,
        n_topics=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=LDA_MAX_ITER,
    )
    lda_runtime = time.perf_counter() - start

    lda_topics = get_lda_top_words(
        lda,
        count_features,
        N_TOP_WORDS,
    )

    print_topics("LDA", lda_topics)
    print(f"LDA runtime: {lda_runtime:.3f} seconds")

    lda_coherence = calculate_coherence(
        lda_topics,
        tokenized_documents,
    )
    print(f"LDA c_v coherence: {lda_coherence:.4f}")

    save_lda_model(
        lda,
        count_vectorizer,
        MODELS_DIR / "lda.joblib",
    )

    # -----------------------------
    # 3. NMF — TF-IDF
    # -----------------------------
    print("\nBuilding TF-IDF matrix...")
    tfidf_vectorizer = build_tfidf_vectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])
    tfidf_features = tfidf_vectorizer.get_feature_names_out()

    print(
        f"TF-IDF shape: {tfidf_matrix.shape[0]} documents × "
        f"{tfidf_matrix.shape[1]} terms"
    )

    print("\nTraining NMF...")
    start = time.perf_counter()
    nmf = train_nmf(
        tfidf_matrix,
        n_topics=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=NMF_MAX_ITER,
    )
    nmf_runtime = time.perf_counter() - start

    nmf_topics = get_nmf_top_words(
        nmf,
        tfidf_features,
        N_TOP_WORDS,
    )

    print_topics("NMF", nmf_topics)
    print(f"NMF runtime: {nmf_runtime:.3f} seconds")

    nmf_coherence = calculate_coherence(
        nmf_topics,
        tokenized_documents,
    )
    print(f"NMF c_v coherence: {nmf_coherence:.4f}")

    save_nmf_model(
        nmf,
        tfidf_vectorizer,
        MODELS_DIR / "nmf.joblib",
    )

    # -----------------------------
    # 4. Evaluation
    # -----------------------------
    results = {
        "LDA": ModelResult(
            name="LDA",
            topics=lda_topics,
            runtime_seconds=lda_runtime,
            coherence_cv=lda_coherence,
        ),
        "NMF": ModelResult(
            name="NMF",
            topics=nmf_topics,
            runtime_seconds=nmf_runtime,
            coherence_cv=nmf_coherence,
        ),
    }

    comparison = pd.DataFrame(
        [
            {
                "model": result.name,
                "coherence_cv": result.coherence_cv,
                "runtime_seconds": result.runtime_seconds,
            }
            for result in results.values()
        ]
    )

    comparison.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )
    save_topic_json(
        results,
        REPORTS_DIR / "topics.json",
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    # -----------------------------
    # 5. Visualizations
    # -----------------------------
    print("\nCreating visualizations...")

    plot_topic_words(
        lda_topics,
        "LDA",
        FIGURES_DIR / "lda_topic_words.png",
    )
    plot_topic_words(
        nmf_topics,
        "NMF",
        FIGURES_DIR / "nmf_topic_words.png",
    )

    plot_wordclouds(
        lda_topics,
        "LDA",
        FIGURES_DIR / "wordclouds",
    )
    plot_wordclouds(
        nmf_topics,
        "NMF",
        FIGURES_DIR / "wordclouds",
    )

    plot_model_comparison(
        ["LDA", "NMF"],
        [lda_coherence, nmf_coherence],
        [lda_runtime, nmf_runtime],
        FIGURES_DIR,
    )

    save_pyldavis(
        lda,
        count_vectorizer,
        count_matrix,
        REPORTS_DIR / "lda_visualization.html",
    )

    print("\nDone.")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Models:  {MODELS_DIR}")
    print(f"Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
