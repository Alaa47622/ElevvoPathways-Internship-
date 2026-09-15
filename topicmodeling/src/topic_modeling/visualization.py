from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


def plot_topic_words(
    topics: list[list[str]],
    model_name: str,
    output_path: Path,
) -> None:
    """Create one horizontal word list plot per model."""
    n_topics = len(topics)
    fig_height = max(4, n_topics * 1.5)

    fig, axes = plt.subplots(
        n_topics,
        1,
        figsize=(12, fig_height),
        squeeze=False,
    )

    for i, words in enumerate(topics):
        ax = axes[i, 0]
        ax.barh(range(len(words)), list(range(len(words), 0, -1)))
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_title(f"{model_name} — Topic {i + 1}")
        ax.set_xlabel("Relative rank")
        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_wordclouds(
    topics: list[list[str]],
    model_name: str,
    output_dir: Path,
) -> None:
    """Create a word cloud for each topic."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for topic_index, words in enumerate(topics, start=1):
        frequencies = {
            word: len(words) - index
            for index, word in enumerate(words)
        }

        cloud = WordCloud(
            width=1000,
            height=600,
            background_color="white",
            collocations=False,
        ).generate_from_frequencies(frequencies)

        path = output_dir / f"{model_name.lower()}_topic_{topic_index}.png"
        cloud.to_file(path)


def plot_model_comparison(
    model_names: list[str],
    coherence_scores: list[float],
    runtimes: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(model_names))
    plt.bar(x, coherence_scores)
    plt.xticks(x, model_names)
    plt.ylabel("Coherence (c_v)")
    plt.title("LDA vs NMF — Topic Coherence")
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_comparison.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(x, runtimes)
    plt.xticks(x, model_names)
    plt.ylabel("Training time (seconds)")
    plt.title("LDA vs NMF — Training Runtime")
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_comparison.png", dpi=160)
    plt.close(fig)


def save_pyldavis(
    lda_model,
    count_vectorizer,
    count_matrix,
    output_path: Path,
) -> None:
    """Create an interactive pyLDAvis page for the sklearn LDA model."""
    try:
        import pyLDAvis
        import pyLDAvis.lda_model

        panel = pyLDAvis.lda_model.prepare(
            lda_model,
            count_matrix,
            count_vectorizer,
            mds="pcoa",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pyLDAvis.save_html(panel, str(output_path))
        return
    except Exception as exc:
        # Visualization should not make the core modeling pipeline fail.
        print(f"[warning] pyLDAvis could not be generated: {exc}")
