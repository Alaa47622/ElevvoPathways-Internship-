from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation


def train_lda(
    document_term_matrix,
    n_topics: int,
    random_state: int,
    max_iter: int,
) -> LatentDirichletAllocation:
    """Train an LDA model on a document-term matrix."""
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=max_iter,
        evaluate_every=-1,
    )
    model.fit(document_term_matrix)
    return model


def get_top_words(model, feature_names, n_words: int) -> list[list[str]]:
    """Return the highest-weighted words for each topic."""
    topics = []

    for topic_weights in model.components_:
        top_indices = np.argsort(topic_weights)[::-1][:n_words]
        topics.append([feature_names[i] for i in top_indices])

    return topics


def save_lda_model(model, vectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        path,
    )
