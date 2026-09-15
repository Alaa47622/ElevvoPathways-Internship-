from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import NMF


def train_nmf(
    tfidf_matrix,
    n_topics: int,
    random_state: int,
    max_iter: int,
) -> NMF:
    """Train NMF on a non-negative TF-IDF matrix."""
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )
    model.fit(tfidf_matrix)
    return model


def get_top_words(model, feature_names, n_words: int) -> list[list[str]]:
    """Return the highest-weighted words for each NMF component."""
    topics = []

    for component in model.components_:
        top_indices = np.argsort(component)[::-1][:n_words]
        topics.append([feature_names[i] for i in top_indices])

    return topics


def save_nmf_model(model, vectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        path,
    )
