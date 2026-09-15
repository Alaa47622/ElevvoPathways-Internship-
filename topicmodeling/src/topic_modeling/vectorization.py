from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import MAX_FEATURES, MIN_DF, MAX_DF


def build_count_vectorizer() -> CountVectorizer:
    """Vectorizer used as input for classical LDA."""
    return CountVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """TF-IDF vectorizer used as input for NMF."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
