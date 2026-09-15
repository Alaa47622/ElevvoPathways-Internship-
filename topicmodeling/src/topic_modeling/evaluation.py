from __future__ import annotations

import time
from dataclasses import dataclass

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


@dataclass
class ModelResult:
    name: str
    topics: list[list[str]]
    runtime_seconds: float
    coherence_cv: float


def calculate_coherence(
    topic_words: list[list[str]],
    tokenized_documents: list[list[str]],
) -> float:
    """Calculate c_v topic coherence using Gensim."""
    dictionary = Dictionary(tokenized_documents)

    if len(dictionary) == 0:
        return 0.0

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(coherence_model.get_coherence())


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
