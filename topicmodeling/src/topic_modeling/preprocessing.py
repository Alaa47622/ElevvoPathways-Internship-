import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


class TextPreprocessor:
    def __init__(self) -> None:
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Normalize one article into a whitespace-separated token string."""
        text = text.lower()

        # Keep alphabetic tokens and discard URLs, numbers, punctuation, etc.
        tokens = re.findall(r"[a-z]+", text)

        cleaned = []
        for token in tokens:
            if token in self.stop_words:
                continue
            if len(token) < 3:
                continue

            lemma = self.lemmatizer.lemmatize(token)
            if lemma not in self.stop_words:
                cleaned.append(lemma)

        return " ".join(cleaned)

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self.clean(text) for text in texts]
