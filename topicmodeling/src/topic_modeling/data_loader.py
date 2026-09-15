from pathlib import Path
import pandas as pd


TEXT_COLUMN_CANDIDATES = [
    "text",
    "article",
    "content",
    "description",
    "body",
    "news",
]

LABEL_COLUMN_CANDIDATES = [
    "category",
    "category_name",
    "label",
    "class",
    "topic",
]


def _find_column(columns, candidates):
    normalized = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Also allow partial matches such as "article_text"
    for column in columns:
        lowered = str(column).strip().lower()
        if any(candidate in lowered for candidate in candidates):
            return column

    return None


def load_bbc_news(path: Path) -> pd.DataFrame:
    """Load a BBC News CSV and normalize it to text/category columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the BBC News CSV and place it at data/raw/BBC_News.csv"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset CSV is empty.")

    text_column = _find_column(df.columns, TEXT_COLUMN_CANDIDATES)

    # Some BBC datasets use the second column for article text.
    if text_column is None and len(df.columns) >= 2:
        object_columns = [
            c for c in df.columns
            if pd.api.types.is_string_dtype(df[c])
        ]
        if object_columns:
            text_column = max(
                object_columns,
                key=lambda c: df[c].fillna("").astype(str).str.len().mean(),
            )

    if text_column is None:
        raise ValueError(
            f"Could not identify the article-text column. "
            f"Available columns: {list(df.columns)}"
        )

    label_column = _find_column(df.columns, LABEL_COLUMN_CANDIDATES)

    result = pd.DataFrame()
    result["text"] = df[text_column].fillna("").astype(str)

    if label_column is not None:
        result["category"] = df[label_column].fillna("unknown").astype(str)

    result = result[result["text"].str.strip().ne("")]
    result = result.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return result
