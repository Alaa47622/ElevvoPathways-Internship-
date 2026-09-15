from pathlib import Path
import pandas as pd

from src.topic_modeling.data_loader import load_bbc_news


def test_loader_detects_text_and_category(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "category": ["sport", "tech"],
            "text": ["A football match happened.", "A new computer was released."],
        }
    ).to_csv(csv_path, index=False)

    df = load_bbc_news(csv_path)

    assert list(df.columns) == ["text", "category"]
    assert len(df) == 2
