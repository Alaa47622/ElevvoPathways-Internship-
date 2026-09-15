from src.topic_modeling.preprocessing import TextPreprocessor


def test_preprocessor_removes_stopwords_and_lowercases():
    processor = TextPreprocessor()
    result = processor.clean("The QUICK brown fox is running!")
    assert "the" not in result.split()
    assert result == result.lower()
    assert "quick" in result
