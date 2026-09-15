from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "BBC_News.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

N_TOPICS = 5
N_TOP_WORDS = 10

# Vectorizer controls
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95

# Reproducibility
RANDOM_STATE = 42

# Training controls
LDA_MAX_ITER = 20
NMF_MAX_ITER = 500


def create_directories() -> None:
    """Create folders required by the pipeline."""
    for directory in [
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
