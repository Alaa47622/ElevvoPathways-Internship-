import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT_DIR / "data" / "ecommerce.db"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
