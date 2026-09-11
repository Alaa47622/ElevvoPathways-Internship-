import sqlite3
from pathlib import Path
from app.config import DATABASE_PATH

def get_connection(read_only: bool = True) -> sqlite3.Connection:
    path = Path(DATABASE_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run: uv run python scripts/init_db.py"
        )
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
