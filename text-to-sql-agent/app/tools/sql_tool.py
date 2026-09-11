from typing import Any
from app.database.connection import get_connection
from app.database.security import validate_read_only_sql

def execute_sql(sql: str) -> dict[str, Any]:
    valid, reason = validate_read_only_sql(sql)
    if not valid:
        return {"ok": False, "error": reason, "rows": [], "columns": []}
    try:
        with get_connection(True) as conn:
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [d[0] for d in cursor.description or []]
        return {"ok": True, "error": None, "rows": rows, "columns": columns}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rows": [], "columns": []}
