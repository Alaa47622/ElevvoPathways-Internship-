import re

FORBIDDEN = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "VACUUM", "PRAGMA", "REINDEX", "TRUNCATE",
}

def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        return False, "SQL query is empty."
    if ";" in cleaned:
        return False, "Multiple SQL statements are not allowed."
    first = cleaned.split(None, 1)[0].upper()
    if first not in {"SELECT", "WITH"}:
        return False, f"Only SELECT/WITH queries are allowed; got {first}."
    tokens = set(re.findall(r"\b[A-Z_]+\b", cleaned.upper()))
    blocked = sorted(tokens & FORBIDDEN)
    if blocked:
        return False, f"Forbidden SQL operation(s): {', '.join(blocked)}."
    return True, "OK"
