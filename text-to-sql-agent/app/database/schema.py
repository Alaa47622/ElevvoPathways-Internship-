from app.database.connection import get_connection

def get_schema() -> str:
    with get_connection(True) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        chunks = []
        for row in tables:
            table = row["name"]
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            cols = ", ".join(
                f'{c["name"]} {c["type"]}' + (" PK" if c["pk"] else "")
                for c in columns
            )
            fk_text = ""
            if fks:
                fk_text = " | FKs: " + ", ".join(
                    f'{fk["from"]}->{fk["table"]}.{fk["to"]}' for fk in fks
                )
            chunks.append(f"TABLE {table} ({cols}){fk_text}")
        return "\n".join(chunks)
