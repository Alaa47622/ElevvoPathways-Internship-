from app.database.schema import get_schema
from app.tools.sql_tool import execute_sql

def test_schema():
    schema = get_schema()
    assert "TABLE customers" in schema
    assert "TABLE orders" in schema

def test_query():
    result = execute_sql("SELECT COUNT(*) AS customer_count FROM customers")
    assert result["ok"]
    assert result["rows"][0]["customer_count"] == 8
