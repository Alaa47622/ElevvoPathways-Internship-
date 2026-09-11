from app.database.security import validate_read_only_sql

def test_select_allowed():
    assert validate_read_only_sql("SELECT * FROM customers")[0]

def test_with_allowed():
    assert validate_read_only_sql("WITH x AS (SELECT 1 n) SELECT n FROM x")[0]

def test_delete_blocked():
    assert not validate_read_only_sql("DELETE FROM customers")[0]

def test_multiple_statements_blocked():
    assert not validate_read_only_sql("SELECT 1; DROP TABLE customers")[0]
