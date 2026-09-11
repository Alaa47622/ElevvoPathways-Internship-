import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ecommerce.db"
DB.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.executescript("""
PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
""")

customers = [
(1,"Alice Johnson","USA","2025-01-12"),(2,"Omar Hassan","Egypt","2025-02-03"),
(3,"Sara Ahmed","Egypt","2025-02-18"),(4,"Liam Smith","UK","2025-03-01"),
(5,"Emma Brown","USA","2025-03-17"),(6,"Noah Wilson","Canada","2025-04-09"),
(7,"Mia Davis","USA","2025-04-25"),(8,"Youssef Ali","Egypt","2025-05-11")]
products = [
(1,"Laptop Pro 14","Electronics",1400.0),(2,"Wireless Mouse","Electronics",35.0),
(3,"Mechanical Keyboard","Electronics",110.0),(4,"USB-C Hub","Electronics",60.0),
(5,"Running Shoes","Sports",95.0),(6,"Yoga Mat","Sports",40.0),
(7,"Coffee Maker","Home",130.0),(8,"Desk Lamp","Home",55.0)]
orders = [
(1,1,"2026-09-04","completed"),(2,2,"2026-09-05","completed"),
(3,3,"2026-09-05","completed"),(4,4,"2026-09-06","completed"),
(5,5,"2026-09-07","completed"),(6,6,"2026-09-08","completed"),
(7,7,"2026-09-09","completed"),(8,8,"2026-09-10","completed"),
(9,1,"2026-09-10","completed"),(10,2,"2026-09-11","completed")]
items = [
(1,1,1,1,1400.0),(2,1,2,2,35.0),(3,2,3,1,110.0),(4,2,4,2,60.0),
(5,3,5,2,95.0),(6,3,6,1,40.0),(7,4,7,1,130.0),(8,4,8,2,55.0),
(9,5,1,1,1400.0),(10,5,3,1,110.0),(11,6,5,1,95.0),(12,6,2,1,35.0),
(13,7,8,3,55.0),(14,8,4,1,60.0),(15,9,3,2,110.0),(16,9,6,2,40.0),
(17,10,1,1,1400.0)]
cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
cur.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
conn.commit()
conn.close()
print(f"Created {DB}")
