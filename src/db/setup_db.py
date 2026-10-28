import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'vendors.db')

def initialize_db():
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS blocked_vendors')
    
    cursor.execute('''
    CREATE TABLE blocked_vendors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        block_reason TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE
    )
    ''')
    
    sanctioned_vendors = [
        ("CompusGub", "Previous contract breach", "2024-01-10", "2028-10-10"),
        ("Luxury Coffee Inc", "Corruption and overpricing", "2025-06-01", None), 
        ("Central Stationery", "Document forgery", "2026-02-15", "2027-05-15")
    ]
    
    cursor.executemany('''
    INSERT INTO blocked_vendors (name, block_reason, start_date, end_date) 
    VALUES (?, ?, ?, ?)
    ''', sanctioned_vendors)
    
    connection.commit()
    connection.close()
    print("✅ Database updated: 'blocked_vendors' table created in English.")

if __name__ == "__main__":
    initialize_db()