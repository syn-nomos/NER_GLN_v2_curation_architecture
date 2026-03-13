import sqlite3
import os

# Initialize a dummy DB for testing
# Robust path resolution
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "data", "db", "annotations.db")

if os.path.exists(db_path):
    os.remove(db_path)

# Ensure dir exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)


conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create Tables (Manual run of schema for init)
# Updated to match new clean schema

c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_type TEXT, 
        status TEXT DEFAULT 'processed',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS sentences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        text TEXT NOT NULL,
        tokens TEXT, -- JSON
        dataset_split TEXT DEFAULT 'pending',
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        meta_info TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id INTEGER,
        label TEXT NOT NULL,
        start_offset INTEGER,
        end_offset INTEGER,
        token_start INTEGER,
        token_end INTEGER,
        text TEXT,
        source TEXT DEFAULT 'manual', 
        is_correct BOOLEAN DEFAULT NULL, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (sentence_id) REFERENCES sentences (id) ON DELETE CASCADE
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS memory_bank (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_text TEXT NOT NULL,
        label TEXT NOT NULL,
        frequency INTEGER DEFAULT 1,
        confidence_score REAL DEFAULT 1.0,
        context_snippet TEXT, 
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS active_learning_pool (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id INTEGER,
        uncertainty_score REAL,
        trigger_reason TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (sentence_id) REFERENCES sentences (id) ON DELETE CASCADE
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id INTEGER,
        action TEXT, 
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        details TEXT
    )
''')

# Insert Dummy Data
c.execute("INSERT INTO documents (filename, file_type) VALUES (?, ?)", ("dummy_import.pdf", "PDF"))
doc_id = c.lastrowid

sentences = [
    ("Ο Νόμος 4000/1950 περί τεντιμποϊσμού.", None),
    ("Εγκρίθηκε το Προεδρικό Διάταγμα 10/2025.", None),
    ("Η Αθήνα είναι πρωτεύουσα της Ελλάδας.", None)
]

for s in sentences:
    c.execute("INSERT INTO sentences (text, document_id, tokens) VALUES (?, ?, ?)", (s[0], doc_id, s[1]))

conn.commit()
conn.close()

print(f"Initialized dummy DB at {db_path}")
