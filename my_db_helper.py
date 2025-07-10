import psycopg2
import numpy as np

# Connect to PostgreSQL (replace with your database credentials)
conn = psycopg2.connect(
    host="localhost",
    database="TestDB",
    user="postgres",
    password="Hitbullseye"
)

cursor = conn.cursor()

# Create table with vector column if it does not exist
cursor.execute("""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding vector(384) -- <<< ENSURE THIS IS 384
)
""")
conn.commit()

def insert_document_with_embedding(text, embedding):
    """
    Insert document text and its embedding into PostgreSQL database.
    """
    cursor.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s)",
                   (text, embedding.tolist()))  # Convert numpy array to list
    conn.commit()