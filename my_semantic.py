# # semantic_search.py
# import psycopg2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Connect to PostgreSQL
# conn = psycopg2.connect(
#     host="localhost",  # Use your PostgreSQL host
#     database="TestDB",  # Your database name
#     user="postgres",  # Your PostgreSQL username
#     password="Hitbullseye"  # Your PostgreSQL password
# )

# cursor = conn.cursor()

# def search_similar_documents(query, top_k=3):
#     """
#     Retrieve the top-k most similar documents from the database based on the query.
#     """
#     # Convert query into a vector (e.g., using TF-IDF)
#     vectorizer = TfidfVectorizer()
#     query_vector = vectorizer.fit_transform([query]).toarray()[0]

#     cursor.execute(f"SELECT id, text, embedding FROM documents")
#     rows = cursor.fetchall()

#     similarities = []
#     for row in rows:
#         doc_id, text, doc_embedding = row
#         doc_embedding = np.array(doc_embedding)  # Convert to numpy array
#         sim = cosine_similarity([query_vector], [doc_embedding])  # Cosine similarity
#         similarities.append((sim[0][0], text))  # Store similarity and text

#     # Sort results by similarity and return the top_k results
#     similarities.sort(key=lambda x: x[0], reverse=True)
#     return similarities[:top_k]



# import psycopg2
# import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity # Will likely remove this
# # from sklearn.feature_extraction.text import TfidfVectorizer # Will remove this

# # You'll need to add your embedding model here
# # Example using Sentence Transformers (install: pip install sentence-transformers)
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Or another suitable model

# # Connect to PostgreSQL
# conn = psycopg2.connect(
#     host="localhost",
#     database="TestDB",
#     user="postgres",
#     password="Hitbullseye"
# )

# cursor = conn.cursor()

# def search_similar_documents(query, top_k=3):
#     """
#     Retrieve the top-k most similar documents from the database based on the query.
#     """
#     # Step 1: Generate embedding for the query using the SAME model as for documents
#     query_embedding = embedding_model.encode(query) # This will be a numpy array

#     # Step 2: Use pgvector's similarity search
#     # For cosine similarity, pgvector uses '<=>' (negative inner product, so smaller is better)
#     # Or '<->' for L2 distance (smaller is better). Choose based on your preference/model.
#     # Let's assume you want to use cosine distance for now.
#     # Note: pgvector's cosine distance is 1 - cosine_similarity. So, lower value means higher similarity.
#     cursor.execute(
#         "SELECT id, text FROM documents ORDER BY embedding <-> %s LIMIT %s",
#         (query_embedding.tolist(), top_k) # Convert numpy array to list for psycopg2
#     )
#     rows = cursor.fetchall()

#     # The results are already sorted by similarity by the DB
#     # We just need to format them.
#     results = []
#     for row in rows:
#         doc_id, text = row
#         # We don't have the similarity score from the query itself here,
#         # but the order is by similarity. If you need the actual score,
#         # you'd need to calculate it explicitly or modify the DB query
#         # if pgvector had a direct way to return it with the distance.
#         # For practical purposes, ordered results are often sufficient.
#         results.append(text) # Just return the text for context
#     return results



import psycopg2
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # No longer needed if using pgvector operators
# from sklearn.feature_extraction.text import TfidfVectorizer # No longer needed, using SentenceTransformer

# You'll need to use the SAME embedding model here as you use for ingesting documents
from sentence_transformers import SentenceTransformer

# Initialize embedding model (should be the same as in VectorLLM.py)
# This assumes 'all-MiniLM-L6-v2' which outputs 384 dimensions.
# Ensure your DB column 'embedding' in 'documents' table is `vector(384)`.
# This model is loaded once globally in the Streamlit app.
# For this utility file, we can also load it, or consider passing it as an argument
# if you want to avoid duplicate model loading, though @st.cache_resource handles it.
# For simplicity, we will load it here as well, assuming it's a small model.
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model in my_semantic.py: {e}")
    print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
    # You might want to raise the exception or handle it more robustly depending on your app's needs


# Connect to PostgreSQL
# Note: In a production Streamlit app, you might want to cache this connection
# with @st.cache_resource if you're not managing it globally like this.
# For simplicity and to match previous examples, keeping it global here.
try:
    conn = psycopg2.connect(
        host="localhost",
        database="TestDB",
        user="postgres",
        password="Hitbullseye"
    )
    cursor = conn.cursor()
except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL in my_semantic.py: {e}")
    print("Please ensure PostgreSQL is running and your connection details are correct.")
    # You might want to re-raise or exit if DB connection is critical

def search_similar_documents(query, top_k=3):
    """
    Retrieve the top-k most similar documents from the database based on the query.
    Uses pgvector for efficient similarity search.
    """
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded in my_semantic.py. Cannot perform search.")
    if conn is None or cursor is None:
        raise RuntimeError("Database connection not established in my_semantic.py. Cannot perform search.")

    # Step 1: Generate embedding for the query using the SAME model as for documents
    query_embedding = embedding_model.encode(query) # This will be a numpy array

    # Step 2: Use pgvector's similarity search directly in the database
    # We use '<->' (L2 distance) or '<=>' (cosine distance).
    # Cosine distance is often preferred for semantic similarity.
    # For cosine distance with pgvector, the operator is '<=>'.
    # A smaller value means higher cosine similarity (as it's 1 - cosine_similarity).
    # CRITICAL FIX: Explicitly cast the input array to `vector` type using `::vector`
    try:
        cursor.execute(
            "SELECT text FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
            (query_embedding.tolist(), top_k) # Convert numpy array to list for psycopg2
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # `row` will contain just the text as per the SELECT clause
            results.append(row[0]) # row[0] is the text
        return results
    except psycopg2.Error as e:
        print(f"Error during database query in my_semantic.py: {e}")
        print("Ensure 'embedding' column is of type 'vector(384)' and 'pgvector' extension is enabled.")
        raise # Re-raise the exception for VectorLLM.py to handle and display