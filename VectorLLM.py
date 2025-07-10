
# # pdf_extractor.py

# # db_helper.py


# # llm_integration.py

# # app.py
# import streamlit as st
# from myPdf_extractor import extract_text_from_pdf
# from my_db_helper import insert_document_with_embedding
# from my_semantic import search_similar_documents
# from my_llm_integration import generate_response
# import numpy as np

# # Streamlit App UI
# st.title("Semantic Search and Q&A")

# # Input field for user query
# query = st.text_input("Enter your question:")

# # PDF upload functionality
# pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

# if pdf_file:
#     # Extract text from uploaded PDF
#     pdf_text = extract_text_from_pdf(pdf_file)
#     st.write("Text Extracted from PDF: ", pdf_text[:1000])  # Show first 1000 chars for verification
    
#     # Store extracted text and its embedding in the database
#     # For simplicity, we're using a random embedding here
#     embedding = np.random.rand(300)  # Replace with actual embedding generation
#     insert_document_with_embedding(pdf_text, embedding)

# if query:
#     # Retrieve relevant context from the database
#     results = search_similar_documents(query)
#     context = " ".join([text for _, text in results])  # Combine top results

#     # Generate a response using Ollama (integrate LLM response generation)
#     response = generate_response(context)

#     # Display the response
#     st.write("### Answer:")
#     st.write(response['text'])



# import streamlit as st
# import numpy as np
# import requests # Needed for catching RequestException from my_llm_integration

# # Import your local modules
# from myPdf_extractor import extract_text_from_pdf
# from my_db_helper import insert_document_with_embedding
# from my_semantic import search_similar_documents
# from my_llm_integration import generate_streaming_response

# # Import embedding model (e.g., Sentence Transformers)
# from sentence_transformers import SentenceTransformer

# # --- Streamlit App UI Configuration (MUST be the first Streamlit command in your script) ---
# st.set_page_config(page_title="Semantic Search and Q&A with Ollama", layout="wide")

# # Initialize embedding model globally and cache it with st.cache_resource
# # This prevents reloading the model every time the app reruns.
# @st.cache_resource
# def load_embedding_model():
#     st.info("Loading embedding model 'all-MiniLM-L6-v2'...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     st.success("Embedding model loaded!")
#     return model

# embedding_model = load_embedding_model()

# # --- Rest of your Streamlit App UI and Logic ---
# st.title("📄 Semantic Search and Q&A with Ollama")
# st.markdown("Upload a PDF, ask a question, and get answers powered by your local Ollama model and semantic search!")

# # Ollama Model Selection in sidebar for user flexibility
# st.sidebar.header("Ollama Configuration")
# ollama_model_name = st.sidebar.text_input(
#     "Ollama Model Name (e.g., llama3, mistral)",
#     value="llama3",
#     help="Ensure this model is downloaded and running in your local Ollama instance (e.g., via 'ollama pull llama3')."
# )
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "This app demonstrates a Retrieval Augmented Generation (RAG) pipeline. "
#     "It extracts text from PDFs, creates embeddings, stores them in a PostgreSQL vector DB, "
#     "retrieves relevant context, and uses a local Ollama LLM to answer questions. "
# )

# # --- PDF Upload and Ingestion ---
# st.header("1. Upload Your Document")
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file:
#     with st.spinner("Extracting text from PDF..."):
#         try:
#             pdf_text = extract_text_from_pdf(pdf_file)
#             st.success("Text extraction complete!")
#             if len(pdf_text) > 1000:
#                 st.expander("View Extracted Text (first 1000 chars)").write(pdf_text[:1000] + "...")
#             else:
#                 st.expander("View Extracted Text").write(pdf_text)

#             # Generate embedding for the PDF content and store it
#             with st.spinner("Generating embedding and storing document..."):
#                 # For simplicity, embedding the whole text for now.
#                 # For very large PDFs, you should chunk the text and embed each chunk.
#                 document_embedding = embedding_model.encode(pdf_text)

#                 # --- CRITICAL CHECK FOR EMBEDDING DIMENSION CONSISTENCY ---
#                 # 'all-MiniLM-L6-v2' outputs 384-dimensional embeddings.
#                 # Your DB table's 'embedding' column MUST be `vector(384)`.
#                 # If it's `vector(300)`, you need to update my_db_helper.py and your DB.
#                 expected_dimension = embedding_model.get_sentence_embedding_dimension()
#                 if document_embedding.shape[0] != expected_dimension:
#                     st.error(
#                         f"Embedding dimension mismatch! The model outputs {document_embedding.shape[0]} dimensions, "
#                         f"but your PostgreSQL 'documents' table's 'embedding' column might be expecting a different size (e.g., 300). "
#                         f"Please ensure `embedding vector({expected_dimension})` is used in `my_db_helper.py` and your database schema."
#                     )
#                 else:
#                     insert_document_with_embedding(pdf_text, document_embedding)
#                     st.success("Document embedded and stored in PostgreSQL!")
#                     st.session_state['pdf_uploaded'] = True # Use session state to remember upload
#         except Exception as e:
#             st.error(f"An error occurred during PDF processing: {e}")
#             st.info("Please ensure the PDF is not corrupted and PyMuPDF is installed correctly (`pip install pymupdf`).")
# else:
#     st.session_state['pdf_uploaded'] = False


# # --- Ask a Question and Generate Response ---
# st.header("2. Ask a Question")
# # Disable query input until a PDF is uploaded
# query = st.text_input("Enter your question here:", disabled=not st.session_state.get('pdf_uploaded', False))

# if query:
#     if not st.session_state.get('pdf_uploaded', False):
#         st.warning("Please upload a PDF document first before asking a question.")
#     else:
#         st.header("3. Answer")
#         with st.spinner("Searching for relevant context in the database..."):
#             try:
#                 # search_similar_documents returns a list of texts directly
#                 relevant_texts = search_similar_documents(query)
#                 if not relevant_texts:
#                     st.warning("No relevant documents found in the database for your query. Try a different question or upload more relevant PDFs.")
#                     st.stop() # Stop execution if no context found
#                 context = "\n\n".join(relevant_texts)
#                 st.expander("View Retrieved Context").write(context)
#                 st.success("Context retrieved!")
#             except Exception as e:
#                 st.error(f"An error occurred during semantic search: {e}")
#                 st.info("Please ensure your PostgreSQL database is running and accessible, and `my_semantic.py` is configured correctly (especially the embedding model).")
#                 st.stop() # Stop execution on DB error


#         # Apply a prompt that structures the output for the LLM
#         system_prompt = f"""You are a helpful AI assistant. Your goal is to provide concise and accurate answers based *only* on the provided context.
#         If the answer cannot be found within the given context, clearly state that you do not have enough information from the provided text.
#         Structure your response in a clear, easy-to-read format. Use bullet points or numbered lists for multiple distinct points.
#         Do not make up information or elaborate beyond the context.

#         ---
#         Context:
#         {context}
#         ---
#         Question: {query}
#         Answer:
#         """

#         st.subheader("Generated Response:")
#         response_container = st.empty() # Placeholder for the streaming response
#         full_response_content = ""

#         with st.spinner(f"Generating response with Ollama model '{ollama_model_name}'..."):
#             try:
#                 # Iterate over the generator from my_llm_integration
#                 for chunk in generate_streaming_response(ollama_model_name, system_prompt):
#                     full_response_content += chunk
#                     # Update the Streamlit markdown component with each new chunk + a blinking cursor
#                     response_container.markdown(full_response_content + "▌")
#                 response_container.markdown(full_response_content) # Final display without cursor
#                 st.success("Response generated!")
#             except requests.exceptions.RequestException as e:
#                 st.error(f"Failed to get response from Ollama: {e}")
#                 st.warning(f"Please ensure the Ollama server is running at `http://localhost:11434` and the model '{ollama_model_name}' is downloaded. "
#                            "You can check downloaded models by running `ollama list` in your terminal.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred during LLM response generation: {e}")



import streamlit as st
import numpy as np
import requests # Needed for catching RequestException from my_llm_integration

# Import your local modules
from myPdf_extractor import extract_text_from_pdf
from my_db_helper import insert_document_with_embedding
from my_semantic import search_similar_documents
from my_llm_integration import generate_streaming_response

# Import embedding model (e.g., Sentence Transformers)
from sentence_transformers import SentenceTransformer

# --- Streamlit App UI Configuration (MUST be the first Streamlit command in your script) ---
st.set_page_config(page_title="Semantic Search and Q&A with Ollama", layout="wide")

# Initialize embedding model globally and cache it with st.cache_resource
# This prevents reloading the model every time the app reruns.
@st.cache_resource
def load_embedding_model():
    st.info("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.success("Embedding model loaded!")
    return model

embedding_model = load_embedding_model()

# --- Rest of your Streamlit App UI and Logic ---
st.title("📄 Semantic Search and Q&A with Ollama")
st.markdown("Upload a PDF, ask a question, and get answers powered by your local Ollama model and semantic search!")

# Ollama Model Selection in sidebar for user flexibility
st.sidebar.header("Ollama Configuration")
ollama_model_name = st.sidebar.text_input(
    "Ollama Model Name (e.g., llama3, mistral)",
    value="llama3.2:1b",
    help="Ensure this model is downloaded and running in your local Ollama instance (e.g., via 'ollama pull llama3')."
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates a Retrieval Augmented Generation (RAG) pipeline. "
    "It extracts text from PDFs, creates embeddings, stores them in a PostgreSQL vector DB, "
    "retrieves relevant context, and uses a local Ollama LLM to answer questions. "
)

# --- PDF Upload and Ingestion ---
st.header("1. Upload Your Document")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    with st.spinner("Extracting text from PDF..."):
        try:
            # Call the updated extract_text_from_pdf with the file-like object
            pdf_text = extract_text_from_pdf(pdf_file)
            st.success("Text extraction complete!")
            if len(pdf_text) > 1000:
                st.expander("View Extracted Text (first 1000 chars)").write(pdf_text[:1000] + "...")
            else:
                st.expander("View Extracted Text").write(pdf_text)

            # Generate embedding for the PDF content and store it
            with st.spinner("Generating embedding and storing document..."):
                # For simplicity, embedding the whole text for now.
                # For very large PDFs, you should chunk the text and embed each chunk.
                document_embedding = embedding_model.encode(pdf_text)

                # --- CRITICAL CHECK FOR EMBEDDING DIMENSION CONSISTENCY ---
                # 'all-MiniLM-L6-v2' outputs 384-dimensional embeddings.
                # Your DB table's 'embedding' column MUST be `vector(384)`.
                # If it's `vector(300)`, you need to update my_db_helper.py and your DB.
                expected_dimension = embedding_model.get_sentence_embedding_dimension()
                if document_embedding.shape[0] != expected_dimension:
                    st.error(
                        f"Embedding dimension mismatch! The model outputs {document_embedding.shape[0]} dimensions, "
                        f"but your PostgreSQL 'documents' table's 'embedding' column might be expecting a different size (e.g., 300). "
                        f"Please ensure `embedding vector({expected_dimension})` is used in `my_db_helper.py` and your database schema."
                    )
                else:
                    insert_document_with_embedding(pdf_text, document_embedding)
                    st.success("Document embedded and stored in PostgreSQL!")
                    st.session_state['pdf_uploaded'] = True # Use session state to remember upload
        except Exception as e:
            st.error(f"An error occurred during PDF processing: {e}")
            st.info("Please ensure the PDF is not corrupted and PyMuPDF is installed correctly (`pip install pymupdf`).")
else:
    st.session_state['pdf_uploaded'] = False


# --- Ask a Question and Generate Response ---
st.header("2. Ask a Question")
# Disable query input until a PDF is uploaded
query = st.text_input("Enter your question here:", disabled=not st.session_state.get('pdf_uploaded', False))

if query:
    if not st.session_state.get('pdf_uploaded', False):
        st.warning("Please upload a PDF document first before asking a question.")
    else:
        st.header("3. Answer")
        with st.spinner("Searching for relevant context in the database..."):
            try:
                # search_similar_documents returns a list of texts directly
                relevant_texts = search_similar_documents(query)
                if not relevant_texts:
                    st.warning("No relevant documents found in the database for your query. Try a different question or upload more relevant PDFs.")
                    st.stop() # Stop execution if no context found
                context = "\n\n".join(relevant_texts)
                st.expander("View Retrieved Context").write(context)
                st.success("Context retrieved!")
            except Exception as e:
                st.error(f"An error occurred during semantic search: {e}")
                st.info("Please ensure your PostgreSQL database is running and accessible, and `my_semantic.py` is configured correctly (especially the embedding model).")
                st.stop() # Stop execution on DB error


        # Apply a prompt that structures the output for the LLM
        system_prompt = f"""You are a helpful AI assistant. Your goal is to provide concise and accurate answers based *only* on the provided context.
        If the answer cannot be found within the given context, clearly state that you do not have enough information from the provided text.
        Structure your response in a clear, easy-to-read format. Use bullet points or numbered lists for multiple distinct points.
        Do not make up information or elaborate beyond the context.

        ---
        Context:
        {context}
        ---
        Question: {query}
        Answer:
        """

        st.subheader("Generated Response:")
        response_container = st.empty() # Placeholder for the streaming response
        full_response_content = ""

        with st.spinner(f"Generating response with Ollama model '{ollama_model_name}'..."):
            try:
                # Iterate over the generator from my_llm_integration
                for chunk in generate_streaming_response(ollama_model_name, system_prompt):
                    full_response_content += chunk
                    # Update the Streamlit markdown component with each new chunk + a blinking cursor
                    response_container.markdown(full_response_content + "▌")
                response_container.markdown(full_response_content) # Final display without cursor
                st.success("Response generated!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to get response from Ollama: {e}")
                st.warning(f"Please ensure the Ollama server is running at `http://localhost:11434` and the model '{ollama_model_name}' is downloaded. "
                           "You can check downloaded models by running `ollama list` in your terminal.")
            except Exception as e:
                st.error(f"An unexpected error occurred during LLM response generation: {e}")