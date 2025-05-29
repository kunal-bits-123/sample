# config_rag.py

import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Document Source ---
# IMPORTANT: Create this folder and put your 4 disease articles here
# (e.g., ./rag_knowledge_base/diabetes.pdf, ./rag_knowledge_base/hypertension.txt)
DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Articles")

# --- Embedding Model ---
# This model's output dimension must match your Pinecone index dimension
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Produces 384-dimensional embeddings

# --- Pinecone Configuration ---
# These are loaded from .env by load_dotenv() above.
# We define them here so they are attributes of the config_rag module.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") 

# IMPORTANT: Create this index in your Pinecone console BEFORE running ingestion.
# Set its dimension to 384 (for all-MiniLM-L6-v2) and metric to "cosine".
PINECONE_INDEX_NAME = "cds" 

# --- LLM Configuration (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Choose a model available on Groq, e.g., "llama3-8b-8192", "mixtral-8x7b-32768"
GROQ_MODEL_IDENTIFIER = "llama3-8b-8192" 

# --- RAG Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS_FROM_VECTOR_DB = 2 # How many relevant chunks to retrieve
