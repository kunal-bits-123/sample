# build_pinecone_index.py

import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore # For Pinecone integration

# It's good practice to use a specific config for this script if it diverges
# from the main app's config, but for now, we can import from the shared one.
import config_rag as config 

def load_all_documents(folder_path: str) -> list:
    """Loads .txt and .pdf documents from the specified folder and its subdirectories."""
    all_docs = []
    print(f"Attempting to load documents from: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: Document folder not found at '{folder_path}'")
        return []

    # Load .txt files
    try:
        txt_loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt", # Look in subdirectories too
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True},
            show_progress=True,
            use_multithreading=True,
            silent_errors=True # Suppress individual file loading errors, check total count later
        )
        txt_docs = txt_loader.load()
        if txt_docs:
            all_docs.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} .txt documents.")
    except Exception as e:
        print(f"Error loading .txt files from {folder_path}: {e}")

    # Load .pdf files
    try:
        pdf_loader = DirectoryLoader(
            folder_path,
            glob="**/*.pdf", # Look in subdirectories too
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            all_docs.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} .pdf documents.")
    except Exception as e:
        print(f"Error loading .pdf files from {folder_path}: {e}")
    
    if not all_docs:
        print(f"Warning: No documents were loaded from {folder_path}. Check path, file types, and permissions.")
    return all_docs

def main_ingestion():
    print("Starting document ingestion process for Pinecone...")
    # load_dotenv() is called in config_rag.py, so PINECONE_API_KEY etc. should be available
    # via os.getenv() if LangChain's PineconeVectorStore needs them explicitly and doesn't
    # pick them up from its own dotenv loading. Often, LangChain integrations do this.

    # Explicitly check for keys from config, as LangChain might not raise clear errors if they are missing
    if not config.PINECONE_API_KEY: # Check the one loaded into config by dotenv
        print("Error: PINECONE_API_KEY not found. Please set it in your .env file.")
        return
    # PINECONE_ENVIRONMENT is also often picked up by LangChain from env vars
    # if not config.PINECONE_ENVIRONMENT:
    #     print("Error: PINECONE_ENVIRONMENT not found. Please set it in your .env file.")
    #     return
    if not config.PINECONE_INDEX_NAME:
        print("Error: PINECONE_INDEX_NAME not set in config_rag.py.")
        return

    # 1. Load Documents
    documents = load_all_documents(config.DOCUMENTS_PATH)
    if not documents:
        print("No documents loaded. Exiting ingestion.")
        return
    print(f"Successfully loaded {len(documents)} documents in total.")

    # 2. Split Documents into Chunks
    print(f"Splitting documents into chunks (size: {config.CHUNK_SIZE}, overlap: {config.CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(split_chunks)} chunks.")

    if not split_chunks:
        print("No chunks to process after splitting. Exiting.")
        return

    # 3. Initialize Embedding Model
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        print("Embedding model initialized.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # 4. Upsert to Pinecone
    # LangChain's PineconeVectorStore will use PINECONE_API_KEY and PINECONE_ENVIRONMENT
    # from environment variables if not passed directly.
    # Ensure your Pinecone index (config.PINECONE_INDEX_NAME) exists and
    # its dimension matches the embedding model (e.g., 384 for all-MiniLM-L6-v2).
    print(f"Upserting {len(split_chunks)} chunks to Pinecone index: {config.PINECONE_INDEX_NAME}...")
    start_time = time.time()
    try:
        # This will add documents to an existing index.
        # If the index doesn't exist, PineconeVectorStore.from_documents usually tries to create it,
        # but it's better to pre-create it in the Pinecone console with the correct dimensions.
        vector_store = PineconeVectorStore.from_documents(
            documents=split_chunks,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME
            # Pinecone API key and environment are typically read from environment variables by LangChain
            # pinecone_api_key=config.PINECONE_API_KEY, # Can be explicit
            # pinecone_env=config.PINECONE_ENVIRONMENT  # Can be explicit
        )
        print("Successfully upserted documents to Pinecone.")
        # To add more documents later to an existing index:
        # existing_vector_store = PineconeVectorStore.from_existing_index(config.PINECONE_INDEX_NAME, embeddings)
        # existing_vector_store.add_documents(new_chunks)

    except Exception as e:
        print(f"Error upserting documents to Pinecone: {e}")
        print("Please check your Pinecone API key, environment/host, index name, and index dimension.")
        print("Ensure the index exists in Pinecone and its dimension matches your embedding model.")
    
    end_time = time.time()
    print(f"Ingestion process finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # This script is meant to be run once to populate your vector store,
    # or again if you update your source documents.
    # Ensure your .env file has PINECONE_API_KEY and PINECONE_ENVIRONMENT.
    # Ensure PINECONE_INDEX_NAME in config_rag.py matches an index you've created
    # in Pinecone with the correct vector dimension (e.g., 384 for all-MiniLM-L6-v2).
    # Ensure DOCUMENTS_PATH in config_rag.py points to your folder of articles.
    
    # Create the directory for knowledge base if it doesn't exist, as a helper for user
    if config.DOCUMENTS_PATH and not os.path.exists(config.DOCUMENTS_PATH):
        try:
            os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
            print(f"Created directory for knowledge base documents: {config.DOCUMENTS_PATH}")
            print(f"Please add your clinical articles (.txt, .pdf) to this directory.")
        except OSError as e:
            print(f"Error creating directory {config.DOCUMENTS_PATH}: {e}")
            print("Please create it manually.")

    main_ingestion()
