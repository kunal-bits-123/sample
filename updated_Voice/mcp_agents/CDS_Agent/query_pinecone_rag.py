# query_pinecone_rag.py

import asyncio
import os
from dotenv import load_dotenv

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

import config_rag as config # Use the RAG-specific config
from updated_Voice.mcp_agents.CDS_Agent.nlp_handler import RAGNLPHandler # Use the RAG-specific NLP handler

class CDSRAGQueryEngine:
    def __init__(self):
        print("Initializing CDS RAG Query Engine...")
        load_dotenv() # Ensure environment variables are loaded

        self.nlp_handler = None
        self.vector_store = None
        self.embedding_model = None

        if not config.GROQ_API_KEY:
            print("ERROR: GROQ_API_KEY not found. Cannot initialize NLP Handler.")
            return
        if not config.PINECONE_API_KEY or not config.PINECONE_INDEX_NAME: # PINECONE_ENVIRONMENT is often inferred or part of host
            print("ERROR: Pinecone API Key or Index Name not configured. Cannot initialize Vector Store.")
            return
        if not config.EMBEDDING_MODEL_NAME:
            print("ERROR: Embedding model name not configured.")
            return

        try:
            self.nlp_handler = RAGNLPHandler(
                api_key=config.GROQ_API_KEY,
                model_identifier=config.GROQ_MODEL_IDENTIFIER
            )
            print("RAGNLPHandler initialized.")

            print(f"Initializing embedding model for querying: {config.EMBEDDING_MODEL_NAME}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
            
            print(f"Connecting to existing Pinecone index: {config.PINECONE_INDEX_NAME}")
            # LangChain's PineconeVectorStore will use PINECONE_API_KEY and PINECONE_ENVIRONMENT
            # from environment variables if they are set.
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=config.PINECONE_INDEX_NAME,
                embedding=self.embedding_model
            )
            print("Successfully connected to Pinecone index for querying.")
        except Exception as e:
            print(f"Error during CDS RAG Query Engine initialization: {e}")
            self.nlp_handler = None
            self.vector_store = None

    def _get_rag_system_prompt(self) -> str:
        """System prompt for the LLM when performing RAG for clinical Q&A."""
        return """You are a helpful AI assistant providing information for a clinical setting.
Your goal is to answer the user's question based ONLY on the provided context documents.
If the context documents do not contain sufficient information to answer the question directly, clearly state that the information is not available in the provided documents or that you cannot answer based on the context.
Do not use any external knowledge or make assumptions beyond the provided text. Be concise and factual.
IMPORTANT: Always include this disclaimer at the end of your response: "This information is for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
"""

    async def ask_question_async(self, clinical_query: str) -> str:
        if not self.vector_store or not self.nlp_handler:
            return "CDS RAG system is not properly initialized. Please check errors."

        print(f"\nUser Query: '{clinical_query}'")
        
        # 1. Retrieve relevant documents (context) from Pinecone
        retrieved_contexts_text = "No relevant context found in the knowledge base."
        try:
            print(f"Searching Pinecone for relevant documents (top_k={config.TOP_K_RESULTS_FROM_VECTOR_DB})...")
            # The embedding of the query is handled by vector_store.similarity_search
            retrieved_docs_langchain = self.vector_store.similarity_search(
                query=clinical_query, 
                k=config.TOP_K_RESULTS_FROM_VECTOR_DB
            )
            
            if retrieved_docs_langchain:
                print(f"Retrieved {len(retrieved_docs_langchain)} context chunks from Pinecone.")
                # for i, doc in enumerate(retrieved_docs_langchain):
                #     print(f"  Context Chunk {i+1} (source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:150]}...")
                retrieved_contexts_text = "\n\n---\nContext Document:\n".join([doc.page_content for doc in retrieved_docs_langchain])
            else:
                print("No relevant documents found in Pinecone for the query.")
        except Exception as e:
            print(f"Error during Pinecone similarity search: {e}")
            return "Sorry, I encountered an error while searching the knowledge base."

        # 2. Construct Augmented Prompt for the LLM
        llm_user_query_with_context = f"""Context from knowledge base:
{retrieved_contexts_text}
---
Based strictly on the context provided above, answer the following question:
Question: {clinical_query}
"""
        system_prompt_for_llm = self._get_rag_system_prompt()
        
        # 3. Call LLM via NLPHandler
        print(f"Sending augmented prompt to LLM (Groq)...")
        llm_answer = await self.nlp_handler.generate_answer_from_context_async(
            system_prompt=system_prompt_for_llm,
            user_query_with_context=llm_user_query_with_context
        )
        
        # 4. Ensure disclaimer is present
        disclaimer = "This information is for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        if llm_answer and disclaimer.lower() not in llm_answer.lower(): # Case-insensitive check
            llm_answer += f"\n\n{disclaimer}"
            
        return llm_answer if llm_answer else "I could not generate an answer based on the available information."

async def test_rag():
    engine = CDSRAGQueryEngine()
    if not engine.vector_store or not engine.nlp_handler:
        print("Exiting due to initialization errors.")
        return

    queries = [
        "What are the common symptoms of Type 2 Diabetes?",
        "Tell me about hypertension treatments.",
        "What causes the common cold?",
        "How is asthma diagnosed?",
        "What are the side effects of lisinopril?" # This might not be in your initial 4 articles
    ]

    for query in queries:
        print("-" * 50)
        answer = await engine.ask_question_async(query)
        print(f"\nLLM Answer:\n{answer}")
        print("-" * 50)
    
    # Important: Close async clients if NLPHandler manages them
    if hasattr(engine.nlp_handler, 'close_clients'):
        await engine.nlp_handler.close_clients()

if __name__ == "__main__":
    # Ensure your .env file has PINECONE_API_KEY, PINECONE_ENVIRONMENT, and GROQ_API_KEY.
    # Ensure you have run build_pinecone_index.py first to populate your Pinecone index.
    # Ensure your Pinecone index name and embedding model in config_rag.py are correct.
    
    # Create the directory for knowledge base if it doesn't exist, as a helper for user
    # This is more relevant for the build_vector_store.py script.
    if config.DOCUMENTS_PATH and not os.path.exists(config.DOCUMENTS_PATH):
        try:
            os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
            print(f"Created directory for knowledge base documents: {config.DOCUMENTS_PATH}")
            print(f"Please add your clinical articles (.txt, .pdf) to this directory and then run build_pinecone_index.py")
        except OSError as e:
            print(f"Error creating directory {config.DOCUMENTS_PATH}: {e}")
            print("Please create it manually.")
            
    asyncio.run(test_rag())
