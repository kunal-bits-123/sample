import asyncio
import os
from dotenv import load_dotenv

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Replace this import with your actual NLP handler file/module name
# It should have a class RAGNLPHandler with async method generate_answer_from_context_async()
from updated_Voice.mcp_agents.CDS_Agent.nlp_handler import RAGNLPHandler  # <-- Your Groq handler module

# Configuration variables - these can be in config_rag.py or loaded from environment
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_IDENTIFIER = os.getenv("GROQ_MODEL_IDENTIFIER", "gpt-4o-mini")  # or your model ID
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_RESULTS_FROM_VECTOR_DB = int(os.getenv("TOP_K_RESULTS_FROM_VECTOR_DB", 4))

class CDSRAGQueryEngine:
    def __init__(self):
        print("Initializing CDS RAG Query Engine...")
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not found in environment.")
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            raise RuntimeError("Pinecone API Key or Index Name not configured.")
        if not EMBEDDING_MODEL_NAME:
            raise RuntimeError("Embedding model name not configured.")

        # Initialize NLP handler for Groq (or any LLM handler you have)
        self.nlp_handler = RAGNLPHandler(
            api_key=GROQ_API_KEY,
            model_identifier=GROQ_MODEL_IDENTIFIER
        )
        print("RAGNLPHandler initialized.")

        # Initialize embeddings model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"Embedding model initialized: {EMBEDDING_MODEL_NAME}")

        # Initialize Pinecone vector store (assumes PINECONE_API_KEY and ENV are set as env vars)
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embedding_model
        )
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

    def _get_rag_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant providing information for a clinical setting.\n"
            "Your goal is to answer the user's question based ONLY on the provided context documents.\n"
            "If the context documents do not contain sufficient information to answer the question directly, "
            "clearly state that the information is not available in the provided documents or that you cannot answer based on the context.\n"
            "Do not use any external knowledge or make assumptions beyond the provided text. Be concise and factual.\n"
            "IMPORTANT: Always include this disclaimer at the end of your response: "
            "\"This information is for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.\""
        )

    async def ask_question_async(self, clinical_query: str) -> str:
        print(f"\nReceived clinical query:\n{clinical_query}\n")
        try:
            retrieved_docs = self.vector_store.similarity_search(
                query=clinical_query,
                k=TOP_K_RESULTS_FROM_VECTOR_DB
            )
        except Exception as e:
            print(f"Error during Pinecone similarity search: {e}")
            return "Sorry, I encountered an error while searching the knowledge base."

        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."

        context_text = "\n\n---\nContext Document:\n".join([doc.page_content for doc in retrieved_docs])

        prompt = (
            f"Context from knowledge base:\n{context_text}\n---\n"
            f"Based strictly on the context provided above, answer the following question:\nQuestion: {clinical_query}"
        )
        system_prompt = self._get_rag_system_prompt()

        print("Sending augmented prompt to the LLM...")
        llm_answer = await self.nlp_handler.generate_answer_from_context_async(
            system_prompt=system_prompt,
            user_query_with_context=prompt
        )

        disclaimer = (
            "This information is for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        )

        if llm_answer and disclaimer.lower() not in llm_answer.lower():
            llm_answer += f"\n\n{disclaimer}"

        return llm_answer or "I could not generate an answer based on the available information."


async def main():
    try:
        engine = CDSRAGQueryEngine()
    except RuntimeError as e:
        print(f"Initialization failed: {e}")
        return

    print("CDS RAG Query Engine is ready.\n")

    # Dynamic query loop for demo - you can replace this with any interface
    while True:
        query = input("\nEnter clinical question (or 'exit' to quit): ").strip()
        if query.lower() in ('exit', 'quit'):
            print("Exiting.")
            break

        answer = await engine.ask_question_async(query)
        print("\nAnswer:\n", answer)

    # Cleanup if your NLP handler requires closing clients
    if hasattr(engine.nlp_handler, 'close_clients'):
        await engine.nlp_handler.close_clients()


class ClinicalDecisionAgent:
    """
    Adapter class to provide a standard agent interface for the clinical decision agent,
    delegating to the underlying CDSRAGQueryEngine (RAG-based clinical Q&A).
    """
    def __init__(self, groq_api_key=None, pinecone_api_key=None, pinecone_index_name=None, embedding_model_name=None, groq_model_identifier=None):
        # Allow override for testing, else use env/config
        self.engine = CDSRAGQueryEngine()

    async def analyze_case(self, query: str) -> str:
        """Analyze a clinical case using RAG Q&A."""
        return await self.engine.ask_question_async(query)

    async def check_guidelines(self, query: str) -> str:
        """Check clinical guidelines using RAG Q&A."""
        return await self.engine.ask_question_async(query)

    async def assess_risk(self, query: str) -> str:
        """Assess risk using RAG Q&A."""
        return await self.engine.ask_question_async(query)


if __name__ == "__main__":
    asyncio.run(main())
