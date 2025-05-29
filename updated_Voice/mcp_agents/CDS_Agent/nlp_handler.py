# nlp_handler_rag.py

import json
import asyncio
import os
from groq import Groq, AsyncGroq

class RAGNLPHandler:
    def __init__(self, api_key: str, model_identifier: str):
        """
        Initializes the NLP Handler for making calls to Groq for RAG.
        Args:
            api_key (str): The Groq API key.
            model_identifier (str): Specific model identifier for Groq.
        """
        self.api_key = api_key
        self.model_identifier = model_identifier

        if not self.api_key:
            raise ValueError("RAGNLPHandler: Groq API key not provided.")
        if not self.model_identifier:
            raise ValueError("RAGNLPHandler: Groq model identifier not provided.")

        try:
            # We primarily need the async client for query_pinecone_rag.py
            self.async_groq_client = AsyncGroq(api_key=self.api_key)
            print(f"RAGNLPHandler: Groq client configured for model: {self.model_identifier}")
        except Exception as e:
            print(f"RAGNLPHandler: Error initializing Groq client: {e}")
            raise

    async def generate_answer_from_context_async(self, system_prompt: str, user_query_with_context: str) -> str:
        """
        Sends a prompt (including retrieved context) to Groq and gets a textual answer.
        Args:
            system_prompt (str): The system prompt guiding the LLM for RAG.
            user_query_with_context (str): The user's original query augmented with retrieved context.
        Returns:
            str: The LLM's generated answer.
        """
        print(f"RAGNLPHandler: Sending to Groq (model: {self.model_identifier}): User query with context (first 100 chars): '{user_query_with_context[:100]}...'")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query_with_context}
        ]
        
        raw_response_content = "Error: No response from LLM."
        try:
            chat_completion = await self.async_groq_client.chat.completions.create(
                messages=messages,
                model=self.model_identifier,
                temperature=0.2,  # Lower for more factual, less creative RAG answers
                max_tokens=1500, # Allow for reasonably detailed answers
                # No response_format={"type": "json_object"} here, we want natural text
            )
            raw_response_content = chat_completion.choices[0].message.content
            if raw_response_content:
                response_text = raw_response_content.strip()
            else:
                print("RAGNLPHandler: Warning - Received empty content from Groq API.")
                response_text = "I found some information, but I'm having trouble formulating a concise answer right now."
            
            # print(f"RAGNLPHandler: Groq response: {response_text[:200]}...") # For debugging
            return response_text

        except Exception as e:
            print(f"RAGNLPHandler: Error during Groq API call: {e}")
            if "authentication" in str(e).lower() or "api key" in str(e).lower() or "permission" in str(e).lower():
                 return "There was an authentication error with the LLM service."
            # import traceback; traceback.print_exc() # Uncomment for detailed error
            return f"Sorry, I encountered an error trying to generate an answer: {str(e).split(':')[0]}"

    async def close_clients(self):
        if hasattr(self, 'async_groq_client'):
            print("RAGNLPHandler (Groq): AsyncGroq client cleanup (if SDK provides explicit close method).")
            # As of groq SDK 0.5.0, explicit close isn't documented for the client object.
            # It likely manages its internal httpx client.
            pass
