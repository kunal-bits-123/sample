import json
import asyncio
import os
import logging
from groq import Groq, AsyncGroq
import config  # Ensure config.py contains GROQ_API_KEY and GROQ_MODEL_IDENTIFIER

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NLPHandler:
    def __init__(self, 
                 provider: str = "groq", 
                 api_key: str = None, 
                 groq_model_identifier: str = None):
        """
        Initialize the NLP handler with provider, API key, and Groq model identifier.
        Defaults read from config if not explicitly passed.
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", None)
        self.groq_model_identifier = groq_model_identifier or getattr(config, "GROQ_MODEL_IDENTIFIER", None)

        if self.provider != "groq":
            raise ValueError(f"NLP: This handler supports only 'groq' provider, got '{self.provider}'")
        if not self.api_key:
            raise ValueError("NLP (Groq): API key not found. Set GROQ_API_KEY in env or config.py or pass it.")
        if not self.groq_model_identifier:
            raise ValueError("NLP (Groq): Model identifier not specified. Set GROQ_MODEL_IDENTIFIER in config.py or pass it.")

        try:
            self.sync_groq_client = Groq(api_key=self.api_key)
            self.async_groq_client = AsyncGroq(api_key=self.api_key)
            logger.info(f"NLP: Groq provider configured with model '{self.groq_model_identifier}'")
        except Exception as e:
            logger.error(f"NLP (Groq): Failed to initialize Groq clients: {e}")
            raise ValueError(f"NLP (Groq): Initialization error: {e}")

    def _get_system_prompt_for_conversation_analysis(self) -> str:
        """System prompt specifying strict JSON clinical info extraction."""
        return (
            "You are an AI assistant specialized in analyzing transcribed conversations between a doctor and a patient. "
            "Your task is to extract key clinical information and structure it.\n\n"
            "You MUST return your response STRICTLY in JSON format. The JSON object should capture the following if present:\n"
            "- \"chief_complaint\": (string) The main reason for the patient's visit.\n"
            "- \"symptoms\": (list of strings) Symptoms mentioned by patient or doctor.\n"
            "- \"patient_history_notes\": (list of strings) Relevant medical history.\n"
            "- \"doctor_observations_assessment\": (list of strings) Doctor's key observations or assessments.\n"
            "- \"proposed_plan_orders\": (list of strings) Tests, medications, referrals, or follow-ups discussed.\n"
            "- \"key_takeaways\": (list of strings) Other important information or decisions.\n\n"
            "If a category is not mentioned, it can be an empty list or omitted.\n"
            "Focus strictly on factual clinical information.\n"
            "Do NOT include any explanatory text or markdown code blocks around JSON."
        )

    def _get_user_content_for_conversation(self, conversation_transcript: str) -> str:
        """Prepare user content prompt with the given conversation transcript."""
        return (
            f"Analyze the following transcribed conversation between a doctor and a patient:\n---\n"
            f"{conversation_transcript}\n---\n"
            "Extract the clinical information as per the specified JSON structure."
        )

    async def analyze_conversation_async(self, conversation_transcript: str) -> dict:
        """
        Asynchronously analyze the conversation transcript using the Groq model.
        Returns parsed JSON dict or error dict.
        """
        logger.info(f"NLP (Async Groq): Analyzing conversation snippet: '{conversation_transcript[:150]}...'")

        system_prompt = self._get_system_prompt_for_conversation_analysis()
        user_content = self._get_user_content_for_conversation(conversation_transcript)
        raw_response_content = ""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            chat_completion = await self.async_groq_client.chat.completions.create(
                model=self.groq_model_identifier,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            raw_response_content = chat_completion.choices[0].message.content
            if raw_response_content:
                raw_response_content = raw_response_content.strip()
            else:
                logger.warning("NLP (Async Groq): Received empty content from Groq API.")
                return {"error": "empty_llm_response", "details": "The LLM returned no content."}

            analysis_result = json.loads(raw_response_content)
            logger.info(f"NLP (Async Groq): Parsed analysis result: {analysis_result}")
            return analysis_result

        except json.JSONDecodeError as jde:
            logger.error(f"NLP (Async Groq): JSON decoding error: {jde}")
            return {
                "error": "JSONDecodeError",
                "details": str(jde),
                "raw_response": raw_response_content
            }
        except Exception as e:
            logger.exception("NLP (Async Groq): Exception during Groq processing.")
            if any(term in str(e).lower() for term in ["authentication", "api key", "permission"]):
                return {"error": "Groq API Authentication/Permission Error", "details": str(e)}
            return {"error": "LLM_processing_error", "details": str(e)}

    def analyze_conversation(self, conversation_transcript: str) -> dict:
        """
        Synchronous wrapper to run analyze_conversation_async with asyncio.run safely.
        Returns the analysis or error dict.
        """
        logger.info(f"NLP (Sync Wrapper): Analyzing conversation snippet: '{conversation_transcript[:150]}...'")
        try:
            return asyncio.run(self.analyze_conversation_async(conversation_transcript))
        except RuntimeError as re:
            logger.warning(f"NLP (Sync Wrapper): RuntimeError with asyncio.run(): {re}")
            return {"error": "RuntimeError", "details": str(re)}
        except Exception as e:
            logger.error(f"NLP (Sync Wrapper): Exception running async call: {e}")
            return {"error": "SyncWrapperError", "details": str(e)}

    async def close_clients(self):
        """
        Placeholder to close async clients if the Groq SDK ever supports it.
        """
        if hasattr(self, 'async_groq_client'):
            logger.info("NLP (Groq): Cleanup async Groq client (no explicit method currently).")
            # await self.async_groq_client.close()  # Uncomment if supported in future
            pass
