from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Configuration for EHR data
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'clinical_ehr'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', '')
}

# MongoDB Configuration for transcription data
MONGO_CONFIG = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', 27017)),
    'database': os.getenv('MONGO_DB', 'clinical_transcripts'),
    'collection': os.getenv('MONGO_COLLECTION', 'transcriptions')
}

# Groq API Configuration
GROQ_CONFIG = {
    'api_key': os.getenv('GROQ_API_KEY'),
    'model': os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')  # Default model
} 