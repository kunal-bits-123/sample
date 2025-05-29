import asyncio
import os
from dotenv import load_dotenv
from updated_Voice.mcp_agents.ehr_agent import EHRAgent
from updated_Voice.mcp_agents.medication_agent import MedicationAgent
from updated_Voice.mcp_agents.order_agent import OrderAgent
from updated_Voice.mcp_agents.clinical_decision_agent import ClinicalDecisionAgent
from updated_Voice.mcp_agents.scheduling_agent import SchedulingAgent
from updated_Voice.mcp_agents.analytics_agent import AnalyticsAgent
from updated_Voice.mcp_agents.inspector_agent import InspectorAgent
from updated_Voice.live_transcription import LiveTranscription
from typing import Optional, Dict, Any
import json
from database.fallback_handler import FallbackHandler

try:
    from database.postgres_handler import PostgresHandler
    USE_POSTGRES = True
except Exception as e:
    print("PostgreSQL not available, using fallback storage")
    USE_POSTGRES = False

# Load environment variables
load_dotenv()

class ClinicalVoiceAssistant:
    def __init__(self, groq_api_key: str):
        """Initialize the clinical voice assistant"""
        print("Initializing clinical voice assistant...")
        
        # Initialize database handlers
        try:
            if USE_POSTGRES:
                self.db = PostgresHandler()
                self.db.initialize_tables()
            else:
                self.db = FallbackHandler()
        except Exception as e:
            print(f"Error initializing database: {e}")
            print("Falling back to file-based storage")
            self.db = FallbackHandler()
        
        # Initialize all agents
        print("Initializing clinical agents...")
        # Prepare db_config for EHRAgent
        db_config = {
            'user': os.getenv('POSTGRES_USER', ''),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'database': os.getenv('POSTGRES_DB', ''),
            'host': os.getenv('POSTGRES_HOST', ''),
            'port': os.getenv('POSTGRES_PORT', '5432'),
        } if USE_POSTGRES else {}
        self.ehr_agent = EHRAgent(groq_api_key, db_config)
        self.medication_agent = MedicationAgent(groq_api_key)
        self.order_agent = OrderAgent(groq_api_key)
        self.clinical_decision_agent = ClinicalDecisionAgent(groq_api_key)
        self.scheduling_agent = SchedulingAgent(groq_api_key)
        self.analytics_agent = AnalyticsAgent(groq_api_key)
        self.inspector_agent = InspectorAgent(groq_api_key)
        print("✅ All agents initialized successfully\n")
        
        # Initialize live transcription with fallback storage
        self.transcriber = LiveTranscription(
            callback=self.handle_transcription,
            model_size="large-v2",  # Using a larger model for better accuracy
            use_fallback=not USE_POSTGRES
        )

    async def handle_transcription(self, text: str):
        """Handle transcribed text"""
        try:
            # Convert command to lowercase for easier matching
            command_lower = text.lower()
            
            # Determine which agent to use based on command keywords
            agent = None
            formatter = None
            
            # Check for clinical guidelines first
            if any(keyword in command_lower for keyword in ['guideline', 'clinical', 'latest', 'standard', 'protocol']):
                agent = self.clinical_decision_agent
                formatter = self.clinical_decision_agent._format_response
            # Check for patient history
            elif any(keyword in command_lower for keyword in ['history', 'record', 'patient', 'medical']):
                agent = self.ehr_agent
                formatter = self.ehr_agent._format_response
            # Check for medication
            elif any(keyword in command_lower for keyword in ['medication', 'drug', 'prescription', 'interaction']):
                agent = self.medication_agent
                formatter = self.medication_agent._format_response
            # Check for orders
            elif any(keyword in command_lower for keyword in ['order', 'test', 'lab', 'procedure']):
                agent = self.order_agent
                formatter = self.order_agent._format_response
            # Check for scheduling
            elif any(keyword in command_lower for keyword in ['schedule', 'appointment', 'available', 'book', 'cancel']):
                agent = self.scheduling_agent
                formatter = self.scheduling_agent._format_response
            # Check for analytics
            elif any(keyword in command_lower for keyword in ['report', 'trend', 'analytics', 'statistic']):
                agent = self.analytics_agent
                formatter = self.analytics_agent._format_response
            else:
                print("❌ I'm not sure how to handle that request. Please try rephrasing your question.")
                return
            
            # Process the command with the selected agent
            response = await agent.process_message(text)
            
            # Validate response with inspector agent
            validation = await self.inspector_agent.validate_response(agent.__class__.__name__, response)
            if validation.get('status') == 'error' or not validation.get('data', {}).get('validation_result', {}).get('is_valid', False):
                print("\n⚠️ Response validation failed:")
                print(self.inspector_agent._format_response(validation))
                return
            
            # Track agent state
            await self.inspector_agent.monitor_state(agent.__class__.__name__, response)
            
            # Format and display the response
            formatted_response = formatter(response)
            print(f"\n💬 Assistant: {formatted_response}")
            
        except Exception as e:
            error_message = f"❌ Error processing command: {str(e)}"
            print(error_message)

    async def run(self) -> None:
        """Run the voice assistant"""
        print("\n=== Clinical Voice Assistant ===")
        print("🎯 Ready to assist! (Press Ctrl+C to exit)\n")
        print("Example commands:")
        print("- 'Show me John Smith's medical history'")
        print("- 'Check interactions between Metformin and Lisinopril'")
        print("- 'Schedule an appointment for next week'")
        print("- 'Order a complete blood count test'")
        print("- 'Show me the latest clinical guidelines for diabetes'")
        print("- 'Generate a report on patient outcomes'\n")
        
        try:
            await self.transcriber.start()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the voice assistant"""
        self.transcriber.stop()
        self.db.close()
        print("\n👋 Goodbye!")

async def main():
    # Get Groq API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        return
    
    print("🔑 API Key Status: Present")
    
    # Initialize and run the voice assistant
    assistant = ClinicalVoiceAssistant(groq_api_key)
    try:
        await assistant.run()
    except KeyboardInterrupt:
        assistant.stop()

if __name__ == "__main__":
    asyncio.run(main()) 