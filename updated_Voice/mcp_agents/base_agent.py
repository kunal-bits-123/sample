from typing import Dict, Any, Optional
import json
import asyncio
from groq import AsyncGroq
import os

class BaseMCPAgent:
    def __init__(self, agent_name: str, groq_api_key: str, groq_model: str = "llama3-70b-8192"):
        self.agent_name = agent_name
        print(f"Initializing {agent_name} with model: {groq_model}")
        self.groq_client = AsyncGroq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.conversation_history = []

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        To be implemented by specific agents.
        """
        raise NotImplementedError("Each agent must implement process_message")

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        To be implemented by specific agents.
        """
        raise NotImplementedError("Each agent must implement _get_system_prompt")

    def _format_response(self, response_data: Dict[str, Any]) -> str:
        """
        Format the response data into a readable text format.
        """
        if response_data.get("status") == "error":
            return f"❌ Error: {response_data.get('error', 'Unknown error')}"
        
        operation = response_data.get("operation", "").lower()
        data = response_data.get("data", {})
        
        if operation == "retrieve":
            return self._format_retrieve_response(data)
        elif operation == "update":
            return self._format_update_response(data)
        elif operation == "create":
            return self._format_create_response(data)
        elif operation == "check_availability":
            return self._format_availability_response(data)
        elif operation == "schedule_appointment":
            return self._format_schedule_response(data)
        elif operation == "reschedule_appointment":
            return self._format_reschedule_response(data)
        elif operation == "cancel_appointment":
            return self._format_cancel_response(data)
        else:
            return str(data)

    def _format_retrieve_response(self, data: Dict[str, Any]) -> str:
        """Format patient retrieval response"""
        if not data:
            return "No patient data found."
        
        response = [
            f"📋 Patient Information for {data.get('name', 'Unknown')}",
            f"ID: {data.get('patient_id', 'N/A')}",
            "\nMedical History:",
            *[f"- {condition}" for condition in data.get('medical_history', [])],
            "\nCurrent Medications:",
            *[f"- {med['name']} ({med['dosage']}, {med['frequency']})" 
              for med in data.get('medications', [])],
            "\nAllergies:",
            *[f"- {allergy}" for allergy in data.get('allergies', [])]
        ]
        return "\n".join(response)

    def _format_update_response(self, data: Dict[str, Any]) -> str:
        """Format update response"""
        if not data:
            return "No update data provided."
        
        updates = data.get('updates', {})
        response = [
            f"✅ Updated Patient {data.get('patient_id', 'Unknown')}",
            "Changes made:",
            *[f"- {field}: {value}" for field, value in updates.items()]
        ]
        return "\n".join(response)

    def _format_create_response(self, data: Dict[str, Any]) -> str:
        """Format create response"""
        if not data:
            return "No patient data provided."
        
        return f"✅ Created new patient record for {data.get('name', 'Unknown')} (ID: {data.get('patient_id', 'N/A')})"

    def _format_availability_response(self, data: Dict[str, Any]) -> str:
        """Format availability response"""
        slots = data.get('available_slots', [])
        if not slots:
            return "No available appointments found."
        
        response = ["📅 Available Appointments:"]
        for slot in slots:
            response.append(f"- {slot['datetime']} with {slot['provider']} ({slot['duration']} minutes)")
        return "\n".join(response)

    def _format_schedule_response(self, data: Dict[str, Any]) -> str:
        """Format schedule response"""
        if not data:
            return "No appointment data provided."
        
        return (
            f"✅ Appointment Scheduled:\n"
            f"- ID: {data.get('appointment_id', 'N/A')}\n"
            f"- Patient: {data.get('patient_id', 'N/A')}\n"
            f"- Date/Time: {data.get('datetime', 'N/A')}\n"
            f"- Type: {data.get('type', 'N/A')}\n"
            f"- Provider: {data.get('provider', 'N/A')}\n"
            f"- Duration: {data.get('duration', 'N/A')} minutes"
        )

    def _format_reschedule_response(self, data: Dict[str, Any]) -> str:
        """Format reschedule response"""
        if not data:
            return "No appointment data provided."
        
        return (
            f"✅ Appointment Rescheduled:\n"
            f"- ID: {data.get('appointment_id', 'N/A')}\n"
            f"- New Date/Time: {data.get('datetime', 'N/A')}\n"
            f"- Provider: {data.get('provider', 'N/A')}\n"
            f"- Duration: {data.get('duration', 'N/A')} minutes"
        )

    def _format_cancel_response(self, data: Dict[str, Any]) -> str:
        """Format cancel response"""
        if not data:
            return "No appointment data provided."
        
        return f"✅ Appointment {data.get('appointment_id', 'N/A')} has been cancelled."

    async def _call_llm(self, messages: list) -> str:
        """
        Make a call to the Groq LLM with the given messages.
        """
        try:
            print(f"\n🤖 Calling LLM with model: {self.groq_model}")
            print(f"📝 Messages being sent: {json.dumps(messages, indent=2)}")
            
            chat_completion = await self.groq_client.chat.completions.create(
                messages=messages,
                model=self.groq_model,
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            response = chat_completion.choices[0].message.content.strip()
            print(f"✅ LLM Response received: {response}")
            
            # Try to parse the response as JSON
            try:
                json_response = json.loads(response)
                return json.dumps(json_response)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON response: {e}")
                return json.dumps({
                    "status": "error",
                    "error": f"Invalid JSON response: {str(e)}",
                    "data": None
                })
                
        except Exception as e:
            error_msg = f"Error calling LLM: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "data": None
            })

    def add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.
        """
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []

    async def close(self):
        """
        Cleanup resources.
        """
        pass  # Implement if needed when Groq client has explicit close method 