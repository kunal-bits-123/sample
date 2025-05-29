from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime, timedelta
from .base_agent import BaseMCPAgent
import logging

class SchedulingAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, debug: bool = False):
        super().__init__("Scheduling Agent", groq_api_key)
        self.schedule_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test_data', 'medical', 'schedule.json')
        self._load_schedule_data()
        self.debug = debug
        self.logger = logging.getLogger("SchedulingAgent")

    def _load_schedule_data(self):
        """Load schedule data from local file"""
        try:
            with open(self.schedule_data_path, 'r') as f:
                self.schedule_data = json.load(f)
        except FileNotFoundError:
            self.schedule_data = {
                "appointments": [
                    {
                        "id": "A001",
                        "patient_id": "P001",
                        "datetime": "2024-03-27 10:00:00",
                        "type": "Follow-up",
                        "duration": 30,
                        "status": "scheduled",
                        "provider": "Dr. Smith"
                    }
                ]
            }
            os.makedirs(os.path.dirname(self.schedule_data_path), exist_ok=True)
            with open(self.schedule_data_path, 'w') as f:
                json.dump(self.schedule_data, f, indent=2)

    def _save_schedule_data(self):
        """Save schedule data to local file"""
        with open(self.schedule_data_path, 'w') as f:
            json.dump(self.schedule_data, f, indent=2)

    def _get_system_prompt(self) -> str:
        return """You are a Scheduling Agent responsible for managing appointments and schedules.
        You can perform the following operations:
        - search_appointments: Search for available appointments
        - check_availability: Check available appointment slots
        - schedule_appointment: Schedule a new appointment
        - reschedule_appointment: Reschedule an existing appointment
        - cancel_appointment: Cancel an appointment
        
        IMPORTANT: You must respond with a valid JSON object. The response must be parseable JSON.
        CRITICAL JSON RULES:
        1. NO escaped newlines (\\n) in any string values
        2. NO escaped quotes (\\") in any string values
        3. NO trailing commas
        4. NO comments
        5. NO extra whitespace
        6. All arrays and objects must be properly closed
        7. All string values must be properly quoted
        8. All dates must be in YYYY-MM-DD format
        9. All times must be in HH:MM AM/PM format
        10. Provider names must not contain newlines or special characters
        11. The error field must ONLY appear at the root level
        12. Do not add any extra fields to the response
        
        For search_appointments operation (when user asks about available appointments), use this exact format:
        {
            "operation": "search_appointments",
            "status": "success",
            "data": {
                "available_appointments": [
                    {
                        "date": "2024-03-13",
                        "provider": "Dr. Smith",
                        "timeslot": "09:00 AM"
                    }
                ]
            },
            "error": null
        }
        
        For check_availability operation, use this exact format:
        {
            "operation": "check_availability",
            "status": "success",
            "data": {
                "available_slots": [
                    {
                        "date": "2024-03-13",
                        "time": "09:00 AM",
                        "provider": "Dr. Smith",
                        "duration": 30
                    }
                ]
            },
            "error": null
        }
        
        For schedule_appointment operation, use this exact format:
        {
            "operation": "schedule_appointment",
            "status": "success",
            "data": {
                "appointment_id": "A001",
                "patient_id": "P001",
                "date": "2024-03-13",
                "time": "09:00 AM",
                "type": "Follow-up",
                "provider": "Dr. Smith",
                "duration": 30
            },
            "error": null
        }
        
        For reschedule_appointment operation, use this exact format:
        {
            "operation": "reschedule_appointment",
            "status": "success",
            "data": {
                "appointment_id": "A001",
                "old_date": "2024-03-13",
                "old_time": "09:00 AM",
                "new_date": "2024-03-14",
                "new_time": "10:00 AM",
                "provider": "Dr. Smith"
            },
            "error": null
        }
        
        For cancel_appointment operation, use this exact format:
        {
            "operation": "cancel_appointment",
            "status": "success",
            "data": {
                "appointment_id": "A001",
                "status": "cancelled"
            },
            "error": null
        }
        
        For errors, use this exact format:
        {
            "operation": "<operation_type>",
            "status": "error",
            "data": null,
            "error": "Error message here"
        }
        
        When user asks about available appointments or searching for appointments:
        1. Always use the search_appointments operation
        2. Return a list of available appointments with dates, providers, and timeslots
        3. Use proper date format (YYYY-MM-DD)
        4. Use proper time format (HH:MM AM/PM)
        5. Include provider names without any special characters
        6. Do not include any other fields in the response
        7. The error field must ONLY appear at the root level
        8. Do not add error field inside the appointments array"""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Process this request: {message}"}
            ]
            
            # Get LLM response
            response = await self._call_llm(messages)
            
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
                print(f"[ERROR] Raw response: {response}")
                # Try to fix common JSON formatting issues
                try:
                    # Remove any escaped newlines and fix quotes
                    fixed_response = response.replace('\\n', ' ').replace('\\"', '"')
                    # Remove any error fields from inside arrays
                    fixed_response = fixed_response.replace('"error": null,', '').replace('"error": null', '')
                    response_data = json.loads(fixed_response)
                except:
                    return {
                        "operation": "unknown",
                        "status": "error",
                        "error": f"Invalid JSON response from LLM: {str(e)}",
                        "data": None
                    }

            if self.debug:
                print("[DEBUG] SchedulingAgent LLM response:", json.dumps(response_data, indent=2))

            # Process based on operation type
            operation = response_data.get("operation", "")
            data = response_data.get("data", {})
            error = response_data.get("error", None)
            
            if error:
                return {
                    "operation": operation,
                    "status": "error",
                    "data": None,
                    "error": error
                }
            
            if not data:
                return {
                    "operation": operation,
                    "status": "error",
                    "data": None,
                    "error": f"No data provided for operation: {operation}"
                }
            
            # Clean up any error fields that might have been added inside arrays
            if "available_appointments" in data:
                data["available_appointments"] = [
                    {k: v for k, v in appt.items() if k != "error"}
                    for appt in data["available_appointments"]
                ]
            
            # Return the complete response with operation type
            return {
                "operation": operation,
                "status": "success",
                "data": data,
                "error": None
            }

        except Exception as e:
            print(f"[ERROR] Exception in process_message: {str(e)}")
            return {
                "operation": "unknown",
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_check_availability(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle check availability request"""
        try:
            # Get available slots from schedule data
            available_slots = []
            for appointment in self.schedule_data.get("appointments", []):
                if appointment.get("status") == "available":
                    available_slots.append({
                        "date": appointment["datetime"].split()[0],
                        "time": appointment["datetime"].split()[1],
                        "provider": appointment["provider"],
                        "duration": appointment["duration"]
                    })
            
            return {
                "status": "success",
                "data": {
                    "available_slots": available_slots
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_schedule_appointment(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle schedule appointment request"""
        try:
            # Create new appointment in schedule data
            appointment_id = f"A{len(self.schedule_data['appointments']) + 1:03d}"
            new_appointment = {
                "id": appointment_id,
                "patient_id": response_data.get("patient_id", "P001"),
                "datetime": f"{response_data.get('date')} {response_data.get('time')}",
                "type": response_data.get("type", "Follow-up"),
                "duration": response_data.get("duration", 30),
                "status": "scheduled",
                "provider": response_data.get("provider", "Dr. Smith")
            }
            
            self.schedule_data["appointments"].append(new_appointment)
            self._save_schedule_data()
            
            return {
                "status": "success",
                "data": new_appointment,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_reschedule_appointment(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reschedule appointment request"""
        try:
            appointment_id = response_data.get("appointment_id")
            new_datetime = f"{response_data.get('new_date')} {response_data.get('new_time')}"
            
            # Find and update appointment
            for appointment in self.schedule_data["appointments"]:
                if appointment["id"] == appointment_id:
                    old_datetime = appointment["datetime"]
                    appointment["datetime"] = new_datetime
                    self._save_schedule_data()
                    
                    return {
                        "status": "success",
                        "data": {
                            "appointment_id": appointment_id,
                            "old_date": old_datetime.split()[0],
                            "old_time": old_datetime.split()[1],
                            "new_date": response_data.get("new_date"),
                            "new_time": response_data.get("new_time"),
                            "provider": appointment["provider"]
                        },
                        "error": None
                    }
            
            return {
                "status": "error",
                "error": f"Appointment {appointment_id} not found",
                "data": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_cancel_appointment(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cancel appointment request"""
        try:
            appointment_id = response_data.get("appointment_id")
            
            # Find and update appointment status
            for appointment in self.schedule_data["appointments"]:
                if appointment["id"] == appointment_id:
                    appointment["status"] = "cancelled"
                    self._save_schedule_data()
                    
                    return {
                        "status": "success",
                        "data": {
                            "appointment_id": appointment_id,
                            "status": "cancelled"
                        },
                        "error": None
                    }
            
            return {
                "status": "error",
                "error": f"Appointment {appointment_id} not found",
                "data": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _format_response(self, response: dict) -> str:
        """Format the response in a human-readable way"""
        if response.get('status') == 'error':
            return f"Error: {response.get('error', 'Unknown error')}"
        
        data = response.get('data', {})
        
        # Handle available slots
        if 'available_slots' in data:
            slots = data['available_slots']
            if not slots:
                return "No available appointments found for the specified time period."
            
            text = "Available appointments:\n\n"
            for slot in slots:
                datetime_str = slot.get('datetime', '').replace('T', ' at ')
                text += f"- {datetime_str} with {slot.get('provider', 'Unknown Provider')}\n"
                text += f"  Duration: {slot.get('duration', 30)} minutes\n\n"
            return text
            
        # Handle appointment details
        elif 'appointment_id' in data:
            datetime_str = data.get('datetime', '').replace('T', ' at ')
            text = f"Appointment {data.get('appointment_id')}:\n\n"
            text += f"- Date and Time: {datetime_str}\n"
            text += f"- Provider: {data.get('provider', 'Unknown Provider')}\n"
            text += f"- Type: {data.get('type', 'Unknown Type')}\n"
            text += f"- Duration: {data.get('duration', 30)} minutes\n"
            if 'status' in data:
                text += f"- Status: {data.get('status')}\n"
            return text
            
        return str(response) 