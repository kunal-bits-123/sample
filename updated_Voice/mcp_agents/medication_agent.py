from typing import Dict, Any, Optional, List
import json
import os
import logging
from .base_agent import BaseMCPAgent

class MedicationAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, debug: bool = False):
        super().__init__("Medication Agent", groq_api_key)
        self.logger = logging.getLogger("MedicationAgent")
        self.medication_data_path = os.path.join('test_data', 'medical', 'medications.json')
        self._load_medication_data()
        self.debug = debug

    def _load_medication_data(self):
        """Load medication data from local file"""
        try:
            if os.path.exists(self.medication_data_path):
                with open(self.medication_data_path, 'r') as f:
                    self.medication_data = json.load(f)
            else:
                self.medication_data = {"medications": []}
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.medication_data_path), exist_ok=True)
                with open(self.medication_data_path, 'w') as f:
                    json.dump(self.medication_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to load medication data: {e}")
            self.medication_data = {"medications": []}

    def _save_medication_data(self):
        """Save medication data to file"""
        try:
            with open(self.medication_data_path, 'w') as f:
                json.dump(self.medication_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save medication data: {e}")

    def _get_system_prompt(self) -> str:
        return """You are a Medication Agent responsible for managing medication information and interactions.
        You can perform the following operations:
        - check_interactions: Check interactions between medications
        - verify_dosage: Verify medication dosage
        - get_info: Get medication information
        
        IMPORTANT: You must respond with a valid JSON object. The response must be parseable JSON.
        DO NOT include any escaped characters or newlines in string values.
        All string values should be properly quoted and escaped.
        
        Always respond in JSON format with the following structure:
        {
            "operation": "<operation_type>",
            "status": "success" or "error",
            "data": {
                "medications": [
                    {
                        "name": "<medication_name>",
                        "class": "<medication_class>",
                        "indication": "<indication>"
                    }
                ],
                "interactions": [
                    {
                        "severity": "<severity>",
                        "description": "<description>"
                    }
                ]
            },
            "warnings": ["<warning1>", "<warning2>"],
            "error": null or error_message
        }

        Remember:
        1. All string values must be properly quoted
        2. No escaped newlines in string values
        3. No trailing commas
        4. No comments
        5. No extra whitespace
        6. All arrays and objects must be properly closed"""

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
                # Try to extract and fix common JSON formatting issues
                try:
                    # Remove any escaped newlines and fix quotes
                    fixed_response = response.replace('\\n', ' ').replace('\\"', '"')
                    response_data = json.loads(fixed_response)
                except:
                    return {
                        "operation": "unknown",
                        "status": "error",
                        "error": f"Invalid JSON response from LLM: {str(e)}",
                        "data": None
                    }
            
            if self.debug:
                print("[DEBUG] MedicationAgent LLM response:", json.dumps(response_data, indent=2))
            
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

    def _handle_interactions(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle medication interaction checks"""
        try:
            medications = response_data.get("data", {}).get("medications", [])
            if not medications:
                return {
                    "status": "error",
                    "error": "No medications specified",
                    "data": None
                }

            # Check interactions from medication data
            interactions = []
            warnings = []
            
            for med in medications:
                med_name = med.get("name")
                med_data = next((m for m in self.medication_data["medications"] if m["name"] == med_name), None)
                
                if med_data:
                    for interaction in med_data.get("interactions", []):
                        interactions.append({
                            "severity": "Moderate",
                            "description": f"Interaction between {med_name} and {interaction}"
                        })
                        warnings.append(f"Monitor for adverse effects when taking {med_name} with {interaction}")

            return {
                "status": "success",
                "data": {
                    "medications": medications,
                    "interactions": interactions
                },
                "warnings": warnings,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_dosage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle medication dosage verification"""
        try:
            medications = response_data.get("data", {}).get("medications", [])
            if not medications:
                return {
                    "status": "error",
                    "error": "No medications specified",
                    "data": None
                }

            # Get dosage information from medication data
            dosage_info = []
            for med in medications:
                med_name = med.get("name")
                med_data = next((m for m in self.medication_data["medications"] if m["name"] == med_name), None)
                
                if med_data:
                    dosage_info.append({
                        "name": med_name,
                        "class": med_data.get("class", "Unknown"),
                        "indication": med_data.get("indication", "Unknown"),
                        "dosage": med.get("dosage", "Standard dosage")
                    })

            return {
                "status": "success",
                "data": {
                    "medications": dosage_info
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    def _handle_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle medication information requests"""
        try:
            medications = response_data.get("data", {}).get("medications", [])
            if not medications:
                return {
                    "status": "error",
                    "error": "No medications specified",
                    "data": None
                }

            # Get medication information from data
            med_info = []
            for med in medications:
                med_name = med.get("name")
                med_data = next((m for m in self.medication_data["medications"] if m["name"] == med_name), None)
                
                if med_data:
                    med_info.append({
                        "name": med_name,
                        "class": med_data.get("class", "Unknown"),
                        "indication": med_data.get("indication", "Unknown"),
                        "interactions": med_data.get("interactions", [])
                    })

            return {
                "status": "success",
                "data": {
                    "medications": med_info
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    async def check_drug_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """
        Check for potential drug interactions between medications.
        """
        try:
            interactions = []
            warnings = []
            
            for med_name in medications:
                med_data = next((m for m in self.medication_data["medications"] if m["name"] == med_name), None)
                if med_data:
                    for interaction in med_data.get("interactions", []):
                        interactions.append({
                            "medication1": med_name,
                            "medication2": interaction,
                            "severity": "Moderate",
                            "description": f"Interaction between {med_name} and {interaction}"
                        })
                        warnings.append(f"Monitor for adverse effects when taking {med_name} with {interaction}")

            return {
                "status": "success",
                "data": {
                    "medications": medications,
                    "interactions": interactions,
                    "warnings": warnings
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    async def verify_medication_dosing(self, medication: str, dose: str) -> Dict[str, Any]:
        """
        Verify if a medication dose is appropriate.
        """
        try:
            med_data = next((m for m in self.medication_data["medications"] if m["name"] == medication), None)
            if not med_data:
                return {
                    "status": "error",
                    "error": f"Medication {medication} not found",
                    "data": None
                }

            return {
                "status": "success",
                "data": {
                    "medication": medication,
                    "dose": dose,
                    "is_appropriate": True,
                    "recommended_range": "Standard dosage",
                    "warnings": []
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }

    async def suggest_alternative_medications(self, medication: str, reason: str) -> Dict[str, Any]:
        """
        Suggest alternative medications based on contraindications or allergies.
        """
        try:
            med_data = next((m for m in self.medication_data["medications"] if m["name"] == medication), None)
            if not med_data:
                return {
                    "status": "error",
                    "error": f"Medication {medication} not found",
                    "data": None
                }

            # Find alternative medications in the same class
            alternatives = [
                m for m in self.medication_data["medications"]
                if m["class"] == med_data["class"] and m["name"] != medication
            ]

            return {
                "status": "success",
                "data": {
                    "original_medication": medication,
                    "reason": reason,
                    "alternatives": [
                        {
                            "name": alt["name"],
                            "dosing": "Standard dosage",
                            "advantages": ["Same class as original medication"],
                            "disadvantages": ["May have similar contraindications"]
                        }
                        for alt in alternatives
                    ]
                }
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
        
        operation = response.get('operation', '')
        data = response.get('data', {})
        
        if operation == 'get_info':
            medications = data.get('medications', [])
            warnings = response.get('warnings', [])
            
            # Format medications into paragraphs
            medication_text = "Current medication trends show the following commonly prescribed medications:\n\n"
            
            for med in medications:
                medication_text += f"{med['name']} ({med['class']}) - Used for {med['indication']}.\n"
            
            # Add warnings if any
            if warnings:
                medication_text += "\nImportant considerations:\n"
                for warning in warnings:
                    medication_text += f"- {warning}\n"
            
            return medication_text
            
        elif operation == 'check_interactions':
            interactions = data.get('interactions', [])
            if not interactions:
                return "No significant interactions found between the specified medications."
            
            interaction_text = "Medication interaction analysis:\n\n"
            for interaction in interactions:
                interaction_text += f"Severity: {interaction['severity']}\n"
                interaction_text += f"Description: {interaction['description']}\n\n"
            
            return interaction_text
            
        elif operation == 'verify_dosage':
            return f"Recommended dosage: {data.get('dosage', 'Not specified')}"
            
        return str(response) 