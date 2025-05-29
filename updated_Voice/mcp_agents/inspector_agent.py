from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from .base_agent import BaseMCPAgent

class InspectorAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, debug: bool = False):
        super().__init__("Inspector Agent", groq_api_key)
        self.debug = debug
        self.agent_states = {}
        self.protocol_violations = []
        self.cross_agent_context = {}

    def _get_system_prompt(self) -> str:
        return """You are an Inspector Agent responsible for monitoring and validating agent interactions.
        You can perform the following operations:
        - validate_response: Validate agent responses against protocol
        - monitor_state: Monitor agent state changes
        - resolve_conflicts: Resolve conflicts between agents
        - track_context: Track cross-agent context
        
        Always respond in JSON format with the following structure:
        {
            "operation": "<operation_type>",
            "status": "success" or "error",
            "data": {
                "validation_result": {
                    "is_valid": true/false,
                    "violations": [],
                    "suggestions": []
                },
                "state_changes": {
                    "agent": "<agent_name>",
                    "previous_state": {},
                    "new_state": {}
                },
                "conflict_resolution": {
                    "conflict_type": "<type>",
                    "resolution": "<resolution>",
                    "affected_agents": []
                },
                "context_update": {
                    "key": "<context_key>",
                    "value": "<context_value>",
                    "source_agent": "<agent_name>"
                }
            },
            "error": null or error_message
        }"""

    async def validate_response(self, agent_name: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an agent's response against the protocol"""
        try:
            # Check required fields
            required_fields = ["operation", "status", "data"]
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                return {
                    "status": "error",
                    "data": {
                        "validation_result": {
                            "is_valid": False,
                            "violations": [f"Missing required field: {field}" for field in missing_fields],
                            "suggestions": ["Ensure all required fields are present in the response"]
                        }
                    },
                    "error": None
                }

            # Validate operation type
            valid_operations = {
                "EHRAgent": ["retrieve", "update", "create"],
                "MedicationAgent": ["check_interactions", "verify_dosage", "get_info"],
                "OrderAgent": ["create_order", "verify_order", "cancel_order"],
                "ClinicalDecisionAgent": ["analyze_case", "check_guidelines", "assess_risk"],
                "SchedulingAgent": ["check_availability", "schedule_appointment", "reschedule_appointment", "cancel_appointment"],
                "AnalyticsAgent": ["generate_metrics", "check_compliance", "analyze_trends"]
            }

            if agent_name in valid_operations:
                if response["operation"] not in valid_operations[agent_name]:
                    return {
                        "status": "error",
                        "data": {
                            "validation_result": {
                                "is_valid": False,
                                "violations": [f"Invalid operation: {response['operation']}"],
                                "suggestions": [f"Valid operations for {agent_name}: {', '.join(valid_operations[agent_name])}"]
                            }
                        },
                        "error": None
                    }

            # Validate data structure
            if not isinstance(response["data"], dict):
                return {
                    "status": "error",
                    "data": {
                        "validation_result": {
                            "is_valid": False,
                            "violations": ["Data field must be a dictionary"],
                            "suggestions": ["Ensure data field is a valid JSON object"]
                        }
                    },
                    "error": None
                }

            return {
                "status": "success",
                "data": {
                    "validation_result": {
                        "is_valid": True,
                        "violations": [],
                        "suggestions": []
                    }
                },
                "error": None
            }

        except Exception as e:
            return {
                "status": "error",
                "data": {
                    "validation_result": {
                        "is_valid": False,
                        "violations": [str(e)],
                        "suggestions": ["Check response format and structure"]
                    }
                },
                "error": str(e)
            }

    async def monitor_state(self, agent_name: str, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and track agent state changes"""
        try:
            previous_state = self.agent_states.get(agent_name, {})
            self.agent_states[agent_name] = new_state

            return {
                "status": "success",
                "data": {
                    "state_changes": {
                        "agent": agent_name,
                        "previous_state": previous_state,
                        "new_state": new_state
                    }
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "error": str(e)
            }

    async def resolve_conflicts(self, agent1: str, agent2: str, conflict_type: str) -> Dict[str, Any]:
        """Resolve conflicts between agents"""
        try:
            # Get current states
            state1 = self.agent_states.get(agent1, {})
            state2 = self.agent_states.get(agent2, {})

            # Record conflict
            self.protocol_violations.append({
                "timestamp": datetime.now().isoformat(),
                "agents": [agent1, agent2],
                "type": conflict_type,
                "states": [state1, state2]
            })

            # Implement conflict resolution logic here
            resolution = f"Conflict between {agent1} and {agent2} resolved"

            return {
                "status": "success",
                "data": {
                    "conflict_resolution": {
                        "conflict_type": conflict_type,
                        "resolution": resolution,
                        "affected_agents": [agent1, agent2]
                    }
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "error": str(e)
            }

    async def track_context(self, key: str, value: Any, source_agent: str) -> Dict[str, Any]:
        """Track cross-agent context"""
        try:
            self.cross_agent_context[key] = {
                "value": value,
                "source_agent": source_agent,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "status": "success",
                "data": {
                    "context_update": {
                        "key": key,
                        "value": value,
                        "source_agent": source_agent
                    }
                },
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "error": str(e)
            }

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process inspection requests"""
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
                return {
                    "operation": "unknown",
                    "status": "error",
                    "error": f"Invalid JSON response from LLM: {str(e)}",
                    "data": None
                }

            if self.debug:
                print("[DEBUG] InspectorAgent LLM response:", json.dumps(response_data, indent=2))

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

    def _format_response(self, response: dict) -> str:
        """Format the response in a human-readable way"""
        if response.get('status') == 'error':
            return f"Error: {response.get('error', 'Unknown error')}"
        
        data = response.get('data', {})
        
        if 'validation_result' in data:
            result = data['validation_result']
            if result['is_valid']:
                return "✅ Response validation successful"
            else:
                text = "❌ Response validation failed:\n"
                for violation in result['violations']:
                    text += f"- {violation}\n"
                if result['suggestions']:
                    text += "\nSuggestions:\n"
                    for suggestion in result['suggestions']:
                        text += f"- {suggestion}\n"
                return text
                
        elif 'state_changes' in data:
            changes = data['state_changes']
            return f"State change for {changes['agent']}:\n" + \
                   f"Previous: {json.dumps(changes['previous_state'], indent=2)}\n" + \
                   f"New: {json.dumps(changes['new_state'], indent=2)}"
                   
        elif 'conflict_resolution' in data:
            resolution = data['conflict_resolution']
            return f"Conflict resolved between {', '.join(resolution['affected_agents'])}:\n" + \
                   f"Type: {resolution['conflict_type']}\n" + \
                   f"Resolution: {resolution['resolution']}"
                   
        elif 'context_update' in data:
            update = data['context_update']
            return f"Context updated by {update['source_agent']}:\n" + \
                   f"Key: {update['key']}\n" + \
                   f"Value: {update['value']}"
                   
        return str(response) 