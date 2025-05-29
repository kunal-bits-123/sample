from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from .base_agent import BaseMCPAgent
import os

class OrderAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, debug: bool = False):
        super().__init__("Order Agent", groq_api_key)
        self.debug = debug
        self.order_data_path = os.path.join('/Users/kjain/groq_mcp/test_data', 'medical', 'orders.json')
        self._load_order_data()

    def _load_order_data(self):
        """Load order data from file"""
        try:
            if os.path.exists(self.order_data_path):
                with open(self.order_data_path, 'r') as f:
                    self.order_data = json.load(f)
            else:
                self.order_data = {"orders": []}
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.order_data_path), exist_ok=True)
                with open(self.order_data_path, 'w') as f:
                    json.dump(self.order_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to load order data: {e}")
            self.order_data = {"orders": []}

    def _save_order_data(self):
        """Save order data to file"""
        try:
            with open(self.order_data_path, 'w') as f:
                json.dump(self.order_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save order data: {e}")

    def _get_system_prompt(self) -> str:
        return """You are an AI assistant specialized in processing clinical orders and prescriptions.
Your task is to help healthcare providers create and manage orders for tests, medications, and procedures.

You MUST return your response in JSON format with the following structure:
{
    "operation": "create_order|verify_order|cancel_order",
    "order_type": "test|medication|procedure",
    "status": "success|error",
    "data": {
        // Order details and confirmation
    },
    "warnings": [
        // List of any warnings or special instructions
    ],
    "error": null or error message if status is "error"
}

Focus on:
1. Order accuracy and completeness
2. Clinical appropriateness
3. Insurance coverage
4. Patient safety
5. Regulatory compliance
6. Order tracking and status

Always verify orders for completeness and appropriateness before processing."""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process order-related requests and return order information or confirmation.
        """
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
                print("[DEBUG] OrderAgent LLM response:", json.dumps(response_data, indent=2))

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

    async def create_order(self, order_type: str, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new order (test, medication, or procedure).
        """
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order = {
            "order_id": order_id,
            "order_type": order_type,
            "details": order_details,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        # Add to orders list and save
        self.order_data["orders"].append(order)
        self._save_order_data()

        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "order_type": order_type,
                "details": order_details,
                "status": "pending",
                "message": f"Order {order_id} has been created and is pending verification"
            }
        }

    async def verify_order(self, order_id: str) -> Dict[str, Any]:
        """
        Verify an existing order.
        """
        # Find the order in orders list
        order = next((o for o in self.order_data["orders"] if o["order_id"] == order_id), None)
        
        if not order:
            return {
                "status": "error",
                "error": f"Order {order_id} not found",
                "data": None
            }

        # Update order status
        order["status"] = "verified"
        self._save_order_data()
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "status": "verified",
                "message": f"Order {order_id} has been verified and is ready for processing"
            }
        }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        """
        # Find the order in orders list
        order = next((o for o in self.order_data["orders"] if o["order_id"] == order_id), None)
        
        if not order:
            return {
                "status": "error",
                "error": f"Order {order_id} not found",
                "data": None
            }

        # Update order status
        order["status"] = "cancelled"
        self._save_order_data()
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "status": "cancelled",
                "message": f"Order {order_id} has been cancelled"
            }
        }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the current status of an order.
        """
        order = next((o for o in self.order_data["orders"] if o["order_id"] == order_id), None)
        
        if not order:
            return {
                "status": "error",
                "error": f"Order {order_id} not found",
                "data": None
            }

        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "order_type": order["order_type"],
                "details": order["details"],
                "status": order["status"],
                "created_at": order["created_at"]
            }
        } 