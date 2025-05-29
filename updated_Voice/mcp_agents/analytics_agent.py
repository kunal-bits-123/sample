from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
from .base_agent import BaseMCPAgent
import os

class AnalyticsAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, debug: bool = False):
        super().__init__("Analytics Agent", groq_api_key)
        self.usage_metrics = {
            "total_encounters": 0,
            "total_orders": 0,
            "total_appointments": 0,
            "total_medication_checks": 0,
            "total_clinical_decisions": 0,
            "encounter_duration": [],
            "user_satisfaction": [],
            "error_rates": {
                "stt_errors": 0,
                "nlp_errors": 0,
                "order_errors": 0,
                "scheduling_errors": 0
            }
        }
        self.compliance_data = {
            "hipaa_compliance": [],
            "clinical_guidelines": [],
            "medication_safety": [],
            "documentation_completeness": []
        }
        self.debug = debug
        self.analytics_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test_data', 'analytics', 'metrics.json')

    def _get_system_prompt(self) -> str:
        return """You are an AI assistant specialized in analyzing clinical system usage and compliance.
Your task is to help healthcare providers track and improve system performance and regulatory compliance.

You MUST return your response in JSON format with the following structure:
{
    "operation": "generate_metrics|check_compliance|analyze_trends",
    "status": "success|error",
    "data": {
        // Analytics data, metrics, or compliance information
    },
    "recommendations": [
        // List of improvement suggestions
    ],
    "error": null or error message if status is "error"
}

Focus on:
1. System usage metrics
2. Clinical impact analysis
3. Compliance monitoring
4. Performance optimization
5. Quality improvement
6. Risk assessment

Always provide actionable insights and evidence-based recommendations."""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process analytics-related requests and return metrics or compliance information.
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
                print("[DEBUG] AnalyticsAgent LLM response:", json.dumps(response_data, indent=2))

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

    async def generate_usage_metrics(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate usage metrics for a specific time range.
        """
        # TODO: Implement actual metrics calculation
        return {
            "status": "success",
            "data": {
                "time_range": time_range,
                "metrics": {
                    "total_encounters": self.usage_metrics["total_encounters"],
                    "average_encounter_duration": "15 minutes",
                    "user_satisfaction_score": 4.5,
                    "error_rate": 0.02,
                    "system_uptime": "99.9%"
                },
                "breakdown": {
                    "by_feature": {
                        "voice_recognition": 1000,
                        "clinical_decision_support": 500,
                        "medication_management": 300,
                        "appointment_scheduling": 200
                    },
                    "by_time_of_day": {
                        "morning": 40,
                        "afternoon": 45,
                        "evening": 15
                    }
                }
            },
            "recommendations": [
                "Increase system usage during evening hours",
                "Focus on reducing medication management errors",
                "Implement additional training for voice recognition features"
            ]
        }

    async def check_compliance_status(self, compliance_area: str) -> Dict[str, Any]:
        """
        Check compliance status for a specific area.
        """
        # TODO: Implement actual compliance checking
        return {
            "status": "success",
            "data": {
                "compliance_area": compliance_area,
                "status": "compliant",
                "last_audit": "2024-02-15",
                "findings": [
                    {
                        "area": "HIPAA Compliance",
                        "status": "compliant",
                        "details": "All required safeguards in place"
                    },
                    {
                        "area": "Clinical Guidelines",
                        "status": "compliant",
                        "details": "Following latest evidence-based guidelines"
                    }
                ],
                "risk_assessment": {
                    "overall_risk": "low",
                    "areas_of_concern": []
                }
            },
            "recommendations": [
                "Schedule quarterly compliance reviews",
                "Update documentation procedures",
                "Implement additional audit logging"
            ]
        }

    async def analyze_usage_trends(self, metric: str) -> Dict[str, Any]:
        """
        Analyze trends for a specific metric.
        """
        # TODO: Implement actual trend analysis
        return {
            "status": "success",
            "data": {
                "metric": metric,
                "trend": "increasing",
                "change_rate": "+15%",
                "historical_data": [
                    {"date": "2024-01", "value": 100},
                    {"date": "2024-02", "value": 115}
                ],
                "forecast": [
                    {"date": "2024-03", "predicted_value": 130},
                    {"date": "2024-04", "predicted_value": 145}
                ]
            },
            "recommendations": [
                "Scale system resources to handle increased load",
                "Review peak usage patterns",
                "Optimize performance for high-traffic periods"
            ]
        }

    def record_metric(self, metric_type: str, value: Any):
        """
        Record a new metric value.
        """
        if metric_type in self.usage_metrics:
            if isinstance(self.usage_metrics[metric_type], list):
                self.usage_metrics[metric_type].append({
                    "timestamp": datetime.now().isoformat(),
                    "value": value
                })
            else:
                self.usage_metrics[metric_type] = value

    def record_compliance_check(self, area: str, status: str, details: Dict[str, Any]):
        """
        Record a compliance check result.
        """
        if area in self.compliance_data:
            self.compliance_data[area].append({
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "details": details
            }) 