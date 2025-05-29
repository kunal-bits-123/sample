import asyncio
import os
import json
from typing import Dict, Any
from updated_Voice.mcp_agents.ehr_agent import EHRAgent
from updated_Voice.mcp_agents.medication_agent import MedicationAgent
from updated_Voice.mcp_agents.order_agent import OrderAgent
from updated_Voice.mcp_agents.clinical_decision_agent import ClinicalDecisionAgent
from updated_Voice.mcp_agents.scheduling_agent import SchedulingAgent
from updated_Voice.mcp_agents.analytics_agent import AnalyticsAgent
import pytest
import pytest_asyncio

# Fixtures for each agent
@pytest_asyncio.fixture
async def ehr_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    db_config = {
        'user': os.getenv('POSTGRES_USER', ''),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
        'database': os.getenv('POSTGRES_DB', ''),
        'host': os.getenv('POSTGRES_HOST', ''),
        'port': os.getenv('POSTGRES_PORT', '5432'),
    }
    return EHRAgent(groq_api_key, db_config)

@pytest_asyncio.fixture
async def medication_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return MedicationAgent(groq_api_key)

@pytest_asyncio.fixture
async def order_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return OrderAgent(groq_api_key)

@pytest_asyncio.fixture
async def clinical_decision_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    if not pinecone_api_key:
        pytest.skip("PINECONE_API_KEY environment variable not set")
    return ClinicalDecisionAgent(groq_api_key)

@pytest_asyncio.fixture
async def scheduling_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return SchedulingAgent(groq_api_key)

@pytest_asyncio.fixture
async def analytics_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return AnalyticsAgent(groq_api_key)

@pytest.mark.asyncio
async def test_ehr_agent(ehr_agent: EHRAgent) -> None:
    """Test EHR Agent functionality"""
    print("\n=== Testing EHR Agent ===")
    
    # Test retrieve operation
    print("\nTesting retrieve operation...")
    response = await ehr_agent.process_message("Show me John Smith's medical history")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test update operation
    print("\nTesting update operation...")
    response = await ehr_agent.process_message("Update John Smith's medical history with new diagnosis of Type 2 Diabetes")
    print(f"Response: {json.dumps(response, indent=2)}")

@pytest.mark.asyncio
async def test_medication_agent(medication_agent: MedicationAgent) -> None:
    """Test Medication Agent functionality"""
    print("\n=== Testing Medication Agent ===")
    
    # Test interaction check
    print("\nTesting medication interaction check...")
    response = await medication_agent.process_message("Check interactions between Metformin and Lisinopril")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test medication info
    print("\nTesting medication information...")
    response = await medication_agent.process_message("What is the recommended dosage for Metformin?")
    print(f"Response: {json.dumps(response, indent=2)}")

@pytest.mark.asyncio
async def test_order_agent(order_agent: OrderAgent) -> None:
    """Test Order Agent functionality"""
    print("\n=== Testing Order Agent ===")
    
    # Test create order
    print("\nTesting create order...")
    response = await order_agent.process_message("Order a complete blood count test for John Smith")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test list orders
    print("\nTesting list orders...")
    response = await order_agent.process_message("List all pending lab orders for patient P001")
    print(f"Response: {json.dumps(response, indent=2)}")

@pytest.mark.asyncio
async def test_clinical_decision_agent(clinical_decision_agent: ClinicalDecisionAgent) -> None:
    """Test Clinical Decision Agent functionality"""
    print("\n=== Testing Clinical Decision Agent ===")
    
    # Test guidelines
    print("\nTesting clinical guidelines...")
    response = await clinical_decision_agent.process_message("Show me the latest clinical guidelines for diabetes")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test recommendations
    print("\nTesting clinical recommendations...")
    response = await clinical_decision_agent.process_message("What are the recommended lifestyle changes for hypertension?")
    print(f"Response: {json.dumps(response, indent=2)}")

@pytest.mark.asyncio
async def test_scheduling_agent(scheduling_agent: SchedulingAgent) -> None:
    """Test Scheduling Agent functionality"""
    print("\n=== Testing Scheduling Agent ===")
    
    # Test availability check
    print("\nTesting availability check...")
    response = await scheduling_agent.process_message("Find available appointments for next week")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test scheduling
    print("\nTesting appointment scheduling...")
    response = await scheduling_agent.process_message("Schedule a follow-up appointment for John Smith")
    print(f"Response: {json.dumps(response, indent=2)}")

@pytest.mark.asyncio
async def test_analytics_agent(analytics_agent: AnalyticsAgent) -> None:
    """Test Analytics Agent functionality"""
    print("\n=== Testing Analytics Agent ===")
    
    # Test report generation
    print("\nTesting report generation...")
    response = await analytics_agent.process_message("Generate a report on patient outcomes")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test trend analysis
    print("\nTesting trend analysis...")
    response = await analytics_agent.process_message("What are the current medication trends?")
    print(f"Response: {json.dumps(response, indent=2)}")

async def run_all_tests():
    """Run all agent tests"""
    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        return
    
    print("🔑 API Key Status: Present")
    
    # Initialize agents
    print("\nInitializing agents...")
    ehr_agent = EHRAgent(groq_api_key)
    medication_agent = MedicationAgent(groq_api_key)
    order_agent = OrderAgent(groq_api_key)
    clinical_decision_agent = ClinicalDecisionAgent(groq_api_key)
    scheduling_agent = SchedulingAgent(groq_api_key)
    analytics_agent = AnalyticsAgent(groq_api_key)
    
    # Run tests
    await test_ehr_agent(ehr_agent)
    await test_medication_agent(medication_agent)
    await test_order_agent(order_agent)
    await test_clinical_decision_agent(clinical_decision_agent)
    await test_scheduling_agent(scheduling_agent)
    await test_analytics_agent(analytics_agent)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 