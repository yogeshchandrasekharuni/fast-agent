import json

from mcp_agent.workflows.router.agent_router import (
    AgentInfo,
    AgentRouter,
    AgentRoutingResult,
    RoutingResponse,
)


def test_agent_selection_model_to_json():
    """Test converting an AgentSelection model to JSON"""
    # Create a test selection
    selection = AgentRouter.create_test_selection(
        agent_name="code_assistant",
        confidence="high",
        reasoning="This agent has skills in code review and generation",
    )

    # Convert to JSON
    json_str = selection.to_json()

    # Parse the JSON to verify it's valid
    parsed = json.loads(json_str)

    # Check the values
    assert parsed["agent_name"] == "code_assistant"
    assert parsed["confidence"] == "high"
    assert "code review" in parsed["reasoning"]


def test_routing_response_model_to_json():
    """Test converting a RoutingResponse model to JSON"""
    # Create test selections
    selection1 = AgentRouter.create_test_selection(agent_name="code_assistant", confidence="high")
    selection2 = AgentRouter.create_test_selection(
        agent_name="data_analyst",
        confidence="medium",
        reasoning="This agent can analyze data but may not be the best fit",
    )

    # Create response with multiple selections
    response = RoutingResponse(selections=[selection1, selection2])

    # Convert to JSON
    json_str = response.to_json()

    # Parse the JSON to verify it's valid
    parsed = json.loads(json_str)

    # Check the values
    assert len(parsed["selections"]) == 2
    assert parsed["selections"][0]["agent_name"] == "code_assistant"
    assert parsed["selections"][1]["agent_name"] == "data_analyst"
    assert parsed["selections"][1]["confidence"] == "medium"
    assert "may not be the best fit" in parsed["selections"][1]["reasoning"]


def test_agent_routing_result_creation():
    """Test creating an AgentRoutingResult with the simplified agent info"""
    # Create agent info
    agent_info = AgentInfo(name="code_assistant", description="An agent that helps with code tasks")

    # Create routing result
    result = AgentRoutingResult(
        agent_info=agent_info,
        agent_name="code_assistant",
        confidence="high",
        reasoning="Perfect match for code-related tasks",
        metadata={"server_names": ["code_server"]},
    )

    # Check values
    assert result.agent_name == "code_assistant"
    assert result.agent_info.description == "An agent that helps with code tasks"
    assert result.confidence == "high"
    assert "Perfect match" in result.reasoning
    assert result.metadata["server_names"] == ["code_server"]
