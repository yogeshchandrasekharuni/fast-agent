from typing import List, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM


# Example model similar to what's used in the Router workflow
class StructuredResponseCategory(BaseModel):
    category: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str | None


class StructuredResponse(BaseModel):
    categories: List[StructuredResponseCategory]


@pytest.mark.asyncio
async def test_direct_pydantic_conversion():
    # Create a minimal mock context
    mock_context = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.config = MagicMock()

    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
                "confidence": "high",
                "reasoning": "Query relates to system troubleshooting"
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reasoning": null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="TestPassthrough", context=mock_context)
    result = await llm.generate_structured(json_str, StructuredResponse)

    # Verify conversion worked correctly
    assert isinstance(result, StructuredResponse)
    assert len(result.categories) == 2
    assert result.categories[0].category == "tech_support"
    assert result.categories[0].confidence == "high"
    assert result.categories[1].category == "documentation"
    assert result.categories[1].confidence == "medium"
    assert result.categories[1].reasoning is None
