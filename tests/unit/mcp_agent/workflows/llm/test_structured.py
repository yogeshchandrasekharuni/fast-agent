from typing import List, Literal

import pytest
from pydantic import BaseModel

from mcp_agent.core.prompt import Prompt
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
    llm = PassthroughLLM(name="structured")
    result = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    # Verify conversion worked correctly
    assert isinstance(result, StructuredResponse)
    assert len(result.categories) == 2
    assert result.categories[0].category == "tech_support"
    assert result.categories[0].confidence == "high"
    assert result.categories[1].category == "documentation"
    assert result.categories[1].confidence == "medium"
    assert result.categories[1].reasoning is None


@pytest.mark.asyncio
async def test_strucutred_with_bad_json():
    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reaso: null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="structured")
    result = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    assert None is result
    # # Verify conversion worked correctly
    # assert isinstance(result, StructuredResponse)
    # assert len(result.categories) == 2
    # assert result.categories[0].category == "tech_support"
    # assert result.categories[0].confidence == "high"
    # assert result.categories[1].category == "documentation"
    # assert result.categories[1].confidence == "medium"
    # assert result.categories[1].reasoning is None
