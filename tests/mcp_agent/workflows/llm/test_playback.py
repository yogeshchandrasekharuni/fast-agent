import pytest
from unittest.mock import MagicMock, patch
from mcp import GetPromptResult
from mcp.types import PromptMessage, TextContent, Role
from mcp_agent.workflows.llm.augmented_llm_playback import PlaybackLLM
from mcp_agent.workflows.llm.model_factory import ModelFactory


@pytest.mark.asyncio
async def test_playback_llm_init():
    """Test that PlaybackLLM is properly initialized"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Verify instance properties
    assert llm.name == "TestPlayback"
    assert len(llm._messages) == 0
    assert llm._current_index == 0


@pytest.mark.asyncio
async def test_playback_llm_messages_exhausted():
    """Test that PlaybackLLM returns an exhausted message when no prompts applied"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Patch the show methods to avoid display issues in tests
    with patch.object(llm, 'show_user_message'), patch.object(llm, 'show_assistant_message'):
        # Generate a string response
        response = await llm.generate_str("Hello")
        
        # Verify the response indicates messages are exhausted
        assert response == "MESSAGES EXHAUSTED (list size 0)"


@pytest.mark.asyncio
async def test_playback_llm_apply_prompt_template():
    """Test that PlaybackLLM correctly applies prompt templates"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Create sample prompt messages using Pydantic models
    prompt_messages = [
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 1")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 2")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 3")
        )
    ]
    
    # Create a GetPromptResult
    prompt_result = GetPromptResult(
        description="Test prompt",
        messages=prompt_messages
    )
    
    # Patch the show_prompt_loaded method to avoid display issues
    with patch.object(llm, 'show_prompt_loaded'):
        # Apply the prompt template
        result = await llm.apply_prompt_template(prompt_result, "test_prompt")
        
        # Verify the result and internal state
        assert "Added 3 messages" in result
        assert len(llm._messages) == 3
        assert llm._current_index == 0


@pytest.mark.asyncio
async def test_playback_llm_sequential_messages():
    """Test that PlaybackLLM returns messages in sequence"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Create sample prompt messages using Pydantic models
    prompt_messages = [
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 1")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 2")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Message 3")
        )
    ]
    
    # Create a GetPromptResult
    prompt_result = GetPromptResult(
        description="Test prompt",
        messages=prompt_messages
    )
    
    # Patch the show methods to avoid display issues
    with patch.object(llm, 'show_prompt_loaded'), patch.object(llm, 'show_user_message'), patch.object(llm, 'show_assistant_message'):
        # Apply the prompt template
        await llm.apply_prompt_template(prompt_result, "test_prompt")
        
        # Get messages sequentially
        response1 = await llm.generate_str("Input 1")
        assert response1 == "Message 1"
        
        response2 = await llm.generate_str("Input 2")
        assert response2 == "Message 2"
        
        response3 = await llm.generate_str("Input 3")
        assert response3 == "Message 3"
        
        # Check that we get exhausted message after all messages have been played
        response4 = await llm.generate_str("Input 4")
        assert response4 == "MESSAGES EXHAUSTED (list size 3)"


@pytest.mark.asyncio
async def test_playback_llm_append_messages():
    """Test that PlaybackLLM correctly appends messages"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Create first batch of prompt messages
    prompt_messages1 = [
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Batch 1 - Message 1")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Batch 1 - Message 2")
        )
    ]
    
    # Create second batch of prompt messages
    prompt_messages2 = [
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Batch 2 - Message 1")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="Batch 2 - Message 2")
        )
    ]
    
    # Create GetPromptResults
    prompt_result1 = GetPromptResult(
        description="Test prompt 1",
        messages=prompt_messages1
    )
    
    prompt_result2 = GetPromptResult(
        description="Test prompt 2",
        messages=prompt_messages2
    )
    
    # Patch the show methods to avoid display issues
    with patch.object(llm, 'show_prompt_loaded'), patch.object(llm, 'show_user_message'), patch.object(llm, 'show_assistant_message'):
        # Apply the first prompt template
        await llm.apply_prompt_template(prompt_result1, "test_prompt_1")
        
        # Get first batch of messages
        response1 = await llm.generate_str("Input 1")
        assert response1 == "Batch 1 - Message 1"
        
        response2 = await llm.generate_str("Input 2")
        assert response2 == "Batch 1 - Message 2"
        
        # Apply the second prompt template
        await llm.apply_prompt_template(prompt_result2, "test_prompt_2")
        
        # Continue getting messages (should get second batch)
        response3 = await llm.generate_str("Input 3")
        assert response3 == "Batch 2 - Message 1"
        
        response4 = await llm.generate_str("Input 4")
        assert response4 == "Batch 2 - Message 2"
        
        # Check that we get exhausted message after all messages have been played
        response5 = await llm.generate_str("Input 5")
        assert response5 == "MESSAGES EXHAUSTED (list size 4)"


@pytest.mark.asyncio
async def test_playback_llm_skips_assistant_messages():
    """Test that PlaybackLLM skips assistant messages and only returns user messages"""
    # Create a mock context
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.executor = MagicMock()
    mock_context.model_selector = MagicMock()
    
    # Create a PlaybackLLM instance
    llm = PlaybackLLM(name="TestPlayback", context=mock_context)
    
    # Create mixed prompt messages with both user and assistant roles
    prompt_messages = [
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="User Message 1")
        ),
        PromptMessage(
            role="assistant", 
            content=TextContent(type="text", text="Assistant Reply 1")
        ),
        PromptMessage(
            role="user", 
            content=TextContent(type="text", text="User Message 2")
        ),
        PromptMessage(
            role="assistant", 
            content=TextContent(type="text", text="Assistant Reply 2")
        )
    ]
    
    # Create a GetPromptResult
    prompt_result = GetPromptResult(
        description="Mixed role test prompt",
        messages=prompt_messages
    )
    
    # Patch the show methods to avoid display issues in tests
    with patch.object(llm, 'show_prompt_loaded'), patch.object(llm, 'show_user_message'), patch.object(llm, 'show_assistant_message'):
        # Apply the prompt template
        await llm.apply_prompt_template(prompt_result, "mixed_prompt")
        
        # Get messages sequentially - should only get user messages
        response1 = await llm.generate_str("Input 1")
        assert response1 == "User Message 1"
        
        response2 = await llm.generate_str("Input 2")
        assert response2 == "User Message 2"
        
        # Check that we get exhausted message after all user messages have been played
        response3 = await llm.generate_str("Input 3")
        assert response3 == "MESSAGES EXHAUSTED (list size 4)"


@pytest.mark.asyncio
async def test_model_factory_creates_playback():
    """Test that ModelFactory correctly creates a PlaybackLLM instance"""
    # Create a factory for the playback model
    factory = ModelFactory.create_factory("playback")
    
    # Verify the factory is callable
    assert callable(factory)
    
    # Create a mock agent
    mock_agent = MagicMock()
    
    # Create an instance using the factory
    instance = factory(mock_agent)
    
    # Verify the instance is a PlaybackLLM
    assert isinstance(instance, PlaybackLLM)