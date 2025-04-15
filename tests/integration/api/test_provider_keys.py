import os

import pytest

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.provider_key_manager import ProviderKeyManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_for_bad_provider_or_not_set(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent()
    async def agent_function():
        async with fast.run():
            assert fast.config

            with pytest.raises(ProviderKeyError):  # invalid provider
                ProviderKeyManager.get_api_key("foo", fast.config)

            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = ""
            openai_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = ""

            try:
                with pytest.raises(ProviderKeyError):  # not supplied
                    ProviderKeyManager.get_api_key("deepseek", fast.config)

                with pytest.raises(ProviderKeyError):  # default string in secrets file
                    ProviderKeyManager.get_api_key("openai", fast.config)
            finally:
                if deepseek_key:
                    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
                if openai_key:
                    os.environ["OPENAI_API_KEY"] = openai_key

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reads_keys_and_prioritises_config_file(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent()
    async def agent_function():
        async with fast.run():
            assert fast.config

            assert "test-key-anth" == ProviderKeyManager.get_api_key("anthropic", fast.config)

            openai_key = os.getenv("OPENAI_API_KEY")
            anth_key = os.getenv("ANTHROPIC_API_KEY")
            try:
                os.environ["OPENAI_API_KEY"] = "test-key"
                os.environ["ANTHROPIC_API_KEY"] = "override"
                assert "test-key" == ProviderKeyManager.get_api_key("openai", fast.config)
                assert "test-key-anth" == ProviderKeyManager.get_api_key(
                    "anthropic", fast.config
                ), "config file > environment variable"
            finally:
                if openai_key:
                    os.environ["OPENAI_API_KEY"] = openai_key
                if anth_key:
                    os.environ["ANTHROPIC_API_KEY"] = anth_key

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_generic_api_key(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent()
    async def agent_function():
        async with fast.run():
            assert fast.config

            assert "ollama" == ProviderKeyManager.get_api_key("generic", fast.config)

    await agent_function()
