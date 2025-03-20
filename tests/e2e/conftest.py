import pytest
import importlib
import os
from pathlib import Path
from mcp_agent.core.fastagent import FastAgent

# Keep the auto-cleanup fixture
@pytest.fixture(scope="function", autouse=True)
def cleanup_event_bus():
    """Reset the AsyncEventBus between tests using its reset method"""
    # Run the test
    yield
    
    # Reset the AsyncEventBus after each test
    try:
        # Import the module with the AsyncEventBus
        transport_module = importlib.import_module('mcp_agent.logging.transport')
        AsyncEventBus = getattr(transport_module, 'AsyncEventBus', None)
        
        # Call the reset method if available
        if AsyncEventBus and hasattr(AsyncEventBus, 'reset'):
            AsyncEventBus.reset()
    except Exception:
        pass

# Set the project root directory for tests
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    # Go up from tests/e2e directory to find project root
    return Path(__file__).parent.parent.parent

# Fixture to set the current working directory for tests
@pytest.fixture
def set_cwd(project_root):
    """Change to the project root directory during tests"""
    # Save the original working directory
    original_cwd = os.getcwd()
    
    # Change to the project root directory
    os.chdir(project_root)
    
    # Run the test
    yield
    
    # Restore the original working directory
    os.chdir(original_cwd)

# Add a fixture for creating FastAgent instances
@pytest.fixture
def fast_agent(set_cwd):
    """Create a FastAgent instance for tests"""
    # Create the FastAgent instance with common configuration
    # Now we can use paths relative to project root because of set_cwd fixture
    agent = FastAgent(
        "Image Tests",
        config_path="tests/e2e/multimodal/fastagent.config.yaml",
        ignore_unknown_args=True,
    )
    
    # Return the agent for the test to use
    yield agent