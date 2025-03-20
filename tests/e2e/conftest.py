import pytest
import importlib

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