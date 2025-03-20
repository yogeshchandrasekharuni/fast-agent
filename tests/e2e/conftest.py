import pytest
import sys
import importlib

@pytest.fixture(scope="function", autouse=True)
def ensure_event_bus_shutdown():
    """Explicitly signal event bus shutdown after each test"""
    # Run the test
    yield
    
    # After the test, signal shutdown to event bus
    try:
        transport_module = importlib.import_module('mcp_agent.logging.transport')
        AsyncEventBus = getattr(transport_module, 'AsyncEventBus', None)
        
        if AsyncEventBus and AsyncEventBus._instance:
            # Signal shutdown
            instance = AsyncEventBus._instance
            instance._running = False
            if hasattr(instance, '_stop_event'):
                instance._stop_event.set()
                
            # Clear the singleton to ensure fresh start for next test
            AsyncEventBus._instance = None
    except Exception as e:
        print(f"Error cleaning up event bus: {e}")
        pass