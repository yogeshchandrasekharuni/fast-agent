"""Debug configuration and utilities for MCP Agent development."""

import os
from rich import print as rprint
from functools import wraps
from typing import Any, Callable

# Check environment variable for debug mode
DEVELOPER_DEBUG = os.environ.get("MCP_DEVELOPER_DEBUG", "0").lower() in ("1", "true", "yes")

def dev_print(color: str, *args: Any, **kwargs: Any) -> None:
    """
    Print debug information if DEVELOPER_DEBUG is enabled.
    
    Args:
        color: Color to use for the message (e.g., "red", "blue", "magenta")
        *args: Arguments to pass to rich.print
        **kwargs: Keyword arguments to pass to rich.print
    """
    if DEVELOPER_DEBUG:
        # Wrap first argument in color if it's a string
        if args and isinstance(args[0], str):
            args = (f"[{color}]{args[0]}[/{color}]",) + args[1:]
        rprint(*args, **kwargs)

def dev_debug(color: str = "yellow") -> Callable:
    """
    Decorator to print entry/exit debugging information for a function if DEVELOPER_DEBUG is enabled.
    
    Args:
        color: Color to use for the debug messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not DEVELOPER_DEBUG:
                return func(*args, **kwargs)
                
            # Get class name if it's a method
            class_name = args[0].__class__.__name__ if args else ""
            prefix = f"{class_name}." if class_name else ""
            
            dev_print(color, f"\n{prefix}{func.__name__} called with:")
            if len(args) > 1:  # Skip self for methods
                dev_print(color, "args:", args[1:])
            if kwargs:
                dev_print(color, "kwargs:", kwargs)
            
            result = func(*args, **kwargs)
            
            dev_print(color, f"{prefix}{func.__name__} returned:", result)
            return result
            
        return wrapper
    return decorator