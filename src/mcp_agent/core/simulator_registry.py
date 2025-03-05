from typing import Optional, Any


class SimulatorRegistry:
    """Registry to access simulator instances for testing assertions"""

    _instances = {}

    @classmethod
    def register(cls, name: str, simulator: "Any"):
        """Register a simulator instance"""
        cls._instances[name] = simulator

    @classmethod
    def get(cls, name: str) -> Optional["Any"]:
        """Get a simulator by name"""
        return cls._instances.get(name)

    @classmethod
    def clear(cls):
        """Clear registry (useful between tests)"""
        cls._instances.clear()
