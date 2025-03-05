# src/mcp_agent/workflows/llm/enhanced_passthrough.py


import datetime
from typing import List, Optional, Union
from mcp_agent.core.simulator_registry import SimulatorRegistry
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    RequestParams,
)


class EnhancedPassthroughLLM(AugmentedLLM):
    """Enhanced passthrough LLM for testing parameter handling and workflows"""

    def __init__(self, name: str = "Simulator", context=None, **kwargs):
        super().__init__(name=name, context=context, **kwargs)
        self.simulation_mode = kwargs.get("simulation_mode", "passthrough")
        self.request_log = []
        self.last_request_params = None

        # Register this instance with the registry
        SimulatorRegistry.register(self.name, self)

    async def generate_str(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> str:
        """Capture parameters and log the request"""
        # Store for assertion testing
        self.last_request_params = request_params

        # Log the request
        self.request_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": str(message),
                "request_params": request_params.model_dump()
                if request_params
                else None,
            }
        )

        # Display for debugging
        self.show_user_message(str(message), model="simulator", chat_turn=0)

        # Simulate response
        result = f"[SIMULATOR] Response to: {message}"
        await self.show_assistant_message(result, title="SIMULATOR")

        return result

    # Other generate methods with similar parameter capture

    def get_parameter_usage_report(self):
        """Generate report of parameter usage"""
        param_usage = {}

        for req in self.request_log:
            params = req.get("request_params", {})
            if params:
                for key, value in params.items():
                    if key not in param_usage:
                        param_usage[key] = {"count": 0, "values": set()}
                    param_usage[key]["count"] += 1
                    param_usage[key]["values"].add(str(value))

        return {"total_requests": len(self.request_log), "parameter_usage": param_usage}
