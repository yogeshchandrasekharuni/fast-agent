import asyncio
import uuid
from typing import Callable, Dict, List, Optional, TypeVar, Union, TYPE_CHECKING

from mcp.server.fastmcp.tools import Tool as FastTool
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    EmbeddedResource,
    ReadResourceResult,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

from mcp_agent.core.exceptions import PromptExitError
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.human_input.types import (
    HumanInputCallback,
    HumanInputRequest,
    HumanInputResponse,
    HUMAN_INPUT_SIGNAL_NAME,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.context import Context
import traceback

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLM)

HUMAN_INPUT_TOOL_NAME = "__human_input__"


class Agent(MCPAggregator):
    """
    An Agent is an entity that has access to a set of MCP servers and can interact with them.
    Each agent should have a purpose defined by its instruction.
    """

    def __init__(
        self,
        config: Union[
            AgentConfig, str
        ],  # Can be AgentConfig or backward compatible str name
        instruction: Optional[Union[str, Callable[[Dict], str]]] = None,
        server_names: Optional[List[str]] = None,
        functions: Optional[List[Callable]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        # Handle backward compatibility where first arg was name
        if isinstance(config, str):
            self.config = AgentConfig(
                name=config,
                instruction=instruction or "You are a helpful agent.",
                servers=server_names or [],
            )
        else:
            self.config = config

        super().__init__(
            context=context,
            server_names=self.config.servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            **kwargs,
        )

        self.name = self.config.name
        self.instruction = self.config.instruction
        self.functions = functions or []
        self.executor = self.context.executor
        self.logger = get_logger(f"{__name__}.{self.name}")

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # Map function names to tools
        self._function_tool_map: Dict[str, FastTool] = {}

        if not self.config.human_input:
            self.human_input_callback = None
        else:
            self.human_input_callback: HumanInputCallback | None = human_input_callback
            if not human_input_callback:
                if self.context.human_input_handler:
                    self.human_input_callback = self.context.human_input_handler

    async def initialize(self):
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await (
            self.__aenter__()
        )  # This initializes the connection manager and loads the servers

        for function in self.functions:
            tool: FastTool = FastTool.from_function(function)
            self._function_tool_map[tool.name] = tool

    async def attach_llm(self, llm_factory: Callable[..., LLM]) -> LLM:
        """
        Create an LLM instance for the agent.

        Args:
            llm_factory: A callable that constructs an AugmentedLLM or its subclass.
                        The factory should accept keyword arguments matching the
                        AugmentedLLM constructor parameters.

        Returns:
            An instance of AugmentedLLM or one of its subclasses.
        """
        return llm_factory(
            agent=self, default_request_params=self._default_request_params
        )

    async def shutdown(self):
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await super().close()

    async def request_human_input(self, request: HumanInputRequest) -> str:
        """
        Request input from a human user. Pauses the workflow until input is received.

        Args:
            request: The human input request

        Returns:
            The input provided by the human

        Raises:
            TimeoutError: If the timeout is exceeded
        """
        if not self.human_input_callback:
            raise ValueError("Human input callback not set")

        # Generate a unique ID for this request to avoid signal collisions
        request_id = f"{HUMAN_INPUT_SIGNAL_NAME}_{self.name}_{uuid.uuid4()}"
        request.request_id = request_id
        # Use metadata as a dictionary to pass agent name
        request.metadata = {"agent_name": self.name}
        self.logger.debug("Requesting human input:", data=request)

        async def call_callback_and_signal():
            try:
                user_input = await self.human_input_callback(request)

                self.logger.debug("Received human input:", data=user_input)
                await self.executor.signal(signal_name=request_id, payload=user_input)
            except PromptExitError as e:
                # Propagate the exit error through the signal system
                self.logger.info("User requested to exit session")
                await self.executor.signal(
                    signal_name=request_id,
                    payload={"exit_requested": True, "error": str(e)},
                )
            except Exception as e:
                await self.executor.signal(
                    request_id, payload=f"Error getting human input: {str(e)}"
                )

        asyncio.create_task(call_callback_and_signal())

        self.logger.debug("Waiting for human input signal")

        # Wait for signal (workflow is paused here)
        result = await self.executor.wait_for_signal(
            signal_name=request_id,
            request_id=request_id,
            workflow_id=request.workflow_id,
            signal_description=request.description or request.prompt,
            timeout_seconds=request.timeout_seconds,
            signal_type=HumanInputResponse,  # TODO: saqadri - should this be HumanInputResponse?
        )

        if isinstance(result, dict) and result.get("exit_requested", False):
            raise PromptExitError(
                result.get("error", "User requested to exit FastAgent session")
            )
        self.logger.debug("Received human input signal", data=result)
        return result

    async def list_tools(self) -> ListToolsResult:
        if not self.initialized:
            await self.initialize()

        result = await super().list_tools()

        # Add function tools
        for tool in self._function_tool_map.values():
            result.tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.parameters,
                )
            )

        # Add a human_input_callback as a tool
        if not self.human_input_callback:
            self.logger.debug("Human input callback not set")
            return result

        # Add a human_input_callback as a tool
        human_input_tool: FastTool = FastTool.from_function(self.request_human_input)
        result.tools.append(
            Tool(
                name=HUMAN_INPUT_TOOL_NAME,
                description=human_input_tool.description,
                inputSchema=human_input_tool.parameters,
            )
        )

        return result

    # todo would prefer to use tool_name to disambiguate agent name
    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        if name == HUMAN_INPUT_TOOL_NAME:
            # Call the human input tool
            return await self._call_human_input_tool(arguments)
        elif name in self._function_tool_map:
            # Call local function and return the result as a text response
            tool = self._function_tool_map[name]
            result = await tool.run(arguments)
            return CallToolResult(content=[TextContent(type="text", text=str(result))])
        else:
            return await super().call_tool(name, arguments)

    async def _call_human_input_tool(
        self, arguments: dict | None = None
    ) -> CallToolResult:
        # Handle human input request
        try:
            # Make sure arguments is not None
            if arguments is None:
                arguments = {}

            # Extract request data
            request_data = arguments.get("request")

            # Handle both string and dict request formats
            if isinstance(request_data, str):
                request = HumanInputRequest(prompt=request_data)
            elif isinstance(request_data, dict):
                request = HumanInputRequest(**request_data)
            else:
                # Fallback for invalid or missing request data
                request = HumanInputRequest(prompt="Please provide input:")

            result = await self.request_human_input(request=request)

            # Use response attribute if available, otherwise use the result directly
            response_text = (
                result.response
                if isinstance(result, HumanInputResponse)
                else str(result)
            )

            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Human response: {response_text}")
                ]
            )

        except PromptExitError:
            raise
        except TimeoutError as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Human input request timed out: {str(e)}",
                    )
                ],
            )
        except Exception as e:
            print(f"Error in _call_human_input_tool: {traceback.format_exc()}")

            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text", text=f"Error requesting human input: {str(e)}"
                    )
                ],
            )

    async def read_resource(
        self, server_name: str, resource_name: str
    ) -> ReadResourceResult:
        return None

    async def apply_prompt(
        self, prompt_name: str, arguments: dict[str, str] = None
    ) -> str:
        """
        Apply an MCP Server Prompt by name and return the assistant's response.
        Will search all available servers for the prompt if not namespaced.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt_name: The name of the prompt to apply
            arguments: Optional dictionary of string arguments to pass to the prompt template

        Returns:
            The assistant's response or error message
        """
        # If we don't have an LLM, we can't apply the prompt
        if not hasattr(self, "_llm") or not self._llm:
            raise RuntimeError("Agent has no attached LLM")

        # Get the prompt - this will search all servers if needed
        self.logger.debug(f"Loading prompt '{prompt_name}'")
        prompt_result = await self.get_prompt(prompt_name, arguments)

        if not prompt_result or not prompt_result.messages:
            error_msg = (
                f"Prompt '{prompt_name}' could not be found or contains no messages"
            )
            self.logger.warning(error_msg)
            return error_msg

        # Get the display name (namespaced version)
        display_name = getattr(prompt_result, "namespaced_name", prompt_name)

        # Apply the prompt template and get the result
        # The LLM will automatically generate a response if needed
        result = await self._llm.apply_prompt_template(prompt_result, display_name)
        return result

    async def get_resource(self, server_name: str, resource_name: str):
        """
        Get a resource directly from an MCP server by name.

        Args:
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name of the resource to retrieve

        Returns:
            The resource object from the MCP server

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        if not self.initialized:
            await self.initialize()

        # Get the specified server connection
        server = self.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found or not connected")

        # Request the resource directly from the server
        try:
            resource_result = await server.get_resource(resource_name)
            return resource_result
        except Exception as e:
            self.logger.error(
                f"Error retrieving resource '{resource_name}' from server '{server_name}': {str(e)}"
            )
            raise ValueError(
                f"Failed to retrieve resource '{resource_name}' from server '{server_name}': {str(e)}"
            )

    async def get_embedded_resources(
        self, server_name: str, resource_name: str
    ) -> List[EmbeddedResource]:
        """
        Get a resource from an MCP server and return it as a list of embedded resources ready for use in prompts.

        Args:
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name or URI of the resource to retrieve

        Returns:
            List of EmbeddedResource objects ready to use in a PromptMessageMultipart

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        result: ReadResourceResult = await super().get_resource(
            server_name, resource_name
        )

        # Convert each resource content to an EmbeddedResource
        embedded_resources: List[EmbeddedResource] = []
        for resource_content in result.contents:
            embedded_resource = EmbeddedResource(
                type="resource", resource=resource_content, annotations=None
            )
            embedded_resources.append(embedded_resource)

        return embedded_resources

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessageMultipart],
        server_name: str,
        resource_name: str,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Either a string message or an existing PromptMessageMultipart
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name or URI of the resource to retrieve

        Returns:
            The agent's response as a string
        """
        # Get the embedded resources
        embedded_resources: List[EmbeddedResource] = await self.get_embedded_resources(
            server_name, resource_name
        )

        # Create or update the prompt message
        prompt: PromptMessageMultipart
        if isinstance(prompt_content, str):
            # Create a new prompt with the text and resources
            content = [TextContent(type="text", text=prompt_content)]
            content.extend(embedded_resources)
            prompt = PromptMessageMultipart(role="user", content=content)
        elif isinstance(prompt_content, PromptMessageMultipart):
            # Add resources to the existing prompt
            prompt = prompt_content
            prompt.content.extend(embedded_resources)
        else:
            raise TypeError("prompt_content must be a string or PromptMessageMultipart")

        # Send the prompt to the agent and return the response
        return await self._llm.generate_prompt(prompt, None)
