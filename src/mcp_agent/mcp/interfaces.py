"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from datetime import timedelta
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from deprecated import deprecated
from mcp import ClientSession, GetPromptResult, ReadResourceResult
from pydantic import BaseModel

from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@runtime_checkable
class MCPConnectionManagerProtocol(Protocol):
    """Protocol for MCPConnectionManager functionality needed by ServerRegistry."""

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
    ) -> "ServerConnection": ...

    async def disconnect_server(self, server_name: str) -> None: ...

    async def disconnect_all_servers(self) -> None: ...


@runtime_checkable
class ServerRegistryProtocol(Protocol):
    """Protocol defining the minimal interface of ServerRegistry needed by gen_client."""

    @property
    def connection_manager(self) -> MCPConnectionManagerProtocol: ...

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
        init_hook: Optional[Callable] = None,
    ) -> AsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...


ModelT = TypeVar("ModelT", bound=BaseModel)


class AugmentedLLMProtocol(Protocol):
    """Protocol defining the interface for augmented LLMs"""

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Apply the prompt and return the result as a Pydantic model, or None if coercion fails"""
        ...

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Apply a list of PromptMessageMultipart messages directly to the LLM.


        Args:
            multipart_messages: List of PromptMessageMultipart objects
            request_params: Optional parameters to configure the LLM request

        Returns:
            A PromptMessageMultipart containing the Assistant response, including Tool Content
        """
        ...


class AgentProtocol(AugmentedLLMProtocol, Protocol):
    """Protocol defining the standard agent interface"""

    name: str

    async def __call__(self, message: Union[str, PromptMessageMultipart] | None = None) -> str:
        """Make the agent callable for sending messages directly."""
        ...

    async def send(self, message: Union[str, PromptMessageMultipart]) -> str:
        """Send a message to the agent and get a response"""
        ...

    async def prompt(self, default_prompt: str = "") -> str:
        """Start an interactive prompt session with the agent"""
        ...

    async def apply_prompt(self, prompt_name: str, arguments: Dict[str, str] | None = None) -> str:
        """Apply an MCP prompt template by name"""
        ...

    async def get_prompt(self, prompt_name: str) -> GetPromptResult: ...

    async def list_prompts(self, server_name: str | None) -> Dict[str, List[Prompt]]: ...

    async def get_resource(self, server_name: str, resource_uri: str) -> ReadResourceResult: ...

    @deprecated
    async def generate_str(self, message: str, request_params: RequestParams | None) -> str:
        """Generate a response. Deprecated: please use send instead"""
        ...

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessageMultipart],
        server_name: str,
        resource_name: str,
    ) -> str:
        """Send a message with an attached MCP resource"""
        ...

    async def initialize(self) -> None:
        """Initialize the agent and connect to MCP servers"""
        ...

    async def shutdown(self) -> None:
        """Shut down the agent and close connections"""
        ...


class ModelFactoryClassProtocol(Protocol):
    """
    Protocol defining the minimal interface of the ModelFactory class needed by sampling.
    This allows sampling.py to depend on this protocol rather than the concrete ModelFactory class.
    """

    @classmethod
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., Any]:
        """
        Creates a factory function that can be used to construct an LLM instance.

        Args:
            model_string: The model specification string
            request_params: Optional parameters to configure LLM behavior

        Returns:
            A factory function that can create an LLM instance
        """
        ...
