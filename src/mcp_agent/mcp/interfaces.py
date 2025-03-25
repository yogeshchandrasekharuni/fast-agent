"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from datetime import timedelta
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.types import PromptMessage

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


# Regular invariant type variables
MessageParamT = TypeVar("MessageParamT")
MessageT = TypeVar("MessageT")
ModelT = TypeVar("ModelT")

# Variance-annotated type variables
MessageParamT_co = TypeVar("MessageParamT_co", contravariant=True)
MessageT_co = TypeVar("MessageT_co")


class AugmentedLLMProtocol(Protocol, Generic[MessageParamT_co, MessageT_co]):
    """Protocol defining the interface for augmented LLMs"""

    async def generate(
        self,
        message: Union[str, MessageParamT_co, List[MessageParamT_co]],
        request_params: RequestParams | None = None,
    ) -> List[MessageT_co]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""
        ...

    async def generate_str(
        self,
        message: Union[str, MessageParamT_co, List[MessageParamT_co]],
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""
        ...

    async def structured(
        self,
        prompt: Union[str, PromptMessage, PromptMessageMultipart, List[str]],
        model: Type[ModelT],
        request_params: RequestParams | None,
    ) -> ModelT:
        """Apply the prompt and return the result as a Pydantic model, or None if coercion fails"""
        ...

    async def generate_prompt(
        self,
        prompt: Union[str, PromptMessage, PromptMessageMultipart, List[str]],
        request_params: RequestParams | None,
    ) -> str:
        """Request an LLM generation and return a string representation of the result"""
        ...

    async def apply_prompt(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Apply a list of PromptMessageMultipart messages directly to the LLM.
        This is a cleaner interface to _apply_prompt_template_provider_specific.

        Args:
            multipart_messages: List of PromptMessageMultipart objects
            request_params: Optional parameters to configure the LLM request

        Returns:
            String representation of the assistant's response
        """
        ...


class ModelFactoryClassProtocol(Protocol):
    """
    Protocol defining the minimal interface of the ModelFactory class needed by sampling.
    This allows sampling.py to depend on this protocol rather than the concrete ModelFactory class.
    """

    @classmethod
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., AugmentedLLMProtocol[Any, Any]]:
        """
        Creates a factory function that can be used to construct an LLM instance.

        Args:
            model_string: The model specification string
            request_params: Optional parameters to configure LLM behavior

        Returns:
            A factory function that can create an LLM instance
        """
        ...
