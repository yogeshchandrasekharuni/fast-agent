"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
)

from mcp import ClientSession
from mcp.types import CreateMessageRequestParams
from pydantic import Field

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class ServerRegistryProtocol(Protocol):
    """
    Protocol defining the minimal interface of ServerRegistry needed by gen_client.
    This allows gen_client to depend on this protocol rather than the full ServerRegistry class.
    """

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory=None,
        init_hook=None,
    ) -> AsyncGenerator[ClientSession, None]:
        """Initialize a server and yield a client session."""
        ...

    @property
    def connection_manager(self) -> "ConnectionManagerProtocol":
        """Get the connection manager."""
        ...


class ConnectionManagerProtocol(Protocol):
    """
    Protocol defining the minimal interface of ConnectionManager needed.
    """

    async def get_server(
        self,
        server_name: str,
        client_session_factory=None,
    ):
        """Get a server connection."""
        ...

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from a server."""
        ...

    async def disconnect_all_servers(self) -> None:
        """Disconnect from all servers."""
        ...


# Type variables for generic protocols
MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""

ModelT = TypeVar("ModelT")
"""A type representing a structured output message from an LLM."""


class RequestParams(CreateMessageRequestParams):
    """
    Parameters to configure the AugmentedLLM 'generate' requests.
    """

    messages: None = Field(exclude=True, default=None)
    """
    Ignored. 'messages' are removed from CreateMessageRequestParams 
    to avoid confusion with the 'message' parameter on 'generate' method.
    """

    maxTokens: int = 2048
    """The maximum number of tokens to sample, as requested by the server."""

    model: str | None = None
    """
    The model to use for the LLM generation.
    If specified, this overrides the 'modelPreferences' selection criteria.
    """

    use_history: bool = True
    """
    Include the message history in the generate request.
    """

    max_iterations: int = 10
    """
    The maximum number of iterations to run the LLM for.
    """

    parallel_tool_calls: bool = True
    """
    Whether to allow multiple tool calls per iteration.
    Also known as multi-step tool use.
    """


class AugmentedLLMProtocol(Protocol, Generic[MessageParamT, MessageT]):
    """Protocol defining the interface for augmented LLMs"""

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""

    async def generate_prompt(
        self, prompt: PromptMessageMultipart, request_params: RequestParams | None
    ) -> str:
        """Request an LLM generation and return a string representation of the result"""

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
