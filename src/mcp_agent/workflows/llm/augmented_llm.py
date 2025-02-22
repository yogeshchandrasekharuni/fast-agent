from abc import abstractmethod

from typing import Generic, List, Optional, Protocol, Type, TypeVar, TYPE_CHECKING

from pydantic import Field

from mcp.types import (
    CallToolRequest,
    CallToolResult,
    CreateMessageRequestParams,
    CreateMessageResult,
    ModelPreferences,
    SamplingMessage,
    TextContent,
)

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.event_progress import ProgressAction
from mcp_agent.mcp.mcp_aggregator import MCPAggregator, SEP
from mcp_agent.workflows.llm.llm_selector import ModelSelector
from rich.panel import Panel
from rich.text import Text
from mcp_agent import console

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.context import Context

MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""

ModelT = TypeVar("ModelT")
"""A type representing a structured output message from an LLM."""

# TODO: saqadri - SamplingMessage is fairly limiting - consider extending
MCPMessageParam = SamplingMessage
MCPMessageResult = CreateMessageResult

# TODO -- move this to a constant
HUMAN_INPUT_TOOL_NAME = "__human_input__"


class Memory(Protocol, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    # TODO: saqadri - add checkpointing and other advanced memory capabilities

    def __init__(self): ...

    def extend(self, messages: List[MessageParamT]) -> None: ...

    def set(self, messages: List[MessageParamT]) -> None: ...

    def append(self, message: MessageParamT) -> None: ...

    def get(self) -> List[MessageParamT]: ...

    def clear(self) -> None: ...


class SimpleMemory(Memory, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    def __init__(self):
        self.history: List[MessageParamT] = []

    def extend(self, messages: List[MessageParamT]):
        self.history.extend(messages)

    def set(self, messages: List[MessageParamT]):
        self.history = messages.copy()

    def append(self, message: MessageParamT):
        self.history.append(message)

    def get(self) -> List[MessageParamT]:
        return self.history

    def clear(self):
        self.history = []


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


class ProviderToMCPConverter(Protocol, Generic[MessageParamT, MessageT]):
    """Conversions between LLM provider and MCP types"""

    @classmethod
    def to_mcp_message_result(cls, result: MessageT) -> MCPMessageResult:
        """Convert an LLM response to an MCP message result type."""

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> MessageT:
        """Convert an MCP message result to an LLM response type."""

    @classmethod
    def to_mcp_message_param(cls, param: MessageParamT) -> MCPMessageParam:
        """Convert an LLM input to an MCP message (SamplingMessage) type."""

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> MessageParamT:
        """Convert an MCP message (SamplingMessage) to an LLM input type."""


class AugmentedLLM(ContextDependent, AugmentedLLMProtocol[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    # TODO: saqadri - add streaming support (e.g. generate_stream)
    # TODO: saqadri - consider adding middleware patterns for pre/post processing of messages, for now we have pre/post_tool_call

    provider: str | None = None

    def __init__(
        self,
        agent: Optional["Agent"] = None,
        server_names: List[str] | None = None,
        instruction: str | None = None,
        name: str | None = None,
        request_params: RequestParams | None = None,
        type_converter: Type[ProviderToMCPConverter[MessageParamT, MessageT]] = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        # Extract request_params before super() call
        self._init_request_params = request_params
        super().__init__(context=context, **kwargs)

        self.executor = self.context.executor
        self.aggregator = (
            agent if agent is not None else MCPAggregator(server_names or [])
        )
        self.name = name or (agent.name if agent else None)
        self.instruction = instruction or (
            agent.instruction if agent and isinstance(agent.instruction, str) else None
        )
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

        # Set initial model preferences
        self.model_preferences = ModelPreferences(
            costPriority=0.3,
            speedPriority=0.4,
            intelligencePriority=0.3,
        )

        # Initialize default parameters
        self.default_request_params = self._initialize_default_params(kwargs)

        # Update model preferences from default params
        if self.default_request_params and self.default_request_params.modelPreferences:
            self.model_preferences = self.default_request_params.modelPreferences

        # Merge with provided params if any
        if self._init_request_params:
            self.default_request_params = self._merge_request_params(
                self.default_request_params, self._init_request_params
            )
            # Update model preferences again if they changed in the merge
            if self.default_request_params.modelPreferences:
                self.model_preferences = self.default_request_params.modelPreferences

        self.model_selector = self.context.model_selector
        self.type_converter = type_converter

    @abstractmethod
    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    @abstractmethod
    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    @abstractmethod
    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""

    async def select_model(
        self, request_params: RequestParams | None = None
    ) -> str | None:
        """
        Select an LLM based on the request parameters.
        If a model is specified in the request, it will override the model selection criteria.
        """
        model_preferences = self.model_preferences
        if request_params is not None:
            model_preferences = request_params.modelPreferences or model_preferences
            model = request_params.model
            if model:
                return model

        ## TODO -- can't have been tested, returns invalid model strings (e.g. claude-35-sonnet)
        if not self.model_selector:
            self.model_selector = ModelSelector()

        model_info = self.model_selector.select_best_model(
            model_preferences=model_preferences, provider=self.provider
        )

        return model_info.name

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize default parameters for the LLM.
        Should be overridden by provider implementations to set provider-specific defaults."""
        return RequestParams(
            modelPreferences=self.model_preferences,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _merge_request_params(
        self, default_params: RequestParams, provided_params: RequestParams
    ) -> RequestParams:
        """Merge default and provided request parameters"""
        # Log parameter merging if debug logging is enabled
        # self.context.config.logger.debug(
        #     "Merging provided request params with defaults",
        #     extra={
        #         "defaults": default_params.model_dump(),
        #         "provided": provided_params.model_dump(),
        #     },
        # )

        merged = default_params.model_dump()
        merged.update(provided_params.model_dump(exclude_unset=True))
        final_params = RequestParams(**merged)

        # self.logger.debug(
        #     "Final merged params:", extra={"params": final_params.model_dump()}
        # )

        return final_params

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
        default: RequestParams | None = None,
    ) -> RequestParams:
        """
        Get request parameters with merged-in defaults and overrides.
        Args:
            request_params: The request parameters to use as overrides.
            default: The default request parameters to use as the base.
                If unspecified, self.default_request_params will be used.
        """
        # Start with the defaults
        default_request_params = default or self.default_request_params

        if not default_request_params:
            default_request_params = self._initialize_default_params({})

        # If user provides overrides, merge them with defaults
        if request_params:
            return self._merge_request_params(default_request_params, request_params)

        return default_request_params

    def to_mcp_message_result(self, result: MessageT) -> MCPMessageResult:
        """Convert an LLM response to an MCP message result type."""
        return self.type_converter.to_mcp_message_result(result)

    def from_mcp_message_result(self, result: MCPMessageResult) -> MessageT:
        """Convert an MCP message result to an LLM response type."""
        return self.type_converter.from_mcp_message_result(result)

    def to_mcp_message_param(self, param: MessageParamT) -> MCPMessageParam:
        """Convert an LLM input to an MCP message (SamplingMessage) type."""
        return self.type_converter.to_mcp_message_param(param)

    def from_mcp_message_param(self, param: MCPMessageParam) -> MessageParamT:
        """Convert an MCP message (SamplingMessage) to an LLM input type."""
        return self.type_converter.from_mcp_message_param(param)

    @classmethod
    def convert_message_to_message_param(
        cls, message: MessageT, **kwargs
    ) -> MessageParamT:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        # Many LLM implementations will allow the same type for input and output messages
        return message

    async def get_last_message(self) -> MessageParamT | None:
        """
        Return the last message generated by the LLM or None if history is empty.
        This is useful for prompt chaining workflows where the last message from one LLM is used as input to another.
        """
        history = self.history.get()
        return history[-1] if history else None

    async def get_last_message_str(self) -> str | None:
        """Return the string representation of the last message generated by the LLM or None if history is empty."""
        last_message = await self.get_last_message()
        return self.message_param_str(last_message) if last_message else None

    def show_tool_result(self, result: CallToolResult):
        """Display a tool result in a formatted panel."""

        if not self.context.config.logger.show_tools:
            return

        if result.isError:
            style = "red"
        else:
            style = "magenta"

        panel = Panel(
            Text(
                str(result.content), overflow="..."
            ),  # TODO support multi-model/multi-part responses
            title="[TOOL RESULT]",
            title_align="right",
            style=style,
            border_style="bold white",
            padding=(1, 2),
        )

        if self.context.config.logger.truncate_tools:
            if len(str(result.content)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    def show_oai_tool_result(self, result):
        """Display a tool result in a formatted panel."""

        if not self.context.config.logger.show_tools:
            return

        panel = Panel(
            Text(str(result), overflow="..."),  # TODO update openai support
            title="[TOOL RESULT]",
            title_align="right",
            style="magenta",
            border_style="bold white",
            padding=(1, 2),
        )

        if self.context.config.logger.truncate_tools:
            if len(str(result)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    def show_tool_call(self, available_tools, tool_name, tool_args):
        """Display a tool call in a formatted panel."""

        if not self.context.config.logger.show_tools:
            return

        display_tool_list = Text()
        for display_tool in available_tools:
            # Handle both OpenAI and Anthropic tool formats
            # TODO -- this should really be using the ToolCall abstraction and converting at the concrete layer??
            if isinstance(display_tool, dict):
                if "function" in display_tool:
                    # OpenAI format
                    tool_call_name = display_tool["function"]["name"]
                else:
                    # Anthropic format
                    tool_call_name = display_tool["name"]
            else:
                # Handle potential object format (e.g., Pydantic models)
                tool_call_name = (
                    display_tool.function.name
                    if hasattr(display_tool, "function")
                    else display_tool.name
                )

            parts = (
                tool_call_name.split(SEP)
                if SEP in tool_call_name
                else [tool_call_name, tool_call_name]
            )
            if tool_name.split(SEP)[0] == parts[0]:
                if tool_call_name == tool_name:
                    style = "magenta"
                else:
                    style = "dim white"

                shortened_name = (
                    parts[1] if len(parts[1]) <= 12 else parts[1][:11] + "…"
                )
                display_tool_list.append(f"[{shortened_name}] ", style)

        panel = Panel(
            Text(str(tool_args), overflow="ellipsis"),
            title="[TOOL CALL]",
            title_align="left",
            style="magenta",
            border_style="bold white",
            subtitle=display_tool_list,
            subtitle_align="left",
            padding=(1, 2),
        )

        if self.context.config.logger.truncate_tools:
            if len(str(tool_args)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    async def show_assistant_message(
        self, message_text: str | Text, highlight_namespaced_tool: str = ""
    ):
        """Display an assistant message in a formatted panel."""

        if not self.context.config.logger.show_chat:
            return

        mcp_server_name = (
            highlight_namespaced_tool.split(SEP)
            if SEP in highlight_namespaced_tool
            else [highlight_namespaced_tool]
        )

        display_server_list = Text()

        tools = await self.aggregator.list_tools()
        if any(tool.name == HUMAN_INPUT_TOOL_NAME for tool in tools.tools):
            style = (
                "green"
                if highlight_namespaced_tool == HUMAN_INPUT_TOOL_NAME
                else "dim white"
            )
            display_server_list.append("[human] ", style)

        for server_name in await self.aggregator.list_servers():
            style = "green" if server_name == mcp_server_name[0] else "dim white"
            display_server_list.append(f"[{server_name}] ", style)

        panel = Panel(
            message_text,
            title=f"[ASSISTANT]{f' ({self.name})' if self.name else ''}",
            title_align="left",
            style="green",
            border_style="bold white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print(panel)
        console.console.print("\n")

    def show_user_message(self, message, model: str | None, chat_turn: int):
        """Display a user message in a formatted panel."""

        if not self.context.config.logger.show_chat:
            return

        panel = Panel(
            message,
            title=f"{f'({self.name}) [USER]' if self.name else '[USER]'}",
            title_align="right",
            style="blue",
            border_style="bold white",
            padding=(1, 2),
            subtitle=Text(f"{model} turn {chat_turn}", style="dim white"),
            subtitle_align="left",
        )
        console.console.print(panel)
        console.console.print("\n")

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(
        self,
        request: CallToolRequest,
        tool_call_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""

        try:
            preprocess = await self.pre_tool_call(
                tool_call_id=tool_call_id,
                request=request,
            )

            if isinstance(preprocess, bool):
                if not preprocess:
                    return CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                text=f"Error: Tool '{request.params.name}' was not allowed to run."
                            )
                        ],
                    )
            else:
                request = preprocess

            tool_name = request.params.name
            tool_args = request.params.arguments
            result = await self.aggregator.call_tool(tool_name, tool_args)

            postprocess = await self.post_tool_call(
                tool_call_id=tool_call_id, request=request, result=result
            )

            if isinstance(postprocess, CallToolResult):
                result = postprocess

            return result
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error executing tool '{request.params.name}': {str(e)}",
                    )
                ],
            )

    def message_param_str(self, message: MessageParamT) -> str:
        """Convert an input message to a string representation."""
        return str(message)

    def message_str(self, message: MessageT) -> str:
        """Convert an output message to a string representation."""
        return str(message)

    def _log_chat_progress(
        self, chat_turn: Optional[int] = None, model: Optional[str] = None
    ):
        """Log a chat progress event"""
        data = {
            "progress_action": ProgressAction.CHATTING,
            "model": model,
            "agent_name": self.name,
            "chat_turn": chat_turn if chat_turn is not None else None,
        }
        self.logger.debug("Chat in progress", data=data)

    def _log_chat_finished(self, model: Optional[str] = None):
        """Log a chat finished event"""
        data = {
            "progress_action": ProgressAction.READY,
            "model": model,
            "agent_name": self.name,
        }
        self.logger.debug("Chat finished", data=data)
