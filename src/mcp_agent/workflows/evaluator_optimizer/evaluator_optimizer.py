import contextlib
from enum import Enum
from typing import Callable, List, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel, Field

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)


class QualityRating(str, Enum):
    """Enum for evaluation quality ratings"""

    POOR = 0  # Major improvements needed
    FAIR = 1  # Several improvements needed
    GOOD = 2  # Minor improvements possible
    EXCELLENT = 3  # No improvements needed


class EvaluationResult(BaseModel):
    """Model representing the evaluation result from the evaluator LLM"""

    rating: QualityRating = Field(description="Quality rating of the response")
    feedback: str = Field(
        description="Specific feedback and suggestions for improvement"
    )
    needs_improvement: bool = Field(
        description="Whether the output needs further improvement"
    )
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific areas to focus on in next iteration"
    )


class EvaluatorOptimizerLLM(AugmentedLLM[MessageParamT, MessageT]):
    """
    Implementation of the evaluator-optimizer workflow where one LLM generates responses
    while another provides evaluation and feedback in a refinement loop.

    This can be used either:
    1. As a standalone workflow with its own optimizer agent
    2. As a wrapper around another workflow (Orchestrator, Router, ParallelLLM) to add
       evaluation and refinement capabilities

    When to use this workflow:
    - When you have clear evaluation criteria and iterative refinement provides value
    - When LLM responses improve with articulated feedback
    - When the task benefits from focused iteration on specific aspects

    Examples:
    - Literary translation with "expert" refinement
    - Complex search tasks needing multiple rounds
    - Document writing requiring multiple revisions
    """

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize default parameters using the workflow's settings."""
        return RequestParams(
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=self.generator_use_history,  # Use generator's history setting
        )

    def _init_request_params(self):
        """Initialize request parameters for both generator and evaluator components."""
        # Set up workflow's default params based on generator's history setting
        self.default_request_params = self._initialize_default_params({})

        # Ensure evaluator's request params have history disabled
        if hasattr(self.evaluator_llm, "default_request_params"):
            self.evaluator_llm.default_request_params.use_history = False

    def __init__(
        self,
        generator: Agent | AugmentedLLM,
        evaluator: str | Agent | AugmentedLLM,
        min_rating: QualityRating = QualityRating.GOOD,
        max_refinements: int = 3,
        llm_factory: Callable[[Agent], AugmentedLLM] | None = None,
        context: Optional["Context"] = None,
        name: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        """
        Initialize the evaluator-optimizer workflow.

        Args:
            generator: The agent/LLM/workflow that generates responses
            evaluator: The evaluator (string instruction, Agent or AugmentedLLM)
            min_rating: Minimum acceptable quality rating
            max_refinements: Maximum refinement iterations
            llm_factory: Factory to create LLMs from agents when needed
            name: Optional name for the workflow (defaults to generator's name)
            instruction: Optional instruction (defaults to generator's instruction)
        """
        # Set initial attributes
        self.name = name or getattr(generator, "name", "EvaluatorOptimizer")
        self.llm_factory = llm_factory
        self.generator = generator
        self.evaluator = evaluator
        self.min_rating = min_rating
        self.max_refinements = max_refinements

        # Determine generator's history setting directly based on type
        self.generator_use_history = False
        if isinstance(generator, Agent):
            self.generator_use_history = generator.config.use_history
        elif isinstance(generator, AugmentedLLM):
            if hasattr(generator, "aggregator") and isinstance(
                generator.aggregator, Agent
            ):
                self.generator_use_history = generator.aggregator.config.use_history
            elif hasattr(generator, "default_request_params"):
                self.generator_use_history = getattr(
                    generator.default_request_params, "use_history", False
                )
        # All other types default to False

        # Initialize parent class
        super().__init__(context=context, name=name or getattr(generator, "name", None))

        # Create a PassthroughLLM as _llm property
        # TODO -- remove this when we fix/remove the inheritance hierarchy
        self._llm = PassthroughLLM(name=f"{self.name}_passthrough", context=context)

        # Set up the generator based on type
        if isinstance(generator, Agent):
            if not llm_factory:
                raise ValueError(
                    "llm_factory is required when using an Agent generator"
                )

            # Use existing LLM if available, otherwise create new one
            self.generator_llm = getattr(generator, "_llm", None) or llm_factory(
                agent=generator
            )
            self.aggregator = generator
            self.instruction = instruction or (
                generator.instruction
                if isinstance(generator.instruction, str)
                else None
            )
        elif isinstance(generator, AugmentedLLM):
            self.generator_llm = generator
            self.aggregator = getattr(generator, "aggregator", None)
            self.instruction = instruction or generator.instruction
        else:
            # ChainProxy-like object
            self.generator_llm = generator
            self.aggregator = None
            self.instruction = (
                instruction or f"Chain of agents: {', '.join(generator._sequence)}"
            )

        # Set up the evaluator - always disable history
        if isinstance(evaluator, str):
            if not llm_factory:
                raise ValueError(
                    "llm_factory is required when using a string evaluator"
                )

            evaluator_agent = Agent(
                name="Evaluator",
                instruction=evaluator,
                config=AgentConfig(
                    name="Evaluator",
                    instruction=evaluator,
                    servers=[],
                    use_history=False,
                ),
            )
            self.evaluator_llm = llm_factory(agent=evaluator_agent)
        elif isinstance(evaluator, Agent):
            if not llm_factory:
                raise ValueError(
                    "llm_factory is required when using an Agent evaluator"
                )

            # Disable history and use/create LLM
            evaluator.config.use_history = False
            self.evaluator_llm = getattr(evaluator, "_llm", None) or llm_factory(
                agent=evaluator
            )
        elif isinstance(evaluator, AugmentedLLM):
            self.evaluator_llm = evaluator
            # Ensure history is disabled
            if hasattr(self.evaluator_llm, "default_request_params"):
                self.evaluator_llm.default_request_params.use_history = False
        else:
            raise ValueError(f"Unsupported evaluator type: {type(evaluator)}")

        # Track iteration history
        self.refinement_history = []

        # Set up workflow's default params
        self.default_request_params = self._initialize_default_params({})

        # Ensure evaluator's request params have history disabled
        if hasattr(self.evaluator_llm, "default_request_params"):
            self.evaluator_llm.default_request_params.use_history = False

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Generate an optimized response through evaluation-guided refinement"""
        refinement_count = 0
        response = None
        best_response = None
        best_rating = QualityRating.POOR
        self.refinement_history = []

        # Get request params with proper use_history setting
        params = self.get_request_params(request_params)

        # Use a single AsyncExitStack for the entire method to maintain connections
        async with contextlib.AsyncExitStack() as stack:
            # Enter all agent contexts once at the beginning
            if isinstance(self.generator, Agent):
                await stack.enter_async_context(self.generator)
            if isinstance(self.evaluator, Agent):
                await stack.enter_async_context(self.evaluator)

            # Initial generation - pass parameters to any type of generator
            response = await self.generator_llm.generate_str(
                message=message,
                request_params=params,  # Pass params which may override use_history
            )

            best_response = response

            while refinement_count < self.max_refinements:
                logger.debug("Generator result:", data=response)

                # Evaluate current response
                eval_prompt = self._build_eval_prompt(
                    original_request=str(message),
                    current_response=response,  # response is already a string
                    iteration=refinement_count,
                )

                # No need for nested AsyncExitStack here - using the outer one
                evaluation_result = await self.evaluator_llm.generate_structured(
                    message=eval_prompt,
                    response_model=EvaluationResult,
                    request_params=request_params,
                )

                # Track iteration
                self.refinement_history.append(
                    {
                        "attempt": refinement_count + 1,
                        "response": response,
                        "evaluation_result": evaluation_result,
                    }
                )

                logger.debug("Evaluator result:", data=evaluation_result)

                # Track best response (using enum ordering)
                if evaluation_result.rating.value > best_rating.value:
                    best_rating = evaluation_result.rating
                    best_response = response
                    logger.debug(
                        "New best response:",
                        data={"rating": best_rating, "response": best_response},
                    )

                # Check if we've reached acceptable quality
                if (
                    evaluation_result.rating.value >= self.min_rating.value
                    or not evaluation_result.needs_improvement
                ):
                    logger.debug(
                        f"Acceptable quality {evaluation_result.rating.value} reached",
                        data={
                            "rating": evaluation_result.rating.value,
                            "needs_improvement": evaluation_result.needs_improvement,
                            "min_rating": self.min_rating.value,
                        },
                    )
                    break

                # Generate refined response
                refinement_prompt = self._build_refinement_prompt(
                    original_request=str(message),
                    current_response=response,
                    feedback=evaluation_result,
                    iteration=refinement_count,
                    use_history=self.generator_use_history,  # Use the generator's history setting
                )

                # Pass parameters to any type of generator
                response = await self.generator_llm.generate_str(
                    message=refinement_prompt,
                    request_params=params,  # Pass params which may override use_history
                )

                refinement_count += 1

            # Return the best response as a list with a single string element
            # This makes it consistent with other AugmentedLLM implementations
            # that return List[MessageT]
            return [best_response]

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Generate an optimized response and return it as a string"""
        response = await self.generate(
            message=message,
            request_params=request_params,
        )
        # Since generate now returns [best_response], just return the first element
        return str(response[0])

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Generate an optimized structured response"""
        response_str = await self.generate_str(
            message=message, request_params=request_params
        )

        return await self.generator.generate_structured(
            message=response_str,
            response_model=response_model,
            request_params=request_params,
        )

    def _build_eval_prompt(
        self, original_request: str, current_response: str, iteration: int
    ) -> str:
        """Build the evaluation prompt for the evaluator"""
        return f"""
You are an expert evaluator for content quality. Your task is to evaluate a response against the user's original request.

Evaluate the response for iteration {iteration + 1} and provide structured feedback on its quality and areas for improvement.

<fastagent:data>
<fastagent:request>
{original_request}
</fastagent:request>

<fastagent:response>
{current_response}
</fastagent:response>

<fastagent:evaluation-criteria>
{self.evaluator.instruction}
</fastagent:evaluation-criteria>
</fastagent:data>

<fastagent:instruction>
Provide a structured evaluation with the following components:

<rating>
Choose one: EXCELLENT, GOOD, FAIR, or POOR
- EXCELLENT: No improvements needed
- GOOD: Only minor improvements possible
- FAIR: Several improvements needed
- POOR: Major improvements needed
</rating>

<details>
Provide specific, actionable feedback and suggestions for improvement.
Be precise about what works well and what could be improved.
</details>

<needs_improvement>
Indicate true/false whether further improvement is needed.
</needs_improvement>

<focus-areas>
List 1-3 specific areas to focus on in the next iteration.
Be concrete and actionable in your recommendations.
</focus-areas>
</fastagent:instruction>
"""

    def _build_refinement_prompt(
        self,
        original_request: str,
        current_response: str,
        feedback: EvaluationResult,
        iteration: int,
        use_history: bool = None,
    ) -> str:
        """Build the refinement prompt for the optimizer"""
        # Get the correct history setting - use param if provided, otherwise class default
        if use_history is None:
            use_history = (
                self.generator_use_history
            )  # Use generator's setting as default

        # Start with clear non-delimited instructions
        prompt = f"""
You are tasked with improving a response based on expert feedback. This is iteration {iteration + 1} of the refinement process.

Your goal is to address all feedback points while maintaining accuracy and relevance to the original request.
"""

        # Add data section with all relevant information
        prompt += """
<fastagent:data>
"""

        # Add request
        prompt += f"""
<fastagent:request>
{original_request}
</fastagent:request>
"""

        # Only include previous response if history is not enabled
        if not use_history:
            prompt += f"""
<fastagent:previous-response>
{current_response}
</fastagent:previous-response>
"""

        # Always include the feedback
        prompt += f"""
<fastagent:feedback>
<rating>{feedback.rating}</rating>
<details>{feedback.feedback}</details>
<focus-areas>{", ".join(feedback.focus_areas) if feedback.focus_areas else "None specified"}</focus-areas>
</fastagent:feedback>
</fastagent:data>
"""

        # Customize instruction based on history availability
        if not use_history:
            prompt += """
<fastagent:instruction>
Create an improved version of the response that:
1. Directly addresses each point in the feedback
2. Focuses on the specific areas mentioned for improvement
3. Maintains all the strengths of the original response
4. Remains accurate and relevant to the original request

Provide your complete improved response without explanations or commentary.
</fastagent:instruction>
"""
        else:
            prompt += """
<fastagent:instruction>
Your previous response is available in your conversation history.

Create an improved version that:
1. Directly addresses each point in the feedback
2. Focuses on the specific areas mentioned for improvement
3. Maintains all the strengths of your original response
4. Remains accurate and relevant to the original request

Provide your complete improved response without explanations or commentary.
</fastagent:instruction>
"""

        return prompt
