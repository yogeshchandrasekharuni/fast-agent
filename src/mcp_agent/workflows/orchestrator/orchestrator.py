"""
Orchestrator implementation for MCP Agent applications.
"""

from typing import (
    List,
    Literal,
    Optional,
    Type,
    TYPE_CHECKING,
)

from mcp_agent.agents.agent import Agent
from mcp_agent.event_progress import ProgressAction
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)
from mcp_agent.workflows.orchestrator.orchestrator_models import (
    format_plan_result,
    format_step_result,
    NextStep,
    Plan,
    PlanResult,
    Step,
    StepResult,
    TaskWithResult,
)
from mcp_agent.workflows.orchestrator.orchestrator_prompts import (
    FULL_PLAN_PROMPT_TEMPLATE,
    ITERATIVE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_PLAN_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
)
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)


class Orchestrator(AugmentedLLM[MessageParamT, MessageT]):
    """
    In the orchestrator-workers workflow, a central planner LLM dynamically breaks down tasks and
    delegates them to pre-configured worker LLMs. The planner synthesizes their results in a loop
    until the task is complete.

    When to use this workflow:
        - This workflow is well-suited for complex tasks where you can't predict the
        subtasks needed (in coding, for example, the number of files that need to be
        changed and the nature of the change in each file likely depend on the task).

    Example where orchestrator-workers is useful:
        - Coding products that make complex changes to multiple files each time.
        - Search tasks that involve gathering and analyzing information from multiple sources
        for possible relevant information.

    Note:
        All agents must be pre-configured with LLMs before being passed to the orchestrator.
        This ensures consistent model behavior and configuration across all components.
    """

    def __init__(
        self,
        name: str,
        planner: AugmentedLLM,  # Pre-configured planner
        available_agents: List[Agent | AugmentedLLM],
        plan_type: Literal["full", "iterative"] = "full",
        context: Optional["Context"] = None,
        **kwargs,
    ):
        """
        Args:
            name: Name of the orchestrator workflow
            planner: Pre-configured planner LLM to use for planning steps
            available_agents: List of pre-configured agents available to this orchestrator
            plan_type: "full" planning generates the full plan first, then executes. "iterative" plans next step and loops.
            context: Application context
        """
        # Initialize logger early so we can log
        self.logger = logger

        # Set a fixed verb - always use PLANNING for all orchestrator activities
        self.verb = ProgressAction.PLANNING

        # Initialize with orchestrator-specific defaults
        orchestrator_params = RequestParams(
            use_history=False,  # Orchestrator doesn't support history
            max_iterations=30,  # Higher default for complex tasks
            maxTokens=8192,  # Higher default for planning
            parallel_tool_calls=True,
        )

        # If kwargs contains request_params, merge our defaults while preserving the model config
        if "request_params" in kwargs:
            base_params = kwargs["request_params"]
            # Create merged params starting with our defaults
            merged = orchestrator_params.model_copy()
            # Update with base params to get model config
            if isinstance(base_params, dict):
                merged = merged.model_copy(update=base_params)
            else:
                merged = merged.model_copy(update=base_params.model_dump())
            # Force specific settings
            merged.use_history = False
            kwargs["request_params"] = merged
        else:
            kwargs["request_params"] = orchestrator_params

        # Pass verb to AugmentedLLM
        kwargs["verb"] = self.verb

        super().__init__(context=context, **kwargs)

        self.planner = planner

        if hasattr(self.planner, "verb"):
            self.planner.verb = self.verb

        self.plan_type = plan_type
        self.server_registry = self.context.server_registry
        self.agents = {agent.name: agent for agent in available_agents}

        # Initialize logger
        self.logger = logger
        self.name = name

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""
        params = self.get_request_params(request_params)
        objective = str(message)
        plan_result = await self.execute(objective=objective, request_params=params)

        return [plan_result.result]

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""
        params = self.get_request_params(request_params)
        # TODO -- properly incorporate this in to message display etc.
        result = await self.generate(
            message=message,
            request_params=params,
        )

        return str(result[0])

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""
        params = self.get_request_params(request_params)
        result_str = await self.generate_str(message=message, request_params=params)

        # Use AugmentedLLM's structured output handling
        return await super().generate_structured(
            message=result_str,
            response_model=response_model,
            request_params=params,
        )

    async def execute(
        self, objective: str, request_params: RequestParams | None = None
    ) -> PlanResult:
        """Execute task with result chaining between steps"""
        iterations = 0

        params = self.get_request_params(request_params)

        # Single progress event for orchestration start
        model = await self.select_model(params) or "unknown-model"

        # Log the progress with minimal required fields
        self.logger.info(
            "Planning task execution",
            data={
                "progress_action": self.verb,
                "model": model,
                "agent_name": self.name,
                "target": self.name,
            },
        )

        plan_result = PlanResult(objective=objective, step_results=[])

        while iterations < params.max_iterations:
            if self.plan_type == "iterative":
                # Get next plan/step
                next_step = await self._get_next_step(
                    objective=objective, plan_result=plan_result, request_params=params
                )
                logger.debug(f"Iteration {iterations}: Iterative plan:", data=next_step)
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(
                    objective=objective, plan_result=plan_result, request_params=params
                )
                logger.debug(f"Iteration {iterations}: Full Plan:", data=plan)
            else:
                raise ValueError(f"Invalid plan type {self.plan_type}")

            plan_result.plan = plan

            if plan.is_complete:
                # Only mark as complete if we have actually executed some steps
                if len(plan_result.step_results) > 0:
                    plan_result.is_complete = True

                    # Synthesize final result into a single message
                    synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                        plan_result=format_plan_result(plan_result)
                    )

                    # Use planner directly - planner already has PLANNING verb
                    plan_result.result = await self.planner.generate_str(
                        message=synthesis_prompt,
                        request_params=params.model_copy(update={"max_iterations": 1}),
                    )

                    return plan_result
                else:
                    # Don't allow completion without executing steps
                    plan.is_complete = False

            # Execute each step, collecting results
            # Note that in iterative mode this will only be a single step
            for step in plan.steps:
                step_result = await self._execute_step(
                    step=step,
                    previous_result=plan_result,
                    request_params=params,
                )

                plan_result.add_step_result(step_result)

            logger.debug(
                f"Iteration {iterations}: Intermediate plan result:", data=plan_result
            )
            iterations += 1

        raise RuntimeError(
            f"Task failed to complete in {params.max_iterations} iterations"
        )

    async def _execute_step(
        self,
        step: Step,
        previous_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> StepResult:
        """Execute a step's subtasks in parallel and synthesize results"""

        step_result = StepResult(step=step, task_results=[])
        context = format_plan_result(previous_result)

        # Execute tasks
        futures = []
        error_tasks = []

        for task in step.tasks:
            agent = self.agents.get(task.agent)
            if not agent:
                # Instead of failing the entire step, track this as an error task
                self.logger.error(
                    f"No agent found matching '{task.agent}'. Available agents: {list(self.agents.keys())}"
                )
                error_tasks.append(
                    (
                        task,
                        f"Error: Agent '{task.agent}' not found. Available agents: {', '.join(self.agents.keys())}",
                    )
                )
                continue

            task_description = TASK_PROMPT_TEMPLATE.format(
                objective=previous_result.objective,
                task=task.description,
                context=context,
            )

            # All agents should now be LLM-capable
            futures.append(agent._llm.generate_str(message=task_description))

        # Wait for all tasks (only if we have valid futures)
        results = await self.executor.execute(*futures) if futures else []

        # Process successful results
        task_index = 0
        for task in step.tasks:
            # Skip tasks that had agent errors (they're in error_tasks)
            if any(et[0] == task for et in error_tasks):
                continue

            if task_index < len(results):
                result = results[task_index]
                step_result.add_task_result(
                    TaskWithResult(**task.model_dump(), result=str(result))
                )
                task_index += 1

        # Add error task results
        for task, error_message in error_tasks:
            step_result.add_task_result(
                TaskWithResult(**task.model_dump(), result=error_message)
            )

        step_result.result = format_step_result(step_result)
        return step_result

    async def _get_full_plan(
        self,
        objective: str,
        plan_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> Plan:
        """Generate full plan considering previous results"""
        params = self.get_request_params(request_params)
        params = params.model_copy(update={"use_history": False})

        agents = "\n".join(
            [
                f"{idx}. {self._format_agent_info(agent)}"
                for idx, agent in enumerate(self.agents, 1)
            ]
        )

        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            agents=agents,
        )

        # Use planner directly - no verb manipulation needed
        plan = await self.planner.generate_structured(
            message=prompt,
            response_model=Plan,
            request_params=params,
        )

        return plan

    async def _get_next_step(
        self,
        objective: str,
        plan_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> NextStep:
        """Generate just the next needed step"""
        params = self.get_request_params(request_params)
        params = params.model_copy(update={"use_history": False})

        agents = "\n".join(
            [
                f"{idx}. {self._format_agent_info(agent)}"
                for idx, agent in enumerate(self.agents, 1)
            ]
        )

        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            agents=agents,
        )

        # Use planner directly - no verb manipulation needed
        next_step = await self.planner.generate_structured(
            message=prompt,
            response_model=NextStep,
            request_params=params,
        )
        return next_step

    def _format_server_info(self, server_name: str) -> str:
        """Format server information for display to planners"""
        server_config = self.server_registry.get_server_config(server_name)
        server_str = f"Server Name: {server_name}"
        if not server_config:
            return server_str

        description = server_config.description
        if description:
            server_str = f"{server_str}\nDescription: {description}"

        return server_str

    def _format_agent_info(self, agent_name: str) -> str:
        """Format Agent information for display to planners"""
        agent = self.agents.get(agent_name)
        if not agent:
            return ""

        servers = "\n".join(
            [
                f"- {self._format_server_info(server_name)}"
                for server_name in agent.server_names
            ]
        )

        return f"Agent Name: {agent.name}\nDescription: {agent.instruction}\nServers in Agent: {servers}"
