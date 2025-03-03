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
    format_step_result_text,
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
    SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE,  # Add the missing import
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
            max_iterations=5,  # Reduced from 10 to prevent excessive iterations
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
        self.name = name

        # Store agents by name - COMPLETE REWRITE OF AGENT STORAGE
        self.agents = {}
        for agent in available_agents:
            # Fix: Remove all special handling of agent names and store them exactly as they are
            agent_name = agent.name

            # Verify if the name is actually "None" (string) or None (NoneType)
            if agent_name == "None":
                # Try to get a better name from config if available
                if hasattr(agent, "config") and agent.config and agent.config.name:
                    agent_name = agent.config.name
            elif agent_name is None:
                # Try to get a better name from config if available
                if hasattr(agent, "config") and agent.config and agent.config.name:
                    agent_name = agent.config.name
                else:
                    agent_name = f"unnamed_agent_{len(self.agents)}"

            self.logger.info(f"Adding agent '{agent_name}' to orchestrator")
            self.agents[agent_name] = agent

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
        import json
        from pydantic import ValidationError

        params = self.get_request_params(request_params)
        result_str = await self.generate_str(message=message, request_params=params)

        try:
            # Directly parse JSON and create model instance
            parsed_data = json.loads(result_str)
            return response_model(**parsed_data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Log the error and fall back to the original method if direct parsing fails
            self.logger.error(
                f"Direct JSON parsing failed: {str(e)}. Falling back to standard method."
            )
            self.logger.debug(f"Failed JSON content: {result_str}")

            # Use AugmentedLLM's structured output handling as fallback
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
        total_steps_executed = 0

        params = self.get_request_params(request_params)
        max_steps = getattr(
            params, "max_steps", params.max_iterations * 5
        )  # Default to 5× max_iterations

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
        plan_result.max_iterations_reached = (
            False  # Add a flag to track if we hit the limit
        )

        while iterations < params.max_iterations:
            if self.plan_type == "iterative":
                # Get next plan/step
                next_step = await self._get_next_step(
                    objective=objective, plan_result=plan_result, request_params=params
                )
                logger.debug(f"Iteration {iterations}: Iterative plan:", data=next_step)
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
                # Validate agent names in the plan early
                self._validate_agent_names(plan)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(
                    objective=objective, plan_result=plan_result, request_params=params
                )
                logger.debug(f"Iteration {iterations}: Full Plan:", data=plan)
                # Validate agent names in the plan early
                self._validate_agent_names(plan)
            else:
                raise ValueError(f"Invalid plan type {self.plan_type}")

            plan_result.plan = plan

            if plan.is_complete:
                # Modified: Remove the requirement for steps to be executed
                plan_result.is_complete = True

                # Synthesize final result into a single message
                # Use the structured XML format for better context
                synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                    plan_result=format_plan_result(plan_result)
                )

                # Use planner directly - planner already has PLANNING verb
                plan_result.result = await self.planner.generate_str(
                    message=synthesis_prompt,
                    request_params=params.model_copy(update={"max_iterations": 1}),
                )

                return plan_result

            # Execute each step, collecting results
            # Note that in iterative mode this will only be a single step
            for step in plan.steps:
                # Check if we've hit the step limit
                if total_steps_executed >= max_steps:
                    self.logger.warning(
                        f"Reached maximum step limit ({max_steps}) without completing objective."
                    )
                    plan_result.max_steps_reached = True
                    break

                step_result = await self._execute_step(
                    step=step,
                    previous_result=plan_result,
                    request_params=params,
                )

                plan_result.add_step_result(step_result)
                total_steps_executed += 1

            # Check for step limit after executing steps
            if total_steps_executed >= max_steps:
                plan_result.max_iterations_reached = True
                break

            logger.debug(
                f"Iteration {iterations}: Intermediate plan result:", data=plan_result
            )

            # Check for diminishing returns
            if iterations > 2 and len(plan.steps) <= 1:
                # If plan has 0-1 steps after multiple iterations, might be done
                self.logger.info("Minimal new steps detected, marking plan as complete")
                plan_result.is_complete = True
                break

            iterations += 1

        # If we get here, we've hit the iteration limit without completing
        self.logger.warning(
            f"Failed to complete in {params.max_iterations} iterations."
        )

        # Mark that we hit the iteration limit
        plan_result.max_iterations_reached = True

        # Synthesize what we have so far, but use a different prompt that explains the incomplete status
        synthesis_prompt = SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE.format(
            plan_result=format_plan_result(plan_result),
            max_iterations=params.max_iterations,
        )

        # Generate a final synthesis that acknowledges the incomplete status
        plan_result.result = await self.planner.generate_str(
            message=synthesis_prompt,
            request_params=params.model_copy(update={"max_iterations": 1}),
        )

        return plan_result

    async def _execute_step(
        self,
        step: Step,
        previous_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> StepResult:
        """Execute a step's subtasks in parallel and synthesize results"""

        step_result = StepResult(step=step, task_results=[])
        # Use structured XML format for context to help agents better understand the context
        context = format_plan_result(previous_result)

        # Execute tasks
        futures = []
        error_tasks = []

        for task in step.tasks:
            # Make sure we're using a valid agent name
            agent = self.agents.get(task.agent)
            if not agent:
                # Log a more prominent error - this is a serious problem that shouldn't happen
                # with the improved prompt
                self.logger.error(
                    f"AGENT VALIDATION ERROR: No agent found matching '{task.agent}'. Available agents: {list(self.agents.keys())}"
                )
                error_tasks.append(
                    (
                        task,
                        f"Error: Agent '{task.agent}' not found. This indicates a problem with the plan generation. Available agents: {', '.join(self.agents.keys())}",
                    )
                )
                continue

            task_description = TASK_PROMPT_TEMPLATE.format(
                objective=previous_result.objective,
                task=task.description,
                context=context,
            )

            # Handle both Agent objects and AugmentedLLM objects
            from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

            if isinstance(agent, AugmentedLLM):
                futures.append(agent.generate_str(message=task_description))
            else:
                # Traditional Agent objects with _llm property
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
                # Create a TaskWithResult that includes the agent name for attribution
                task_model = task.model_dump()
                task_result = TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],  # Track which agent produced this result
                    result=str(result),
                )
                step_result.add_task_result(task_result)
                task_index += 1

        # Add error task results
        for task, error_message in error_tasks:
            task_model = task.model_dump()
            step_result.add_task_result(
                TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],
                    result=f"ERROR: {error_message}",
                )
            )

        # Use text formatting for display in logs
        step_result.result = format_step_result_text(step_result)
        return step_result

    async def _get_full_plan(
        self,
        objective: str,
        plan_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> Plan:
        """Generate full plan considering previous results"""
        import json
        from pydantic import ValidationError
        from mcp_agent.workflows.orchestrator.orchestrator_models import (
            Plan,
            Step,
            AgentTask,
        )

        params = self.get_request_params(request_params)
        params = params.model_copy(update={"use_history": False})

        # Format agents without numeric prefixes for cleaner XML
        agent_formats = []
        for agent_name in self.agents.keys():
            formatted = self._format_agent_info(agent_name)
            agent_formats.append(formatted)

        agents = "\n".join(agent_formats)

        # Create clear plan status indicator for the template
        plan_status = "Plan Status: Not Started"
        if hasattr(plan_result, "is_complete"):
            plan_status = (
                "Plan Status: Complete"
                if plan_result.is_complete
                else "Plan Status: In Progress"
            )

        # Fix the iteration counting display
        max_iterations = params.max_iterations
        # Simplified iteration counting logic
        current_iteration = len(plan_result.step_results)
        current_iteration = min(current_iteration, max_iterations - 1)  # Cap at max-1
        iterations_remaining = max(
            0, max_iterations - current_iteration - 1
        )  # Ensure non-negative
        iterations_info = f"Planning Budget: Iteration {current_iteration + 1} of {max_iterations} (with {iterations_remaining} remaining)"

        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get raw JSON response from LLM
        result_str = await self.planner.generate_str(
            message=prompt,
            request_params=params,
        )

        try:
            # Parse JSON directly
            data = json.loads(result_str)

            # Create models manually to ensure agent names are preserved exactly as returned
            steps = []
            for step_data in data.get("steps", []):
                tasks = []
                for task_data in step_data.get("tasks", []):
                    # Create AgentTask directly from dict, preserving exact agent string
                    task = AgentTask(
                        description=task_data.get("description", ""),
                        agent=task_data.get("agent", ""),  # Preserve exact agent name
                    )
                    tasks.append(task)

                # Create Step with the exact task objects we created
                step = Step(description=step_data.get("description", ""), tasks=tasks)
                steps.append(step)

            # Create final Plan
            plan = Plan(steps=steps, is_complete=data.get("is_complete", False))

            return plan

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            # Log detailed error and fall back to the original method as last resort
            self.logger.error(f"Error parsing plan JSON: {str(e)}")
            self.logger.debug(f"Failed JSON content: {result_str}")

            # Use the normal structured parsing as fallback
            plan = await self.planner.generate_structured(
                message=result_str,
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
        import json
        from pydantic import ValidationError
        from mcp_agent.workflows.orchestrator.orchestrator_models import (
            NextStep,
            AgentTask,
        )

        params = self.get_request_params(request_params)
        params = params.model_copy(update={"use_history": False})

        # Format agents without numeric prefixes for cleaner XML
        # FIX: Iterate over agent names instead of agent objects
        agents = "\n".join(
            [self._format_agent_info(agent_name) for agent_name in self.agents.keys()]
        )

        # Create clear plan status indicator for the template
        plan_status = "Plan Status: Not Started"
        if hasattr(plan_result, "is_complete"):
            plan_status = (
                "Plan Status: Complete"
                if plan_result.is_complete
                else "Plan Status: In Progress"
            )

        # Add max_iterations info for the LLM
        max_iterations = params.max_iterations
        current_iteration = len(plan_result.step_results)
        iterations_remaining = max_iterations - current_iteration
        iterations_info = f"Planning Budget: {iterations_remaining} of {max_iterations} iterations remaining"

        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get raw JSON response from LLM
        result_str = await self.planner.generate_str(
            message=prompt,
            request_params=params,
        )

        try:
            # Parse JSON directly
            data = json.loads(result_str)

            # Create task objects manually to preserve exact agent names
            tasks = []
            for task_data in data.get("tasks", []):
                # Preserve the exact agent name as specified in the JSON
                task = AgentTask(
                    description=task_data.get("description", ""),
                    agent=task_data.get("agent", ""),
                )
                tasks.append(task)

            # Create step with manually constructed tasks
            next_step = NextStep(
                description=data.get("description", ""),
                tasks=tasks,
                is_complete=data.get("is_complete", False),
            )

            return next_step

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            # Log detailed error and fall back to the original method
            self.logger.error(f"Error parsing next step JSON: {str(e)}")
            self.logger.debug(f"Failed JSON content: {result_str}")

            # Use the normal structured parsing as fallback
            next_step = await self.planner.generate_structured(
                message=result_str,
                response_model=NextStep,
                request_params=params,
            )

            return next_step

    def _format_server_info(self, server_name: str) -> str:
        """Format server information for display to planners using XML tags"""
        from mcp_agent.workflows.llm.prompt_utils import format_server_info

        server_config = self.server_registry.get_server_config(server_name)

        # Get description or empty string if not available
        description = ""
        if server_config and server_config.description:
            description = server_config.description

        return format_server_info(server_name, description)

    def _validate_agent_names(self, plan: Plan) -> None:
        """
        Validate all agent names in a plan before execution.
        This helps catch invalid agent references early.
        """
        invalid_agents = []

        for step in plan.steps:
            for task in step.tasks:
                if task.agent not in self.agents:
                    invalid_agents.append(task.agent)

        if invalid_agents:
            available_agents = ", ".join(self.agents.keys())
            invalid_list = ", ".join(invalid_agents)
            error_msg = f"Plan contains invalid agent names: {invalid_list}. Available agents: {available_agents}"
            self.logger.error(error_msg)
            # We don't raise an exception here as the execution will handle invalid agents
            # by logging errors for individual tasks

    def _format_agent_info(self, agent_name: str) -> str:
        """Format Agent information for display to planners using XML tags"""
        from mcp_agent.workflows.llm.prompt_utils import format_agent_info

        agent = self.agents.get(agent_name)
        if not agent:
            self.logger.error(f"Agent '{agent_name}' not found in orchestrator agents")
            return ""
        instruction = agent.instruction

        # Get servers information
        server_info = []
        for server_name in agent.server_names:
            server_config = self.server_registry.get_server_config(server_name)
            description = ""
            if server_config and server_config.description:
                description = server_config.description

            server_info.append({"name": server_name, "description": description})

        return format_agent_info(
            agent.name, instruction, server_info if server_info else None
        )
