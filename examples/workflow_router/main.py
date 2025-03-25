import asyncio
import os

from rich import print

from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.app import MCPApp
from mcp_agent.logging.logger import get_logger
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.model_factory import ModelFactory
from mcp_agent.workflows.router.router_llm import LLMRouter

app = MCPApp(name="router")


def print_to_console(message: str):
    """
    A simple function that prints a message to the console.
    """
    logger = get_logger("workflow_router.print_to_console")
    logger.info(message)


def print_hello_world():
    """
    A simple function that prints "Hello, world!" to the console.
    """
    print_to_console("Hello, world!")


async def example_usage():
    async with app.run() as router_app:
        logger = router_app.logger
        context = router_app.context
        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        # Create configs for our agents
        finder_config = AgentConfig(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            servers=["fetch", "filesystem"],
            model="gpt-4o",  # Use openai model for finder
            default_request_params=RequestParams(use_history=True),
        )

        writer_config = AgentConfig(
            name="writer",
            instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
            servers=["filesystem"],
            model="gpt-4o",  # Use openai model for writer
            default_request_params=RequestParams(use_history=True),
        )

        reasoning_config = AgentConfig(
            name="reasoner",  # Fixed duplicate name
            instruction="""You are a generalist with knowledge about a vast
            breadth of subjects. You are tasked with analyzing and reasoning over
            the user's query and providing a thoughtful response.""",
            servers=[],
            model="sonnet",  # Use Claude for reasoning
            default_request_params=RequestParams(use_history=True),
        )

        # Create and initialize agents with their LLMs
        finder_agent = Agent(config=finder_config, context=context)
        writer_agent = Agent(config=writer_config, context=context)
        reasoning_agent = Agent(config=reasoning_config, context=context)

        async with finder_agent:
            finder_factory = ModelFactory.create_factory(
                finder_config.model, request_params=finder_config.default_request_params
            )
            finder_agent._llm = await finder_agent.attach_llm(finder_factory)

        async with writer_agent:
            writer_factory = ModelFactory.create_factory(
                writer_config.model, request_params=writer_config.default_request_params
            )
            writer_agent._llm = await writer_agent.attach_llm(writer_factory)

        async with reasoning_agent:
            reasoning_factory = ModelFactory.create_factory(
                reasoning_config.model,
                request_params=reasoning_config.default_request_params,
            )
            reasoning_agent._llm = await reasoning_agent.attach_llm(reasoning_factory)

        # Create a router using OpenAI's GPT-4
        openai_llm_factory = ModelFactory.create_factory("gpt-4o")
        router = await LLMRouter.create(
            llm_factory=openai_llm_factory,
            agents=[finder_agent, writer_agent, reasoning_agent],
            functions=[print_to_console, print_hello_world],
        )

        # This should route the query to finder agent, and also give an explanation of its decision
        results = await router.route_to_agent(
            request="Print the contents of mcp_agent.config.yaml verbatim", top_k=1
        )
        logger.info("Router Results:", data=results)

        # We can use the agent returned by the router
        agent = results[0].result
        async with agent:
            result = await agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            result = await agent.call_tool(
                name="read_file",
                arguments={
                    "path": str(os.path.join(os.getcwd(), "mcp_agent.config.yaml"))
                },
            )
            logger.info("read_file result:", data=result.model_dump())

        # Create another router using Anthropic's Claude
        anthropic_llm_factory = ModelFactory.create_factory("sonnet")
        anthropic_router = await LLMRouter.create(
            llm_factory=anthropic_llm_factory,
            server_names=["fetch", "filesystem"],
            agents=[finder_agent, writer_agent, reasoning_agent],
            functions=[print_to_console, print_hello_world],
        )

        # This should route the query to print_to_console function
        # Note that even though top_k is 2, it should only return print_to_console and not print_hello_world
        results = await anthropic_router.route_to_function(
            request="Print the input to console", top_k=2
        )
        logger.info("Router Results:", data=results)
        function_to_call = results[0].result
        function_to_call("Hello, world!")

        # This should route the query to fetch MCP server (inferring just by the server name alone!)
        # You can also specify a server description in mcp_agent.config.yaml to help the router make a more informed decision
        results = await anthropic_router.route_to_server(
            request="Print the first two paragraphs of https://www.anthropic.com/research/building-effective-agents",
            top_k=1,
        )
        logger.info("Router Results:", data=results)

        # Using the 'route' function will return the top-k results across all categories the router was initialized with (servers, agents and callables)
        # top_k = 3 should likely print: 1. filesystem server, 2. finder agent and possibly 3. print_to_console function
        results = await anthropic_router.route(
            request="Print the contents of mcp_agent.config.yaml verbatim",
            top_k=3,
        )
        logger.info("Router Results:", data=results)


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
