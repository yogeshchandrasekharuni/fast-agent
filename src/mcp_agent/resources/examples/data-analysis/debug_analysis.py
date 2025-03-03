import asyncio

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)
from mcp_agent.core.proxies import WorkflowProxy

# Create the application
fast = FastAgent("Data Analysis & Campaign Generator Debug")


# Simple agents for testing
@fast.agent(
    name="generator",
    instruction="Test generator",
    servers=["filesystem"],
    model="sonnet",
)
@fast.agent(
    "evaluator",
    "Test evaluator",
    model="gpt-4o",
)
@fast.evaluator_optimizer(
    "analysis_tool",
    generator="generator",
    evaluator="evaluator",
    max_refinements=3,
    min_rating="EXCELLENT",
)
async def main():
    # Debug the workflow creation
    async with fast.run() as agent:
        # The agents are available as attributes on the agent object
        # and internally stored in the _proxies dictionary
        print("DEBUGGING AGENT STRUCTURE:")
        print(f"Agent app type: {type(agent)}")

        # Inspect available proxies
        print("\nAvailable proxies:")
        for name in dir(agent):
            if not name.startswith("_"):
                print(f"  {name}")

        # In AgentApp, agents are stored in the _proxies dictionary
        if hasattr(agent, "_proxies"):
            print("\nDEBUGGING AGENT PROXIES:")
            for name, proxy in agent._proxies.items():
                print(f"Proxy: {name}, Type: {type(proxy)}")

                if isinstance(proxy, WorkflowProxy) and isinstance(
                    proxy._workflow, EvaluatorOptimizerLLM
                ):
                    eo_llm = proxy._workflow
                    print(f"  EvaluatorOptimizerLLM found: {name}")
                    print(f"  Object name: {eo_llm.name}")
                    print(f"  Dict: {sorted(eo_llm.__dict__.keys())}")

                    # Try to find where name might be getting lost
                    generator = eo_llm.generator
                    print(f"  Generator type: {type(generator)}")
                    print(f"  Generator name: {generator.name}")

        # Try to access our workflow - will it work?
        print("\nCHECKING OUR WORKFLOW:")
        try:
            workflow = agent.analysis_tool
            print(f"Got workflow by name 'analysis_tool': {workflow}")
            print(f"Workflow's type: {type(workflow)}")

            # For WorkflowProxy, the actual workflow is in _workflow attribute
            if hasattr(workflow, "_workflow"):
                print(f"Inner workflow type: {type(workflow._workflow)}")
                print(f"Inner workflow name: {workflow._workflow.name}")
        except Exception as e:
            print(f"Error getting workflow: {str(e)}")

        # Check the name in pre-initialization state
        print("\nCHECKING AGENT REGISTRY:")
        for name, agent_data in fast.agents.items():
            if agent_data["type"] == "evaluator_optimizer":
                print(f"Registered agent: {name}")
                print(f"  Type: {agent_data['type']}")
                print(f"  Generator: {agent_data.get('generator')}")
                print(f"  Evaluator: {agent_data.get('evaluator')}")

        # Look directly at the MCPApp structure
        print("\nDEBUGGING MCPAPP STRUCTURE:")
        if hasattr(agent, "_agent_app"):
            app = agent._agent_app
            print(f"App type: {type(app)}")

            # Check workflows dict
            if hasattr(app, "_workflows"):
                print("Workflows in MCPApp:")
                for wf_name, workflow in app._workflows.items():
                    print(f"  {wf_name}: {type(workflow)}")
                    if isinstance(workflow, EvaluatorOptimizerLLM):
                        print(f"    Name attribute: {workflow.name}")


if __name__ == "__main__":
    asyncio.run(main())
