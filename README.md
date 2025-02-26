## FastAgent

<p align="center">
<a href="https://pypi.org/project/fast-agent-mcp/"><img src="https://img.shields.io/pypi/v/fast-agent-mcp?color=%2334D058&label=pypi" /></a>
<a href="https://github.com/evalstate/fast-agent/issues"><img src="https://img.shields.io/github/issues-raw/evalstate/fast-agent" /></a>
<a href="https://lmai.link/discord/mcp-agent"><img src="https://shields.io/discord/1089284610329952357" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/fast-agent-mcp?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/evalstate/fast-agent-mcp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/fast-agent-mcp" /></a>
</p>

## Overview

**`fast-agent`** lets you build and interact with Agents and Workflows in minutes.

The simple declarative syntax lets you concentrate on composing your Prompts and MCP Servers to [build effective agents](https://www.anthropic.com/research/building-effective-agents).

Evaluate how different models handle Agent and MCP Server calling tasks, then build multi-model workflows using the best provider for each task.

### Agent Application Development

Prompts and configurations that define your Agent Applications are stored in simple files, with minimal boilerplate, enabling simple management and version control.

Chat with individual Agents and Components before, during and after workflow execution to tune and diagnose your agent application. Simple model selection makes testing Model <-> MCP Server interaction painless.

## Get started:

Start by installing the [uv package manager](https://docs.astral.sh/uv/) for Python. Then:

```bash
uv pip install fast-agent-mcp       # install fast-agent
fast-agent setup                    # create an example agent and config files
uv run agent.py                     # run your first agent
uv run agent.py --model=o3-mini.low # specify a model
fast-agent bootstrap workflow       # create "building effective agents" examples
```

Other bootstrap examples include a Researcher Agent (with Evaluator-Optimizer workflow) and Data Analysis Agent (similar to the ChatGPT experience), demonstrating MCP Roots support.

> Windows Users - there are a couple of configuration changes needed for the Filesystem and Docker MCP Servers - necessary changes are detailed within the configuration files.

### Basic Agents

Defining an agent is as simple as:

```python
@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
```

We can then send messages to the Agent:

```python
async with fast.run() as agent:
  moon_size = await agent("the moon")
  print(moon_size)
```

Or start an interactive chat with the Agent:
```python
async with fast.run() as agent:  
  await agent()
```

Here is the complete `sizer.py` Agent application, with boilerplate code:
```python
import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)

async def main():
  async with fast.run() as agent:
    await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

The Agent can be run with `uv run sizer.py` and with a specific model using the command line option `--model gpt-4o-mini`.

### Combining Agents and using MCP Servers

_To generate examples use `fast-agent bootstrap workflow`. This example can be run with `uv run chaining.py`. fast-agent looks for configuration files in the current directory before checking parent directories recursively._

Agents can be chained to build a workflow:
```python
@fast.agent(
    "url_fetcher",
    instruction="Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"], # Name of an MCP Server defined in fastagent.config.yaml
)
@fast.agent(
    "social_media",
    instruction="""
    Write a 280 character social media post for any given text.
    Respond only with the post, never use hashtags.
    """,
)

async def main():
    async with fast.run() as agent:
        await agent.social_media(
            await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        )
```

All Agents and Workflows respond to `.send("message")` to send a message and `.prompt()` to begin a chat session. 

## Workflows

### Chain

Alternatively, use the `chain` workflow type and the `prompt()` method to capture user input:
```python

@fast.chain(
  "post_writer",
  sequence=["url_fetcher","social_media"]
)

# we can them prompt it directly:
async with fast.run() as agent:
  await agent.post_writer.prompt()

```
Chains can be incorporated in other workflows, or contain other workflow elements (including other Chains). You can set an `instruction` to precisely describe it's capabilities to other workflow steps if needed.

### Parallel

The Parallel Workflow sends the same message to multiple Agents simultaneously (`fan-out`), then uses the `fan-in` agent to process the combined content. 

```python

@fast.agent(
  name="consolidator"
  instruction="combine the lists, remove duplicates"
)

@fast.parallel(
  name="ensemble"
  fan_out=["agent_o3-mini","agent_sonnet37",...]
  fan_in="consolidator"
)

async with fast.run() as agent:
  result = agent.ensemble.send("what are the 10 most important aspects of project management")
```

Look at the `parallel.py` workflow example for more details.

### Evaluator-Optimizer

Evaluator-Optimizers use 2 agents: one to generate content (the `generator`), and one to judge the content and provide actionable feedback (the `evaluator`). Messages are sent to the generator first, then the pair run in a loop until either the evaluator is satisfied with the quality, or the maximum number of refinements is reached.

```python
@fast.evaluator_optimizer(
  name="researcher"
  generator="web_searcher"
  evaluator="quality_assurance"
  min_rating="EXCELLENT"
  max_refinements=3
)

async with fast.run() as agent:
  await agent.researcher.send("produce a report on how to make the perfect espresso")
```

See the `evaluator.py` workflow example, or `fast-agent bootstrap researcher` for a more complete example. 

### Router

Routers use an LLM to assess a message, and route it to the most appropriate Agent direct . The routing prompt is automatically generated by the router.

```python
@fast.router(
  name="route"
  agents["agent1","agent2","agent3"]
)
```

Look at the `router.py` workflow for an example.

### Orchestrator

Given a task, an Orchestrator uses an LLM to generate a plan to divide the task amongst the available agents and aggregate a result. The planning and aggregation prompts are generated by the Orchestrator, which benefits from using more capable models. Plans can either be built once at the beginning (`plantype="full"`) or iteratively (`plantype="iterative"`).  

```python
@fast.orchestrator(
  name="orchestrate"
  agents=["task1","task2","task3"]
)
```

## Agent Features

```python
@fast.agent(
  name="agent",
  instructions="instructions",
  servers=["filesystem"],     # list of MCP Servers for the agent, configured in fastagent.config.yaml
  model="o3-mini.high",       # specify a model for the agent
  use_history=True,           # agent can maintain chat history
  human_input=True,           # agent can request human input
)
```

### Human Input

When `human_input` is set to true for an Agent, it is presented with the option to prompt the User for input.

## Project Notes

`fast-agent` builds on the [`mcp-agent`](https://github.com/lastmile-ai/mcp-agent) project by Sarmad Qadri.

### llmindset.co.uk fork:

- "FastAgent" style prototyping, with per-agent models
- Api keys through Environment Variables
- Warm-up / Post-Workflow Agent Interactions
- Quick Setup
- Interactive Prompt Mode
- Simple Model Selection with aliases
- User/Assistant and Tool Call message display
- MCP Sever Environment Variable support
- MCP Roots support
- Comprehensive Progress display
- JSONL file logging with secret revokation
- OpenAI o1/o3-mini support with reasoning level
- Enhanced Human Input Messaging and Handling
- Declarative workflows

### Features to add.

 - Chat History Clear.

